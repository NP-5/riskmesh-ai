"""Evidence extraction, REVLENS-style rule routing (decision/reason/evidence/confidence), and the
'what the evidence does / does NOT establish' sections."""
RULESET_VERSION = "rules-v1"

NOT_ESTABLISHED = {
    "shared_device": "A shared device does not establish fraud or common control; it may be a household tablet, a corporate kiosk or a reseller device.",
    "shared_ip": "A shared IP /24 does not establish that the customers are controlled by the same person; offices, ISPs and carrier NAT share ranges.",
    "shared_instrument": "A shared payment instrument does not establish intent; family members, corporate cards and wallets are legitimately shared.",
    "beneficiary_concentration": "Beneficiary concentration does not establish malicious intent; payroll, vendor and marketplace payouts are legitimately concentrated.",
    "amount_similarity": "Similar amounts do not establish structuring; fixed-price invoices or subscriptions look alike.",
    "temporal_burst": "Timing proximity does not establish coordination; scheduled batch payments also produce bursts.",
    "escalation_signal": "The 6-hour escalation signal is a prototype heuristic, not a prediction that a cash-out will occur.",
    "beneficiary_history": "Beneficiary age/verification status does not establish who controls the beneficiary or why funds were sent.",
    "prior_relationship": "Presence or absence of prior history does not establish legitimacy or illegitimacy of the current transfers.",
    "conversation": "Conversation signals are language-based hints from short messages; they do not establish intent, identity, or that any transaction occurred.",
}


def build_evidence(ring: dict, benes: dict, convs: list[dict]) -> list[dict]:
    ev = []

    def add(kind, family, statement, polarity, strength, refs=None):
        ev.append({"id": f"E{len(ev) + 1}", "kind": kind, "family": family, "statement": statement,
                   "polarity": polarity, "strength": round(float(strength), 2), "source_refs": refs or []})

    n = ring["max_on_device"]
    if n >= 2:
        add("shared_device", "graph", f"{n} customers share device {ring['top_device']}.",
            "supports" if n >= 3 else "neutral", min(n / 4, 1) * 0.9 if n >= 3 else 0.2)
    add("shared_ip", "graph", f"{ring['max_on_ip']} customers share IP range {ring['top_ip']}.", "supports", 0.25)
    if ring["max_on_inst"] >= 2:
        add("shared_instrument", "graph", f"{ring['max_on_inst']} customers use payment instrument {ring['top_inst']}.",
            "supports", min(ring["max_on_inst"] / 3, 1) * 0.8)
    b, bi = ring["top_beneficiary"], benes.get(ring["top_beneficiary"], {})
    c = ring["concentration"]
    add("beneficiary_concentration", "transaction",
        f"{c:.0%} of members' outgoing value in the last 7 days (INR {ring['exposure_inr']:,.0f}) went to beneficiary {b}.",
        "supports" if c >= 0.6 else "neutral", c * 0.9 if c >= 0.6 else 0.2)
    if ring["amount_cv"] is not None and ring["amount_cv"] < 0.05:
        add("amount_similarity", "transaction", f"Transfers to {b} in the last 24h have near-identical amounts (CV={ring['amount_cv']}).", "supports", 0.6)
    w = ring["window_minutes"]
    if w is not None:
        if w <= 90:
            add("temporal_burst", "temporal", f"60% of transfers to {b} ({ring['n_top_txns']} total) occurred within {w:.0f} minutes.", "supports", 0.5 + 0.4 * (1 - w / 90))
        elif w <= 360:
            add("temporal_burst", "temporal", f"60% of transfers to {b} occurred within {w / 60:.1f} hours.", "supports", 0.35)
        elif w <= 1440:
            add("temporal_burst", "temporal", f"60% of transfers to {b} span {w / 60:.1f} hours (moderate clustering).", "neutral", 0.3)
        else:
            add("temporal_burst", "temporal", f"Transfers to {b} are spread over {w / 1440:.1f} days; no burst pattern.", "contradicts", 0.7)
    e = ring["escalation"]
    add("escalation_signal", "temporal", f"6-hour escalation signal {e:.0f}/100 (prototype signal).",
        "supports" if e >= 60 else "neutral", 0.7 if e >= 60 else 0.2)
    if bi.get("verified"):
        add("beneficiary_history", "history", f"Beneficiary {b} is verified and {bi['age_days']} days old.", "contradicts", 0.9)
    elif bi.get("age_days", 999) < 30:
        add("beneficiary_history", "history", f"Beneficiary {b} is unverified and only {bi['age_days']} days old.", "supports", 0.7)
    else:
        add("beneficiary_history", "history", f"Beneficiary {b} is unverified ({bi.get('age_days', '?')} days old).", "neutral", 0.3)
    if ring["prior_n"] >= 10 and ring["prior_share"] >= 0.5:
        add("prior_relationship", "history", f"{ring['prior_share']:.0%} of members' {ring['prior_n']} transfers older than 7 days went to {b}: established relationship.", "contradicts", 0.8)
    elif ring["prior_share"] < 0.1:
        add("prior_relationship", "history", f"Members had no meaningful transfer history to {b} before the last 7 days.", "supports", 0.4 if ring["prior_n"] < 30 else 0.5)
    for cv in convs:
        ev.append({"id": cv["id"], "kind": "conversation", "family": "conversation", "statement": cv["text"],
                   "polarity": "pending", "strength": 0.0, "source_refs": [cv["id"]], "customer_id": cv.get("customer_id")})
    return ev


def run_rules(ring: dict, evidence: list[dict], benes: dict) -> list[dict]:
    K = {e["kind"]: e for e in evidence if e["kind"] != "conversation"}
    ids = lambda *ks: [K[k]["id"] for k in ks if k in K]
    verified = benes.get(ring["top_beneficiary"], {}).get("verified", False)
    dev, conc, w, esc = ring["max_on_device"], ring["concentration"], ring["window_minutes"], ring["escalation"]
    out = []

    def fire(rule_id, decision, reason, eids, conf):
        assert eids and all(i in {e["id"] for e in evidence} for i in eids), "rule cites evidence that does not exist"
        out.append({"rule_id": rule_id, "rule_version": RULESET_VERSION, "decision": decision, "reason": reason,
                    "evidence": eids, "confidence": conf})

    r1 = dev >= 3 and conc >= 0.6 and w is not None and w <= 90 and esc >= 50 and not verified
    if r1:
        fire("RING_TARGETED_HOLD_CANDIDATE", "HOLD_BENEFICIARY_TRANSFERS",
             "Multiple customers share a device and transfer to a concentrated, unverified beneficiary within a short period with elevated 6-hour escalation.",
             ids("shared_device", "beneficiary_concentration", "temporal_burst", "escalation_signal", "beneficiary_history"), 0.8)
    if verified and ring["prior_share"] >= 0.5 and ring["prior_n"] >= 10:
        fire("VERIFIED_ESTABLISHED_COUNTERPARTY", "NO_ACTION",
             "Shared infrastructure exists, but transfers go to a verified beneficiary with an established payment history.",
             ids("beneficiary_history", "prior_relationship"), 0.75)
    if not r1 and dev >= 3 and conc >= 0.5 and not (verified and ring["prior_share"] >= 0.5):
        fire("MIXED_SIGNALS_REVIEW", "NEEDS_REVIEW",
             "Some structural and behavioural signals are present, but the full hold conditions are not met.",
             ids("shared_device", "beneficiary_concentration", "temporal_burst", "beneficiary_history"), 0.5)
    if not r1 and (dev >= 3 or ring["max_on_ip"] >= 3) and (w is None or w > 1440) and esc < 30:
        fire("SHARED_INFRA_NO_BEHAVIOUR", "MONITOR",
             "Shared infrastructure without burst behaviour or escalation.",
             ids("shared_device", "shared_ip", "temporal_burst", "escalation_signal"), 0.7)
    order = ["RING_TARGETED_HOLD_CANDIDATE", "VERIFIED_ESTABLISHED_COUNTERPARTY", "MIXED_SIGNALS_REVIEW", "SHARED_INFRA_NO_BEHAVIOUR"]
    out.sort(key=lambda r: order.index(r["rule_id"]))
    return out


def establishes_and_not(evidence: list[dict]) -> tuple[list[dict], list[str]]:
    established = [{"statement": e["statement"], "evidence": [e["id"]]} for e in evidence
                   if e["kind"] not in ("conversation",) and e["polarity"] != "pending"]
    kinds = {e["kind"] for e in evidence}
    nots = [t for k, t in NOT_ESTABLISHED.items() if k in kinds]
    nots.append("None of the evidence establishes that any account holder committed fraud; RiskMesh output is a risk recommendation for human review.")
    return established, nots
