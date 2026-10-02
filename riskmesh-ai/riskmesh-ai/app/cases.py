"""Case lifecycle: detection -> evidence -> agents -> simulation -> human decision -> audit/export."""
import hashlib, json, time
from collections import Counter
import pandas as pd
from . import datagen, graph
from .ai import ALLOWED_ACTIONS, get_provider, guarded, simulate, validate_critic, validate_fusion, validate_signals
from .db import DATA_DIR, audit, now_iso, read_audit, reset_db
from .evidence import RULESET_VERSION, build_evidence, establishes_and_not, run_rules
from .ingest import ingest_events, load_active_payments, snapshot_stats
from .rules_engine import RULES, evaluate_rule

BASELINE = {  # carried over from the existing Abuse-Ring Sentinel synthetic baseline (as supplied by the project owner)
    "label": "Synthetic Prototype Evaluation (existing Abuse-Ring Sentinel baseline)",
    "ring_detection_precision": 0.839, "ring_detection_recall": 0.833, "avg_lead_time_hours": 4.1,
    "cashout_prediction_precision": 0.233, "legitimate_customers_affected": 15}
EXPECTED = {"ring": {"HOLD_BENEFICIARY_TRANSFERS"}, "legit": {"NO_ACTION", "MONITOR"}, "ambiguous": {"NEEDS_REVIEW"}}


class Conflict(Exception): pass


class State:
    def __init__(self):
        self.df = None; self.benes = {}; self.convs = []; self.labels = {}; self.cases = {}; self.sigs = {}
        self.next_id = 101; self.now = pd.Timestamp(datagen.NOW); self.counts = {}; self.timings = []


S = State()


def _case(cid):
    if cid not in S.cases: raise KeyError(cid)
    return S.cases[cid]


# ------------------------------------------------------------------ detection
def generate_demo(seed=7) -> dict:
    reset_db(); S.__init__()
    data = datagen.generate(seed)
    S.benes, S.convs, S.labels, S.counts = data["beneficiaries"], data["conversations"], data["labels"], data["counts"]
    audit("SYSTEM", "demo_generator", "DEMO_GENERATED", "Synthetic dataset generated", {"seed": seed, **S.counts})
    ingest_events(data["events"])
    return run_detection()


def run_detection() -> dict:
    S.df = load_active_payments()
    rings, _ = graph.detect_rings(S.df, S.now)
    audit("SYSTEM", "risk_engine", "RING_DETECTION_RUN", "DBSCAN + graph closure", {"candidate_rings": len(rings)})
    known = {frozenset(c["members"]) for c in S.cases.values()}
    for members in rings:
        if frozenset(members) in known: continue
        sig = graph.ring_signals(S.df, members, S.now, S.benes)
        convs = [c for c in S.convs if c["customer_id"] in members]
        ev = build_evidence(sig, S.benes, convs)
        rules = run_rules(sig, ev, S.benes)
        cid = f"CASE_{S.next_id}"; S.next_id += 1
        lab = Counter(S.labels.get(m, "normal") for m in members).most_common(1)[0][0]
        S.cases[cid] = {"case_id": cid, "created_at": now_iso(), "demo_scenario": lab, "status": "PENDING_REVIEW", "ring": sig,
                        "evidence": ev, "rule_results": rules, "conversation_signals": None, "fusion": None, "critic": None,
                        "simulation": None, "recommendation": None, "human": None, "decision_history": [], "comments": [],
                        "ai_provider": {}, "investigation_ms": 0.0, "conversations": convs, "decision_version": 0}
        audit(cid, "risk_engine", "CASE_CREATED", "Candidate coordinated ring detected", {"members": members})
        for r in rules:
            audit(cid, "rule_engine", "RULE_EXECUTED", r["reason"], {"rule": r["rule_id"], "decision": r["decision"], "evidence": r["evidence"]})
    return {"cases": len(S.cases), "scenarios": scenario_map()}


def scenario_map():
    out = {}
    for cid, c in S.cases.items(): out.setdefault(c["demo_scenario"], cid)
    return out


# ------------------------------------------------------------------ evidence view
def case_evidence(c) -> list[dict]:
    """Base evidence with conversation items resolved from the Conversation Agent's signals."""
    sigs = c["conversation_signals"] or []
    out = []
    for e in c["evidence"]:
        if e["kind"] == "conversation":
            mine = [s for s in sigs if e["id"] in s["evidence"]]
            if mine:
                top = max(mine, key=lambda s: s["confidence"] * (1 if s["polarity"] != "neutral" else .5))
                e = {**e, "polarity": top["polarity"], "strength": top["confidence"], "signal": top["signal"]}
            elif c["conversation_signals"] is not None:
                e = {**e, "polarity": "neutral", "strength": 0.0}
        out.append(e)
    return out


def _ctx(c): return {"case_id": c["case_id"], "evidence": case_evidence(c), "rule_results": c["rule_results"],
                     "scores": {"structural": c["ring"]["structural"], "escalation": c["ring"]["escalation"],
                                "max_on_device": c["ring"]["max_on_device"], "max_on_ip": c["ring"]["max_on_ip"]}}


# ------------------------------------------------------------------ agents
def run_conversation_agent(c, provider=None):
    provider = provider or get_provider()
    convs = c["conversations"]
    sigs, used, fb = guarded(provider, "extract_conversation_signals", validate_signals, {x["id"] for x in convs}, convs)
    return sigs, used, fb


def run_fusion_agent(c, provider=None):
    provider = provider or get_provider(); ctx = _ctx(c)
    return guarded(provider, "fuse", validate_fusion, {e["id"] for e in ctx["evidence"]}, ctx)


def run_critic_agent(c, fusion, provider=None):
    provider = provider or get_provider(); ctx = _ctx(c)
    return guarded(provider, "critique", validate_critic, {e["id"] for e in ctx["evidence"]}, ctx, fusion)


def investigate(cid) -> dict:
    c = _case(cid); t0 = time.perf_counter(); p = get_provider()
    for n in c["comments"]:   # analyst notes are additional conversation context
        if not any(x["id"] == n["id"] for x in c["conversations"]):
            c["conversations"].append({"id": n["id"], "customer_id": None, "ts": n["timestamp"], "channel": "analyst_note", "text": n["text"]})
            c["evidence"].append({"id": n["id"], "kind": "conversation", "family": "conversation", "statement": n["text"],
                                  "polarity": "pending", "strength": 0.0, "source_refs": [n["id"]]})
    sigs, u1, f1 = run_conversation_agent(c, p)
    c["conversation_signals"] = sigs
    audit(cid, "conversation_agent", "CONVERSATION_ANALYZED", f"{len(sigs)} contextual signals (provider={u1})", {"signals": sigs, "fallback": f1})
    fusion, u2, f2 = run_fusion_agent(c, p)
    c["fusion"] = fusion; c["critic"] = c["simulation"] = c["recommendation"] = None
    c["ai_provider"] = {"conversation": u1, "fusion": u2, "fallbacks": [x for x in (f1, f2) if x]}
    c["investigation_ms"] = (time.perf_counter() - t0) * 1000
    audit(cid, "fusion_agent", "AI_INVESTIGATION_COMPLETED", fusion["risk_summary"], {"confidence": fusion["confidence"], "provider": u2})
    return fusion


def critic(cid) -> dict:
    c = _case(cid)
    if not c["fusion"]: investigate(cid)
    t0 = time.perf_counter()
    out, used, fb = run_critic_agent(c, c["fusion"], get_provider())
    c["critic"] = out; c["simulation"] = c["recommendation"] = None
    c["ai_provider"]["critic"] = used; c["investigation_ms"] += (time.perf_counter() - t0) * 1000
    audit(cid, "critic_agent", "CRITIC_RESULT", out["alternative_explanation"], {"verdict": out["verdict"], "fallback": fb})
    return out


def run_simulation(cid) -> dict:
    c = _case(cid)
    if not c["critic"]: critic(cid)
    t0 = time.perf_counter()
    sim = simulate(c["ring"], c["fusion"], c["critic"])
    c["simulation"] = sim; c["investigation_ms"] += (time.perf_counter() - t0) * 1000
    c["recommendation"] = {"action": sim["recommended_action"], "human_review_required": sim["human_review_required"],
                           "rationale": sim["rationale"], "rule_version": RULESET_VERSION, "timestamp": now_iso(),
                           "automated": True, "scope": f"beneficiary {c['ring']['top_beneficiary']} only; no account freeze"
                           if sim["recommended_action"] == "HOLD_BENEFICIARY_TRANSFERS" else "case-level"}
    audit(cid, "intervention_engine", "RECOMMENDATION", sim["rationale"], {"action": sim["recommended_action"]})
    S.timings.append(c["investigation_ms"])
    return sim


# ------------------------------------------------------------------ human governance
def decide(cid, kind, actor, decision, reason) -> dict:
    c = _case(cid); rec = c["recommendation"]
    if not rec: raise Conflict("run investigate -> critic -> simulate before a human decision")
    if c["status"] == "RESOLVED": raise Conflict("case already resolved")
    auto = rec["action"]
    if kind == "reject" and not decision:
        decision = "NO_ACTION" if auto == "MONITOR" else "MONITOR"
    human = decision or auto
    if human not in ALLOWED_ACTIONS: raise ValueError(f"decision must be one of {ALLOWED_ACTIONS}")
    override = human != auto
    if kind == "reject" and not override: raise ValueError("a rejection must choose a different action")
    if override and not reason.strip(): raise ValueError("a reason is required to override the automated recommendation")
    c["decision_version"] += 1
    c["human"] = {"automated_decision": auto, "human_decision": human, "human_reason": reason, "timestamp": now_iso(),
                  "actor": actor, "override": override, "decision_version": c["decision_version"]}
    c["decision_history"].append(c["human"]); c["status"] = "RESOLVED"
    audit(cid, actor, "HUMAN_OVERRIDE" if override else "HUMAN_APPROVED", reason, c["human"])
    if human == "HOLD_BENEFICIARY_TRANSFERS":
        audit(cid, "intervention_engine", "INTERVENTION_APPLIED_SIMULATED",
              "Simulation only: no payment system is connected", {"scope": f"beneficiary {c['ring']['top_beneficiary']}"})
    return c["human"]


def comment(cid, actor, text) -> dict:
    c = _case(cid)
    n = {"id": f"NOTE_{cid[-3:]}_{len(c['comments']) + 1}", "actor": actor, "text": text, "timestamp": now_iso()}
    c["comments"].append(n); audit(cid, actor, "ANALYST_COMMENT", text); return n


# ------------------------------------------------------------------ views
def case_summary(c):
    return {"case_id": c["case_id"], "demo_scenario": c["demo_scenario"], "status": c["status"], "members": c["ring"]["n_members"],
            "structural_risk": c["ring"]["structural"], "escalation_signal": c["ring"]["escalation"],
            "top_beneficiary": c["ring"]["top_beneficiary"], "rule_decision": c["rule_results"][0]["decision"] if c["rule_results"] else "NO_ACTION",
            "recommendation": (c["recommendation"] or {}).get("action"), "flagged_by_active_rule": RULES.flags(c["ring"])}


def case_view(cid) -> dict:
    c = _case(cid); ev = case_evidence(c)
    est, nots = establishes_and_not(ev)
    return {**{k: v for k, v in c.items() if k not in ("evidence", "conversations")}, "evidence": ev, "conversations": c["conversations"],
            "established": est, "not_established": nots, "labels": {"structural": "prototype risk signal", "escalation": "prototype risk signal, not a guaranteed prediction"},
            "timeline": graph.timeline(S.df, c["ring"], S.now), "synthetic_notice": "Synthetic data. Not an internal Razorpay system.",
            "active_rule": RULES.active_rule()["id"]}


def case_graph(cid): c = _case(cid); return graph.case_graph(S.df, c["ring"], S.benes)


# ------------------------------------------------------------------ export (REVLENS manifest)
def export_case(cid) -> dict:
    c = _case(cid); view = case_view(cid); d = DATA_DIR / "exports" / cid; d.mkdir(parents=True, exist_ok=True)
    files = {"case.json": {k: view[k] for k in ("case_id", "status", "demo_scenario", "ring", "rule_results", "fusion", "critic", "simulation", "established", "not_established", "comments")},
             "evidence.json": view["evidence"],
             "decision.json": {"automated_decision": c["recommendation"], "human_decision": c["human"], "decision_history": c["decision_history"], "decision_version": c["decision_version"]}}
    for n, obj in files.items(): (d / n).write_text(json.dumps(obj, indent=2, default=str))
    audit(cid, "export_service", "CASE_EXPORTED", "Case export bundle written")
    (d / "audit_log.jsonl").write_text("".join(json.dumps(a, default=str) + "\n" for a in read_audit(cid)))
    refs = sorted(map(tuple, c["ring"]["source_refs"]))
    manifest = {"case_id": cid, "exported_at": now_iso(), "data_snapshot": {**snapshot_stats(), "case_source_events": len(refs),
                "snapshot_sha256": hashlib.sha256(json.dumps(refs).encode()).hexdigest()},
                "rule_version": RULESET_VERSION, "active_rule_registry_version": RULES.active, "ai_provider": c["ai_provider"],
                "evidence_ids": [e["id"] for e in view["evidence"]], "decision_version": c["decision_version"],
                "human_overrides": [h for h in c["decision_history"] if h["override"]],
                "files": {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in [*files, "audit_log.jsonl"]},
                "synthetic_notice": "Synthetic prototype data."}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {"directory": str(d), "manifest": manifest}


# ------------------------------------------------------------------ evaluation & metrics
def evaluate() -> dict:
    p, rows = get_provider(), []
    for c in S.cases.values():
        t0 = time.perf_counter()
        sigs, _, _ = run_conversation_agent(c, p)
        tmp = {**c, "conversation_signals": sigs}
        fusion, _, _ = run_fusion_agent(tmp, p); cr, _, _ = run_critic_agent(tmp, fusion, p)
        sim = simulate(c["ring"], fusion, cr); ms = (time.perf_counter() - t0) * 1000
        allowed = {e["id"] for e in case_evidence(tmp)}
        cited = [i for s in sigs for i in s["evidence"]] + fusion["supporting_evidence"] + fusion["contradicting_evidence"] \
            + cr["supporting_evidence"] + [i for r in c["rule_results"] for i in r["evidence"]]
        claims = [x["evidence"] for x in fusion["claims"]] + [r["evidence"] for r in c["rule_results"]]
        rows.append({"case": c["case_id"], "scenario": c["demo_scenario"], "cited": len(cited), "grounded": sum(i in allowed for i in cited),
                     "claims": len(claims), "unsupported": sum((not x) or any(i not in allowed for i in x) for x in claims),
                     "challenged": cr["challenged"], "action": sim["recommended_action"], "ms": ms,
                     "agree": sim["recommended_action"] in EXPECTED.get(c["demo_scenario"], set())})
    scen = [r for r in rows if r["scenario"] in EXPECTED]
    humans = [c["human"] for c in S.cases.values() if c["human"]]
    tot = lambda k: sum(r[k] for r in rows)
    return {"label": "Measured on this run's synthetic cases", "cases_evaluated": len(rows),
            "evidence_grounding": round(tot("grounded") / max(tot("cited"), 1), 3),
            "unsupported_claim_rate": round(tot("unsupported") / max(tot("claims"), 1), 3),
            "critic_challenge_rate": round(sum(r["challenged"] for r in rows) / max(len(rows), 1), 3),
            "recommendation_agreement_with_scenario_label": round(sum(r["agree"] for r in scen) / max(len(scen), 1), 3) if scen else None,
            "human_override_rate": round(sum(h["override"] for h in humans) / len(humans), 3) if humans else None,
            "human_decisions_recorded": len(humans), "avg_investigation_ms": round(tot("ms") / max(len(rows), 1), 2),
            "provider": p.name, "per_case": rows,
            "ring_detection_on_demo": {"scenarios_found": sorted({c["demo_scenario"] for c in S.cases.values()} & set(EXPECTED)), "scenarios_injected": sorted(EXPECTED)},
            "baseline": BASELINE}


def metrics() -> dict:
    cs = list(S.cases.values()); ev = evaluate() if cs else {}
    bt = {"after": evaluate_rule(RULES.active_rule(), RULES.history)} if cs else {}
    bens = Counter()
    for c in cs: bens[c["ring"]["top_beneficiary"]] += c["ring"]["exposure_inr"]
    return {"synthetic_notice": "All values derive from synthetic data; none are Razorpay statistics.",
            "active_cases": sum(c["status"] != "RESOLVED" for c in cs),
            "high_risk_networks": sum(RULES.flags(c["ring"]) for c in cs),
            "pending_reviews": sum(c["status"] != "RESOLVED" and (c["recommendation"] or {}).get("human_review_required", True) for c in cs),
            "false_positive_rate_active_rule_backtest": bt.get("after", {}).get("false_positive_rate"),
            "avg_investigation_seconds": round(sum(S.timings) / len(S.timings) / 1000, 3) if S.timings else None,
            "beneficiary_concentration": [{"beneficiary": b, "exposure_inr": v} for b, v in bens.most_common(5)],
            "recent_escalations": sorted([{"case_id": c["case_id"], "escalation": c["ring"]["escalation"]} for c in cs], key=lambda x: -x["escalation"])[:5],
            "ai_recommendation_agreement": ev.get("recommendation_agreement_with_scenario_label"),
            "active_rule_version": RULES.active, "dataset": {**S.counts, **snapshot_stats()}, "baseline": BASELINE}
