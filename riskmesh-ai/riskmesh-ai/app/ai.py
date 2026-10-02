"""AI layer. AIProvider -> MockProvider (deterministic, default) | LLMProvider (any OpenAI-compatible endpoint).
All model output is validated against the supplied evidence IDs; ungrounded output is rejected and the
deterministic provider is used instead. The LLM never chooses interventions: the simulator is plain code."""
import json, math, os, re, urllib.request

ALLOWED_ACTIONS = ["NO_ACTION", "MONITOR", "STEP_UP_VERIFICATION", "MERCHANT_FOLLOW_UP", "HOLD_BENEFICIARY_TRANSFERS", "NEEDS_REVIEW"]
POLARITIES = {"supports", "contradicts", "neutral"}
VERDICTS = {"CONCLUSION_SURVIVES", "LEGITIMATE_EXPLANATION_LIKELY", "INSUFFICIENT_EVIDENCE"}


# ------------------------------------------------------------------ validators (guardrails)
def _ids_ok(ids, allowed):
    if not isinstance(ids, list) or not all(isinstance(i, str) and i in allowed for i in ids):
        raise ValueError(f"ungrounded evidence ids: {ids}")


def validate_signals(sigs, allowed):
    for s in sigs:
        _ids_ok(s["evidence"], allowed)
        if not s["evidence"] or not 0 <= s["confidence"] <= 1 or s["polarity"] not in POLARITIES:
            raise ValueError("bad signal")


def validate_fusion(f, allowed):
    _ids_ok(f["supporting_evidence"], allowed); _ids_ok(f["contradicting_evidence"], allowed)
    if not 0 <= f["confidence"] <= 1:
        raise ValueError("bad confidence")
    for c in f.get("claims", []):
        if not c["evidence"]:
            raise ValueError("claim without evidence")
        _ids_ok(c["evidence"], allowed)


def validate_critic(c, allowed):
    _ids_ok(c["supporting_evidence"], allowed)
    if c["verdict"] not in VERDICTS:
        raise ValueError("bad verdict")


# ------------------------------------------------------------------ providers
class AIProvider:
    name = "base"
    def extract_conversation_signals(self, conversations): raise NotImplementedError
    def fuse(self, ctx): raise NotImplementedError
    def critique(self, ctx, fusion): raise NotImplementedError


CONV_PATTERNS = [  # (regex, signal, polarity, confidence)
    (r"(alternate|other|another)\s+(merchant\s+)?account", "possible alternate-account usage", "supports", 0.52),
    (r"under (the )?\d+k|below (the )?(limit|threshold)|smaller amounts|split", "possible amount structuring", "supports", 0.78),
    (r"does not get flagged|so (it|we) (does not|do not|don't) get (flagged|blocked)|before .{0,20}(review|block)", "possible evasion language", "supports", 0.7),
    (r"invoice|vendor|payroll|salary|standard process", "business-process explanation", "contradicts", 0.72),
    (r"housemate|household|family|roommate|home router", "plausible shared-household explanation", "contradicts", 0.6),
    (r"not sure|is (it|that) allowed|can we", "ambiguous intent", "neutral", 0.5),
]


class MockProvider(AIProvider):
    name = "mock"

    def extract_conversation_signals(self, conversations):
        out = []
        for cv in conversations:
            text, hits = cv["text"].lower(), []
            for rx, sig, pol, conf in CONV_PATTERNS:
                if re.search(rx, text):
                    hits.append([sig, pol, conf])
            acct = any(h[0] == "possible alternate-account usage" for h in hits)
            if acct and re.search(r"before (tonight|today|midnight|end of day)|tonight|urgent", text):
                hits = [h for h in hits if h[0] != "possible alternate-account usage"] + \
                       [["possible coordinated transfer behavior", "supports", 0.81]]
            for sig, pol, conf in hits:
                out.append({"signal": sig, "evidence": [cv["id"]], "confidence": conf, "polarity": pol,
                            "note": "contextual evidence only; not a fraud determination"})
        return out

    def fuse(self, ctx):
        ev = ctx["evidence"]
        sup = [e for e in ev if e["polarity"] == "supports" and e["strength"] > 0]
        con = [e for e in ev if e["polarity"] == "contradicts"]
        fam = lambda L: {f: max(e["strength"] for e in L if e["family"] == f) for f in {e["family"] for e in L}}
        fs, fc = fam(sup), fam(con)
        score = sum(fs.values()) - 1.2 * sum(fc.values())
        conf = round(1 / (1 + math.exp(-(score - 2.2))), 2)
        kinds = {e["kind"] for e in ev}
        missing = []
        if not any(e["kind"] == "conversation" and e["polarity"] != "pending" and e["strength"] > 0 for e in ev):
            missing.append("No interpretable conversation/contextual evidence for this case.")
        if "shared_device" in kinds: missing.append("Device fingerprint metadata (household / corporate kiosk / reseller) is not available.")
        if "shared_ip" in kinds: missing.append("ISP/ASN type of the shared IP range (office, residential, carrier NAT) is not available.")
        if any(e["kind"] == "beneficiary_history" and e["polarity"] != "contradicts" for e in ev):
            missing.append("Beneficiary KYC / ownership verification detail is not available.")
        missing.append("KYC linkage between the member customers is not available.")
        claims = [{"claim": max((e for e in sup if e["family"] == f), key=lambda e: e["strength"])["statement"],
                   "evidence": [e["id"] for e in sup if e["family"] == f]} for f in fs]
        claims += [{"claim": "Counter-evidence: " + e["statement"], "evidence": [e["id"]]} for e in con]
        if conf >= 0.7: lead = f"Independent signals from {len(fs)} families ({', '.join(sorted(fs))}) point toward coordinated activity"
        elif conf <= 0.3: lead = "Evidence mainly points toward legitimate shared infrastructure"
        else: lead = "Evidence is mixed: some independent signals support coordination while others do not"
        sids = ",".join(e["id"] for e in sup)
        return {"risk_summary": f"{lead} [supporting: {sids or 'none'}]"
                                + (f"; contradicted by [{','.join(e['id'] for e in con)}]" if con else "") + ".",
                "supporting_evidence": [e["id"] for e in sup], "contradicting_evidence": [e["id"] for e in con],
                "missing_information": missing, "confidence": conf, "claims": claims,
                "supporting_families": sorted(fs), "contradicting_families": sorted(fc)}

    def critique(self, ctx, fusion):
        ev, byk = ctx["evidence"], {}
        for e in ev: byk.setdefault(e["kind"], e)
        legit, weak, supp = [], [], []
        def L(kind, text):
            if kind in byk and byk[kind]["polarity"] == "contradicts":
                legit.append(text); supp.append(byk[kind]["id"])
        L("beneficiary_history", "The beneficiary may be a known, verified legitimate counterparty.")
        L("prior_relationship", "The transfers may continue an established payment relationship.")
        L("temporal_burst", "The transfer pattern is spread out and explainable as routine scheduled payments.")
        for e in ev:
            if e["kind"] == "conversation" and e["polarity"] == "contradicts":
                legit.append(f"Conversation {e['id']} offers a benign business/household explanation."); supp.append(e["id"])
        if byk.get("shared_device", {}).get("polarity") == "supports" and ctx["scores"]["max_on_device"] <= 3:
            weak.append("Only 3 customers share the device; a household or small-office device is plausible.")
        if "shared_instrument" not in byk:
            weak.append("No shared payment instrument: the device/IP link alone is weak evidence of common control.")
        if ctx["scores"]["max_on_ip"] >= 3 and "shared_instrument" not in byk:
            weak.append("A shared IP /24 can simply reflect an office network or ISP/NAT.")
        convs = [e for e in ev if e["kind"] == "conversation" and e["polarity"] != "pending"]
        if any(e["polarity"] == "neutral" or e["strength"] < 0.6 for e in convs):
            weak.append("Conversation evidence is ambiguous or low-confidence.")
        strong = {e["family"] for e in ev if e["polarity"] == "supports" and e["strength"] >= 0.5}
        if len(strong) < 4:
            weak.append(f"Only {len(strong)} independent signal families are strong ({', '.join(sorted(strong)) or 'none'}).")
        if len(legit) >= 2:
            verdict, conf, alt = "LEGITIMATE_EXPLANATION_LIKELY", "low", "; ".join(legit)
        elif fusion["confidence"] >= 0.7 and len(weak) <= 1:
            verdict, conf, alt = "CONCLUSION_SURVIVES", "high", "Shared infrastructure could be legitimate, but no contradicting evidence was found."
        else:
            verdict, conf = "INSUFFICIENT_EVIDENCE", "medium" if fusion["confidence"] >= 0.4 else "low"
            alt = "; ".join(legit) or "A benign shared household/office with an ordinary payment need cannot be ruled out."
        return {"alternative_explanation": alt, "supporting_evidence": sorted(set(supp)), "weaknesses": weak,
                "missing_evidence": fusion["missing_information"], "recommended_confidence": conf, "verdict": verdict,
                "challenged": verdict != "CONCLUSION_SURVIVES"}


class LLMProvider(AIProvider):
    """OpenAI-compatible chat endpoint. Configure RISKMESH_LLM_URL / _KEY / _MODEL. Output is JSON-validated."""
    name = "llm"
    SYS = ("You are a payment-risk analysis component. Use ONLY the evidence supplied. Cite evidence IDs exactly as given. "
           "Never invent evidence, transactions, customers or policy. Never decide that fraud occurred. "
           "INSUFFICIENT_EVIDENCE is a valid answer. Reply with a single JSON object only.")

    def __init__(self):
        self.url = os.getenv("RISKMESH_LLM_URL", "https://api.openai.com/v1/chat/completions")
        self.key, self.model = os.getenv("RISKMESH_LLM_KEY"), os.getenv("RISKMESH_LLM_MODEL", "gpt-4o-mini")

    def _chat(self, task, payload):
        body = json.dumps({"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": self.SYS},
                                        {"role": "user", "content": task + "\n" + json.dumps(payload)}]}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
        return json.loads(re.sub(r"^```(json)?|```$", "", txt.strip()))

    def extract_conversation_signals(self, conversations):
        o = self._chat('Return {"signals":[{"signal":str,"evidence":[conversation ids],"confidence":0-1,"polarity":"supports|contradicts|neutral"}]}. '
                       'Contextual hints only.', conversations)
        return o["signals"]

    def fuse(self, ctx):
        return self._chat('Return {"risk_summary":str,"supporting_evidence":[ids],"contradicting_evidence":[ids],"missing_information":[str],'
                          '"confidence":0-1,"claims":[{"claim":str,"evidence":[ids]}]}.', ctx)

    def critique(self, ctx, fusion):
        return self._chat('Try to DISPROVE the fusion conclusion. Return {"alternative_explanation":str,"supporting_evidence":[ids],"weaknesses":[str],'
                          '"missing_evidence":[str],"recommended_confidence":"low|medium|high","verdict":"CONCLUSION_SURVIVES|LEGITIMATE_EXPLANATION_LIKELY|INSUFFICIENT_EVIDENCE"}.',
                          {"context": ctx, "fusion": fusion})


MOCK = MockProvider()


def get_provider() -> AIProvider:
    if os.getenv("RISKMESH_AI_MODE", "mock").lower() == "llm" and os.getenv("RISKMESH_LLM_KEY"):
        return LLMProvider()
    return MOCK


def guarded(provider, method, validator, allowed, *args):
    """Run a provider method; reject ungrounded output and fall back to the deterministic provider."""
    try:
        out = getattr(provider, method)(*args)
        validator(out, allowed)
        return out, provider.name, None
    except Exception as exc:
        if provider.name == "mock":
            raise
        return getattr(MOCK, method)(*args), "mock", f"fallback ({type(exc).__name__}): {exc}"


# ------------------------------------------------------------------ intervention simulator (plain code)
OPTIONS = {  # synthetic prototype parameters -- NOT measured Razorpay values
    "NO_ACTION":                  dict(eff=0.00, intr=0.00, cost=0.00, cost_l="None",   review=False, rev="N/A"),
    "MONITOR":                    dict(eff=0.10, intr=0.02, cost=0.01, cost_l="Low",    review=False, rev="Yes"),
    "STEP_UP_VERIFICATION":       dict(eff=0.45, intr=0.20, cost=0.06, cost_l="Low",    review=True,  rev="Yes"),
    "MERCHANT_FOLLOW_UP":         dict(eff=0.35, intr=0.12, cost=0.07, cost_l="Medium", review=True,  rev="Yes"),
    "HOLD_BENEFICIARY_TRANSFERS": dict(eff=0.80, intr=0.35, cost=0.08, cost_l="Medium", review=True,  rev="Yes (release after review)"),
    "NEEDS_REVIEW":               dict(eff=0.25, intr=0.03, cost=0.08, cost_l="Medium", review=True,  rev="Yes"),
}
MULT = {"CONCLUSION_SURVIVES": 1.0, "INSUFFICIENT_EVIDENCE": 0.6, "LEGITIMATE_EXPLANATION_LIKELY": 0.3}
lab = lambda x, hi, mid: "High" if x >= hi else "Medium" if x >= mid else "Low"


def simulate(ring, fusion, critic) -> dict:
    p = round(fusion["confidence"] * MULT[critic["verdict"]], 3)
    opts = []
    for a, o in OPTIONS.items():
        rr, imp = o["eff"] * p, o["intr"] * (0.5 + 0.5 * (1 - p))
        allowed, why = True, None
        if a == "HOLD_BENEFICIARY_TRANSFERS" and not (p >= 0.7 and critic["verdict"] == "CONCLUSION_SURVIVES"):
            allowed, why = False, "Requires fused confidence >= 0.7 and a Critic verdict of CONCLUSION_SURVIVES."
        if critic["verdict"] == "INSUFFICIENT_EVIDENCE" and a != "NEEDS_REVIEW":
            allowed, why = False, "Critic returned INSUFFICIENT_EVIDENCE: only analyst review is permitted."
        opts.append({"action": a, "potential_risk_reduction": lab(rr, 0.45, 0.2), "risk_reduction_score": round(rr, 3),
                     "potential_legitimate_impact": "None" if o["intr"] == 0 else lab(imp, 0.25, 0.1),
                     "legit_impact_score": round(imp, 3), "operational_cost": o["cost_l"], "review_required": o["review"],
                     "reversible": o["rev"], "est_amount_protected_inr": round(o["eff"] * p * ring["exposure_inr"], 0),
                     "utility": round(rr - imp - o["cost"], 3), "allowed": allowed, "blocked_reason": why})
    ok = [o for o in opts if o["allowed"]]
    best = max(ok, key=lambda o: o["utility"])
    if best["action"] == "NO_ACTION" and ring["structural"] >= 60:
        best = next(o for o in ok if o["action"] == "MONITOR")      # notable graph structure: keep watching
    assert best["action"] in ALLOWED_ACTIONS
    return {"options": opts, "recommended_action": best["action"], "effective_risk_probability": p,
            "human_review_required": best["review_required"] or best["action"] == "HOLD_BENEFICIARY_TRANSFERS",
            "rationale": f"Highest expected utility among permitted options given fused confidence {fusion['confidence']} and Critic verdict {critic['verdict']}.",
            "disclaimer": "Synthetic prototype estimates; not measured Razorpay values. No action is executed automatically."}
