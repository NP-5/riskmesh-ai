"""RiskMesh AI API (synthetic prototype; not an internal Razorpay system)."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from . import cases as C
from .ai import get_provider
from .db import read_audit
from .ingest import ingest_events
from .rules_engine import RULES, rule_text


@asynccontextmanager
async def lifespan(app):
    if not C.S.cases: C.generate_demo()
    yield


app = FastAPI(title="RiskMesh AI", version="1.0", lifespan=lifespan,
              description="Explainable AI Risk Intelligence for Digital Payments. Synthetic prototype; not an internal Razorpay system.")


def guard(fn, *a):
    try: return fn(*a)
    except KeyError as e: raise HTTPException(404, f"not found: {e}")
    except C.Conflict as e: raise HTTPException(409, str(e))
    except ValueError as e: raise HTTPException(422, str(e))


class IngestReq(BaseModel): events: list[dict]; detect: bool = False
class GenReq(BaseModel): seed: int = 7
class DecisionReq(BaseModel): actor: str = "analyst"; decision: Optional[str] = None; reason: str = ""
class CommentReq(BaseModel): actor: str = "analyst"; text: str = Field(min_length=1)
class BacktestReq(BaseModel): rule: Optional[dict] = None; proposal_id: Optional[str] = None
class RuleDecisionReq(BaseModel): actor: str; reason: str


@app.get("/api/health")
def health(): return {"status": "ok", "ai_provider": get_provider().name, "cases": len(C.S.cases), "synthetic": True}

@app.post("/events/ingest")
def ingest(r: IngestReq):
    res = ingest_events(r.events)
    if r.detect: res["detection"] = C.run_detection()
    return res

@app.post("/demo/generate")
def demo_generate(r: GenReq = GenReq()): return C.generate_demo(r.seed)

@app.get("/demo/scenarios")
def scenarios(): return C.scenario_map()

@app.get("/cases")
def list_cases(): return [C.case_summary(c) for c in C.S.cases.values()]

@app.get("/cases/{cid}")
def get_case(cid: str): return guard(C.case_view, cid)

@app.get("/cases/{cid}/evidence")
def evidence(cid: str):
    v = guard(C.case_view, cid)
    return {"evidence": v["evidence"], "rule_results": v["rule_results"], "established": v["established"], "not_established": v["not_established"]}

@app.get("/cases/{cid}/graph")
def graph_(cid: str): return guard(C.case_graph, cid)

@app.post("/cases/{cid}/investigate")
def investigate(cid: str):
    f = guard(C.investigate, cid); c = C.S.cases[cid]
    return {"conversation_signals": c["conversation_signals"], "fusion": f, "provider": c["ai_provider"]}

@app.post("/cases/{cid}/critic")
def critic(cid: str): return guard(C.critic, cid)

@app.post("/cases/{cid}/simulate")
def simulate_(cid: str): return guard(C.run_simulation, cid)

@app.post("/cases/{cid}/approve")
def approve(cid: str, r: DecisionReq = DecisionReq()): return guard(C.decide, cid, "approve", r.actor, r.decision, r.reason)

@app.post("/cases/{cid}/reject")
def reject(cid: str, r: DecisionReq = DecisionReq()): return guard(C.decide, cid, "reject", r.actor, r.decision, r.reason)

@app.post("/cases/{cid}/comment")
def comment(cid: str, r: CommentReq): return guard(C.comment, cid, r.actor, r.text)

@app.get("/cases/{cid}/audit")
def audit_(cid: str): guard(C._case, cid); return read_audit(cid)

@app.get("/cases/{cid}/export")
def export(cid: str): return guard(C.export_case, cid)

class ChatReq(BaseModel): message: str = Field(min_length=1, max_length=500); case_id: Optional[str] = None

@app.post("/chat")
def chat(r: ChatReq):
    """Grounded assistant: answers only from live case/metric data; never executes actions."""
    q = r.message.lower(); m = C.metrics(); cs = [C.case_summary(c) for c in C.S.cases.values()]
    cid = next((c["case_id"] for c in cs if c["case_id"].lower() in q.replace(" ", "_")), None) or r.case_id
    pend = [c["case_id"] for c in cs if c["status"] == "PENDING_REVIEW"]
    if cid and cid in C.S.cases and not any(k in q for k in ("pending", "all cases", "overview", "agree")):
        v = C.case_view(cid); g = v["ring"]; rec = v.get("recommendation") or {}
        return {"reply": f"{cid}: {g['n_members']} customers, structural risk {g['structural']}/100, 6-hour escalation signal {g['escalation']}/100. "
                f"7-day value to {g['top_beneficiary']} is Rs {round(g['exposure_inr']):,} ({g['concentration']*100:.0f}% concentration). Status: {v['status']}. "
                + (f"Recommended action: {rec.get('action')} (human review {'required' if rec.get('human_review_required') else 'optional'}). " if rec else "No recommendation yet; run the investigation, critic and simulation. ")
                + (f"Not established by the evidence: {v['not_established'][0]}" if v.get("not_established") else "")}
    if any(k in q for k in ("pending", "review")):
        return {"reply": f"{m['pending_reviews']} case(s) await human review: {', '.join(pend) or 'none'}. Open a case to run the Critic Agent and approve or override."}
    if any(k in q for k in ("agree", "recommend")):
        a = m.get("ai_recommendation_agreement")
        return {"reply": "AI recommendation agreement has not been measured yet; run the AI evaluation." if a is None else f"AI recommendation agreement is {a*100:.0f}% across the demo cases."}
    if any(k in q for k in ("highest", "riskiest", "high-risk", "high risk", "ring")):
        t = max(cs, key=lambda c: c["structural_risk"])
        return {"reply": f"{m['high_risk_networks']} high-risk network(s). Highest structural risk: {t['case_id']} at {t['structural_risk']}/100 with escalation {t['escalation_signal']}/100, top beneficiary {t['top_beneficiary']}."}
    if any(k in q for k in ("beneficiar", "exposure", "concentration")):
        return {"reply": "Top beneficiary exposure: " + "; ".join(f"{x['beneficiary']} Rs {round(x['exposure_inr']):,}" for x in m["beneficiary_concentration"])}
    if any(k in q for k in ("rule", "false", "fpr")):
        return {"reply": f"Active rule {m['active_rule_version']}; backtest false-positive rate {m['false_positive_rate_active_rule_backtest']}. Use Rule re-engineering to propose and backtest a change; a human must approve it."}
    if any(k in q for k in ("active", "how many", "overview", "summary", "cases")):
        return {"reply": f"{m['active_cases']} active case(s), {m['high_risk_networks']} high-risk network(s), {m['pending_reviews']} pending review(s)."}
    return {"reply": "I can summarise active cases, pending reviews, AI agreement, beneficiary exposure, rules, or any case (for example 'CASE_101'). I only report data; humans make every decision."}

@app.get("/metrics")
def metrics(): return C.metrics()

@app.post("/evaluation/run")
def evaluation(): return C.evaluate()

@app.post("/rules/propose")
def propose(): return guard(RULES.propose)

@app.post("/rules/backtest")
def backtest(r: BacktestReq):
    if r.proposal_id: rule = guard(lambda: RULES.proposals[r.proposal_id]["rule"])
    elif r.rule: rule = r.rule
    else: raise HTTPException(422, "provide rule or proposal_id")
    try: return RULES.backtest(rule)
    except (KeyError, TypeError) as e: raise HTTPException(422, f"invalid rule: {e}")

@app.get("/rules")
def rules(): return {"active": RULES.active, "versions": {k: {**v, "text": rule_text(v)} for k, v in RULES.versions.items()}, "proposals": RULES.proposals}

@app.post("/rules/{pid}/approve")
def rule_approve(pid: str, r: RuleDecisionReq): return guard(RULES.decide, pid, True, r.actor, r.reason)

@app.post("/rules/{pid}/reject")
def rule_reject(pid: str, r: RuleDecisionReq): return guard(RULES.decide, pid, False, r.actor, r.reason)

@app.get("/")
def home(): return FileResponse(Path(__file__).parent / "static" / "index.html")
