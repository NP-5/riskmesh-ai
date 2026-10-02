import json, os
from app import ai, cases as C, db
from app.ingest import ingest_events, snapshot_stats
from app.rules_engine import RULES, RuleRegistry


def ev(rev, amt=1.0, sid="PAY_X1"):
    return {"source_type": "payment_event", "source_id": sid, "revision": rev, "timestamp": "2026-09-30T14:20:00Z", "payload": {"amount": amt}}


def test_revision_aware_ingest_is_idempotent(fresh):
    ingest_events([ev(1)]); r = ingest_events([ev(2, 5)]); assert r["superseded"] == 1
    before = snapshot_stats()
    again = ingest_events([ev(2, 5), ev(1)])
    assert again["duplicates"] == 2 and again["inserted"] == 0 and snapshot_stats() == before
    ingest_events([ev(3, sid="PAY_X2")])
    assert ingest_events([ev(2, sid="PAY_X2")])["stale"] == 1      # late older revision never becomes active


def test_synthetic_volume():
    n = C.S.counts
    assert n["transactions"] >= 10000 and n["devices"] >= 500 and n["ip_ranges"] >= 500 and n["beneficiaries"] >= 100 and n["customers"] >= 500


def test_dbscan_finds_three_scenarios(fresh):
    _, sc = fresh
    assert set(sc) == {"ring", "legit", "ambiguous"}


def run_all(client, cid):
    client.post(f"/cases/{cid}/investigate"); client.post(f"/cases/{cid}/critic")
    return client.post(f"/cases/{cid}/simulate").json()


def test_scenario_outcomes(fresh):
    client, sc = fresh
    assert run_all(client, sc["ring"])["recommended_action"] == "HOLD_BENEFICIARY_TRANSFERS"
    assert run_all(client, sc["legit"])["recommended_action"] in ("NO_ACTION", "MONITOR")
    amb = run_all(client, sc["ambiguous"])
    assert amb["recommended_action"] == "NEEDS_REVIEW"
    assert client.get(f"/cases/{sc['ambiguous']}").json()["critic"]["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_not_established_section_and_no_freeze(fresh):
    client, sc = fresh; v = client.get(f"/cases/{sc['ring']}").json()
    assert any("does not establish" in t for t in v["not_established"])
    assert "no account freeze" in (run_all(client, sc["ring"]) and client.get(f"/cases/{sc['ring']}").json()["recommendation"]["scope"])


def test_rule_results_cite_existing_evidence(fresh):
    client, sc = fresh
    for cid in sc.values():
        v = client.get(f"/cases/{cid}").json(); ids = {e["id"] for e in v["evidence"]}
        assert all(r["evidence"] and set(r["evidence"]) <= ids for r in v["rule_results"])


def test_human_override_kept_separate_and_audited(fresh):
    client, sc = fresh; cid = sc["ring"]; run_all(client, cid)
    assert client.post(f"/cases/{cid}/reject", json={"actor": "a1", "decision": "MONITOR"}).status_code == 422   # reason required
    r = client.post(f"/cases/{cid}/reject", json={"actor": "a1", "decision": "MONITOR", "reason": "Verified enterprise vendor"}).json()
    assert r["automated_decision"] == "HOLD_BENEFICIARY_TRANSFERS" and r["human_decision"] == "MONITOR" and r["override"]
    v = client.get(f"/cases/{cid}").json()
    assert v["recommendation"]["action"] == "HOLD_BENEFICIARY_TRANSFERS"          # original not overwritten
    acts = [a["action"] for a in client.get(f"/cases/{cid}/audit").json()]
    for a in ("CASE_CREATED", "RULE_EXECUTED", "CONVERSATION_ANALYZED", "CRITIC_RESULT", "RECOMMENDATION", "HUMAN_OVERRIDE"):
        assert a in acts
    assert client.post(f"/cases/{cid}/approve", json={}).status_code == 409        # already resolved


def test_decision_requires_simulation(fresh):
    client, sc = fresh
    assert client.post(f"/cases/{sc['ring']}/approve", json={"actor": "x"}).status_code == 409


def test_guardrail_rejects_ungrounded_llm_output():
    class Bad(ai.AIProvider):
        name = "llm"
        def fuse(self, ctx): return {"supporting_evidence": ["E999"], "contradicting_evidence": [], "confidence": .9, "claims": []}
    ctx = {"evidence": [{"id": "E1"}]}
    out, used, fb = ai.guarded(Bad(), "fuse", ai.validate_fusion, {"E1"}, {"case_id": "x", "evidence": [
        {"id": "E1", "kind": "shared_device", "family": "graph", "statement": "s", "polarity": "supports", "strength": .9, "source_refs": []}],
        "rule_results": [], "scores": {}})
    assert used == "mock" and "E999" in fb


def test_simulator_only_emits_allowed_actions(fresh):
    client, sc = fresh
    for cid in sc.values():
        sim = run_all(client, cid)
        assert {o["action"] for o in sim["options"]} == set(ai.ALLOWED_ACTIONS) and sim["recommended_action"] in ai.ALLOWED_ACTIONS


def test_rule_never_activates_without_human(fresh):
    client, _ = fresh; before = client.get("/rules").json()["active"]
    p = client.post("/rules/propose").json()
    assert p["status"] == "PENDING_HUMAN_APPROVAL" and client.get("/rules").json()["active"] == before
    bt = client.post("/rules/backtest", json={"proposal_id": p["proposal_id"]}).json()
    assert bt["after"]["precision"] > bt["before"]["precision"] and client.get("/rules").json()["active"] == before
    assert client.post(f"/rules/{p['proposal_id']}/approve", json={"actor": "", "reason": ""}).status_code == 422
    client.post(f"/rules/{p['proposal_id']}/approve", json={"actor": "lead", "reason": "Backtest reviewed"})
    assert client.get("/rules").json()["active"] == p["proposed_version"]


def test_export_manifest(fresh):
    client, sc = fresh; cid = sc["ring"]; run_all(client, cid)
    m = client.get(f"/cases/{cid}/export").json()
    d = m["directory"]
    for f in ("case.json", "evidence.json", "decision.json", "audit_log.jsonl", "manifest.json"):
        assert os.path.exists(os.path.join(d, f))
    assert m["manifest"]["case_id"] == cid and m["manifest"]["evidence_ids"]


def test_evaluation_is_computed_and_grounded(fresh):
    client, _ = fresh; e = client.post("/evaluation/run").json()
    assert e["evidence_grounding"] == 1.0 and e["unsupported_claim_rate"] == 0.0
    assert e["recommendation_agreement_with_scenario_label"] == 1.0 and e["baseline"]["ring_detection_precision"] == 0.839


def test_health_metrics_and_audit_append_only(fresh):
    client, _ = fresh
    assert client.get("/api/health").json()["status"] == "ok" and client.get("/metrics").status_code == 200
    n = len(open(db.AUDIT_FILE).readlines()); client.post("/rules/propose")
    assert len(open(db.AUDIT_FILE).readlines()) > n
