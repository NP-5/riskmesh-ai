# RiskMesh AI — Explainable AI Risk Intelligence for Digital Payments

**Live demo:** hosted as a *static* Hugging Face Space (`static_site/index.html`, single self-contained page).
**Full source:** this repository — Python/FastAPI backend (`app/`), tests (`tests/`), and the static demo page (`static_site/`).


**RiskMesh AI** is an explainable AI risk-intelligence platform for digital payment ecosystems. It combines revision-aware event processing, graph-based coordinated-risk detection, behavioral signals, contextual conversation analysis, and multi-agent investigation. The system uses a Fusion Agent to connect independent evidence, a Critic Agent to challenge conclusions, an Intervention Simulator to compare bounded responses, and a Re-engineer Agent to backtest proposed risk-rule changes. Humans remain responsible for consequential decisions, while every automated recommendation and override is recorded in an auditable trail.

> This project is an independent prototype using synthetic data. It is not an internal Razorpay system and does not use real Razorpay customer or transaction data.

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860      # open http://localhost:7860  (API docs: /docs)
pytest -q
docker build -t riskmesh . && docker run -p 7860:7860 riskmesh   # also works as a Hugging Face Docker Space
```
Mock AI mode (deterministic) is the default, so no API key is needed. To use an LLM:
`RISKMESH_AI_MODE=llm RISKMESH_LLM_KEY=... [RISKMESH_LLM_URL=... RISKMESH_LLM_MODEL=...]` (any OpenAI-compatible endpoint).
LLM output is validated against supplied evidence IDs; ungrounded output is rejected and the deterministic provider is used.

## Layout
`app/ingest.py` revision-aware ingest · `graph.py` entity graph, DBSCAN, structural + 6h escalation signals · `evidence.py` evidence, rules, "does NOT establish" ·
`ai.py` providers, Conversation/Fusion/Critic agents, guardrails, intervention simulator · `rules_engine.py` Re-engineer Agent + backtest + approval ·
`cases.py` case lifecycle, human override, export manifest, evaluation · `main.py` FastAPI.

## Honest limits
- Everything is synthetic. The three demo scenarios are hand-built, so "recommendation agreement" (3 cases) and 100% evidence grounding of the *mock* provider are sanity checks, not accuracy claims.
- Simulator parameters and the rule-backtest history are synthetic assumptions, not measured values.
- The baseline block (0.839 / 0.833 / ~4.1h / 0.233 / 15) is carried over from your existing Abuse-Ring Sentinel evaluation; it is not re-measured here.
- No payment or account action is ever executed; `HOLD_BENEFICIARY_TRANSFERS` is recorded as a simulated, human-approved action.
