# 🛡️ RiskMesh AI

### Explainable AI Risk Intelligence for Digital Payments

🔗 **Live demo:** https://huggingface.co/spaces/nareshpavan/riskmesh-ai

Fraud systems that score one payment at a time miss the pattern that matters: **several customers sharing a device, an IP range and a payment instrument, all sending money to the same new beneficiary within minutes.**

But shared infrastructure is also what an *office*, a *household* or a *vendor payout* looks like. Blindly flagging graph connections freezes legitimate users.

**RiskMesh AI** finds coordinated risk, then **argues against itself** before recommending anything, and keeps a human in charge of every consequential decision.

---

## What makes it different

> Not "an LLM chatbot with a fraud score."

| Capability | What it does |
|---|---|
| **Revision-aware ingest** | Idempotent imports, source revisions, superseded/stale handling. Re-importing the same revision never duplicates an event. |
| **Graph risk detection** | NetworkX entity graph (customer, account, device, IP /24, instrument, beneficiary, transaction) + **DBSCAN** to surface *candidate* risk rings. |
| **Structural + 6-hour escalation signals** | Two transparent, labelled prototype signals (not guaranteed predictions). |
| **Evidence-first rules** | Every rule returns `decision · reason · evidence IDs · confidence` and cannot cite evidence that doesn't exist. |
| **"What the evidence does NOT establish"** | Every case states what the data cannot prove (e.g. *a shared IP does not mean the same person*). |
| **Conversation Agent** | Extracts contextual hints from support messages. It never decides fraud occurred. |
| **Fusion Agent** | Combines graph, transaction, timing, rule and conversation evidence. Every claim cites evidence IDs. |
| **Critic Agent** | Tries to *disprove* the conclusion. Can answer `INSUFFICIENT_EVIDENCE`. |
| **Intervention Simulator** | Compares 6 bounded actions on risk reduction, legitimate-user impact, cost, reversibility. |
| **Targeted holds, not freezes** | `HOLD_BENEFICIARY_TRANSFERS` affects one beneficiary, never every account in the graph. |
| **Human governance** | Human decisions are stored *separately*; overrides need a reason; the AI recommendation is never overwritten. |
| **Re-engineer Agent** | Proposes rule changes, backtests them, and requires human approval before activation. |
| **Append-only audit log** | Ingest, rules, detection, AI steps, recommendation, human decision, override, rule change, backtest, export. |
| **Mock AI mode** | Deterministic provider, so the whole demo runs with **no API key**. Pluggable LLM provider behind the same interface. |

---

## Architecture

```text
 PAYMENT EVENTS ──► REVISION-AWARE INGEST
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   TRANSACTION FEATURES            CONTEXT / CONVERSATIONS
          │                               │
   ENTITY GRAPH + DBSCAN            RULE ENGINE + EVIDENCE
          └───────────────┬───────────────┘
                          ▼
                    CASE CREATION
                          ▼
   Conversation Agent → Fusion Agent → Critic Agent
                          ▼
                 Intervention Simulator
                          ▼
              POLICY GUARDRAILS (allow-listed actions only)
                          ▼
                     HUMAN REVIEW
                    ┌─────┴─────┐
                    ▼           ▼
          BOUNDED ACTION    AUDIT LOG ──► Re-engineer Agent
                                          (propose → backtest → human approval)
```

---

## Three demo scenarios

| Scenario | Fusion confidence | Critic verdict | Final recommendation |
|---|---|---|---|
| **Coordinated payment ring** | 0.87 | `CONCLUSION_SURVIVES` | `HOLD_BENEFICIARY_TRANSFERS` (one beneficiary only, human approval required) |
| **Legitimate corporate network** | 0.04 | `LEGITIMATE_EXPLANATION_LIKELY` | `MONITOR` |
| **Ambiguous case** | 0.40 | `INSUFFICIENT_EVIDENCE` | `NEEDS_REVIEW` |

The second scenario is the point of the project: the graph looks scary (shared device, shared IP, concentrated beneficiary), but a verified vendor, steady history and a benign conversation lead the system to **not** intervene.

---

## AI guardrails

- The model may **only** reason over supplied evidence; every cited ID is validated.
- Ungrounded output is **rejected** and the deterministic provider takes over.
- The LLM **never chooses interventions**. The simulator is plain code limited to an allow-list of six actions.
- No payments are executed and no accounts are frozen automatically.
- Rule changes are never applied silently; they require an explicit human approval.

---

## Quick start

```bash
git clone https://github.com/<your-username>/riskmesh-ai.git
cd riskmesh-ai
pip install -r requirements.txt
uvicorn app.main:app --port 7860     # open http://localhost:7860  (API docs at /docs)
pytest -q                            # 14 tests
```

Optional LLM mode (any OpenAI-compatible endpoint):

```bash
RISKMESH_AI_MODE=llm RISKMESH_LLM_KEY=... uvicorn app.main:app
```

Docker: `docker build -t riskmesh . && docker run -p 7860:7860 riskmesh`

---

## API highlights

`POST /demo/generate` · `GET /cases` · `GET /cases/{id}/graph` · `GET /cases/{id}/evidence` · `POST /cases/{id}/investigate` · `POST /cases/{id}/critic` · `POST /cases/{id}/simulate` · `POST /cases/{id}/approve` · `POST /cases/{id}/reject` · `GET /cases/{id}/audit` · `GET /cases/{id}/export` · `POST /rules/propose` · `POST /rules/backtest` · `POST /evaluation/run` · `GET /metrics` · `GET /api/health`

Each case export produces `case.json`, `evidence.json`, `decision.json`, `audit_log.jsonl` and a hashed `manifest.json`.

---

## Results (synthetic data)

- Synthetic world: **10,500+ transactions**, 500+ customers, devices and IP ranges, 100+ beneficiaries, with legitimate traffic plus three injected scenarios.
- DBSCAN + graph closure recovers **all three injected groups with no extra candidates**.
- Rule re-engineering backtest (400 synthetic historical groups): adding beneficiary-concentration and 6-hour-escalation conditions raised precision **0.425 → 0.923** and cut affected legitimate customers **1005 → 44**, at a recall cost of **0.90 → 0.70**. The trade-off is shown, not hidden.
- AI evaluation metrics (evidence grounding, unsupported-claim rate, critic challenge rate, recommendation agreement, human override rate, investigation time) are **computed from real runs** via `POST /evaluation/run`.

---

## Honest limitations

- All data is synthetic and the three scenarios are hand-built, so scenario "agreement" (3 cases) is a sanity check, **not** an accuracy claim.
- Intervention-simulator parameters and the backtest history are synthetic assumptions, not measured values.
- The mock provider's perfect evidence grounding is true by construction; the LLM path is implemented but not benchmarked.
- This is a prototype: no authentication, in-memory case state, SQLite storage.

---

## Tech stack

Python 3.11 · FastAPI · Pydantic · Pandas · NumPy · scikit-learn (DBSCAN) · NetworkX · SQLAlchemy + SQLite · Pytest · HTML/CSS/JS front end · Docker

## Repository layout

```text
app/
  ingest.py        revision-aware, idempotent ingest
  graph.py         entity graph, DBSCAN, structural + escalation signals
  evidence.py      evidence, rule routing, "does NOT establish"
  ai.py            provider abstraction, agents, guardrails, simulator
  rules_engine.py  Re-engineer Agent, backtesting, approval
  cases.py         case lifecycle, human override, export, evaluation
  main.py          FastAPI app
  static/          demo UI served by the API
tests/             14 tests (ingest, scenarios, guardrails, governance, export)
```

---

> **Disclaimer:** RiskMesh AI is an independent prototype that runs entirely on synthetic data. It uses no real customer, payment or credential data and is not affiliated with any payment company.
