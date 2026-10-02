"""Revision-aware, idempotent event ingest (adapted from REVLENS)."""
import json
from collections import defaultdict
import pandas as pd
from .db import Event, Session, audit

REQUIRED = {"source_type", "source_id", "revision", "timestamp", "payload"}


def ingest_events(events: list[dict], actor="ingest_service", log=True) -> dict:
    res = dict(received=len(events), inserted=0, duplicates=0, superseded=0, stale=0, rejected=0)
    clean = []
    for e in events:
        if not REQUIRED <= set(e) or not isinstance(e["revision"], int) or e["revision"] < 1:
            res["rejected"] += 1
        else:
            clean.append(e)
    with Session() as s:
        existing, ids_by_type = defaultdict(dict), defaultdict(set)
        for e in clean:
            ids_by_type[e["source_type"]].add(e["source_id"])
        for st, ids in ids_by_type.items():
            ids = list(ids)
            for i in range(0, len(ids), 500):
                for row in s.query(Event).filter(Event.source_type == st, Event.source_id.in_(ids[i:i + 500])):
                    existing[(st, row.source_id)][row.revision] = row
        for e in sorted(clean, key=lambda x: (x["source_type"], x["source_id"], x["revision"])):
            revs = existing[(e["source_type"], e["source_id"])]
            if e["revision"] in revs:          # idempotent re-import
                res["duplicates"] += 1
                continue
            row = Event(source_type=e["source_type"], source_id=e["source_id"], revision=e["revision"],
                        ts=e["timestamp"], payload=json.dumps(e["payload"]), status="active")
            if revs:
                latest = max(revs)
                if e["revision"] > latest:
                    revs[latest].status = "superseded"; res["superseded"] += 1
                else:                           # late-arriving older revision is stale on arrival
                    row.status = "superseded"; res["stale"] += 1
            revs[e["revision"]] = row
            s.add(row); res["inserted"] += 1
        s.commit()
    if log:
        audit("SYSTEM", actor, "EVENTS_INGESTED", "Revision-aware ingest", res)
    return res


def load_active_payments() -> pd.DataFrame:
    rows = []
    with Session() as s:
        for r in s.query(Event).filter(Event.source_type == "payment_event", Event.status == "active"):
            p = json.loads(r.payload); p["revision"] = r.revision; p["event_ts"] = r.ts
            rows.append(p)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def snapshot_stats() -> dict:
    with Session() as s:
        total = s.query(Event).count()
        active = s.query(Event).filter(Event.status == "active").count()
        superseded = s.query(Event).filter(Event.status == "superseded").count()
    return {"events_total": total, "events_active": active, "events_superseded": superseded}
