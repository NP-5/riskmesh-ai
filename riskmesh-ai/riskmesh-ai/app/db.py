"""SQLite persistence: revision-aware event store + append-only audit log."""
import datetime as dt, json, os, time
from pathlib import Path
from sqlalchemy import Column, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATA_DIR = Path(os.getenv("RISKMESH_DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_FILE = DATA_DIR / "audit_log.jsonl"
engine = create_engine(f"sqlite:///{DATA_DIR / 'riskmesh.db'}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    source_type = Column(String, index=True)
    source_id = Column(String, index=True)
    revision = Column(Integer)
    ts = Column(String)
    payload = Column(Text)
    status = Column(String, default="active")  # active | superseded (stale)
    __table_args__ = (UniqueConstraint("source_type", "source_id", "revision"),)


class Audit(Base):
    __tablename__ = "audit"
    id = Column(Integer, primary_key=True)
    ts = Column(String)
    case_id = Column(String, index=True)
    actor = Column(String)
    action = Column(String)
    reason = Column(Text)
    details = Column(Text)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def reset_db():
    """Demo reset. The previous audit file is rotated, never edited in place."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    if AUDIT_FILE.exists() and AUDIT_FILE.stat().st_size:
        AUDIT_FILE.rename(DATA_DIR / f"audit_log.{int(time.time() * 1000)}.bak.jsonl")


def audit(case_id, actor, action, reason="", details=None) -> dict:
    rec = {"timestamp": now_iso(), "case_id": case_id, "actor": actor, "action": action,
           "reason": reason, "details": details or {}}
    with Session() as s:
        s.add(Audit(ts=rec["timestamp"], case_id=case_id, actor=actor, action=action,
                    reason=reason, details=json.dumps(rec["details"], default=str)))
        s.commit()
    with open(AUDIT_FILE, "a") as f:  # append-only
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def read_audit(case_id=None) -> list[dict]:
    with Session() as s:
        q = s.query(Audit).order_by(Audit.id)
        if case_id:
            q = q.filter(Audit.case_id == case_id)
        return [{"timestamp": r.ts, "case_id": r.case_id, "actor": r.actor, "action": r.action,
                 "reason": r.reason, "details": json.loads(r.details or "{}")} for r in q]


Base.metadata.create_all(engine)
