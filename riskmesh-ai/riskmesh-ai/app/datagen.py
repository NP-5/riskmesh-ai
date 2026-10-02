"""Synthetic, Razorpay-*like* payment data. Entirely fabricated; no real customer/payment data."""
import random
from datetime import datetime, timedelta, timezone
import numpy as np

NOW = datetime(2026, 9, 30, 18, 0, 0, tzinfo=timezone.utc)   # reference "now" of the synthetic world
RESERVED_IPS = {"103.20.45.0/24", "14.140.22.0/24", "49.36.18.0/24"}


def _iso(t): return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate(seed: int = 7) -> dict:
    rng, pr = np.random.default_rng(seed), random.Random(seed)
    N = 520
    customers = [f"C{i:04d}" for i in range(1, N + 1)]
    merchants = [f"M{i:03d}" for i in range(1, 111)]
    bene_ids = [f"B{i:03d}" for i in range(1, 111)]
    benes = {b: {"name": f"Settlement Account {b}", "verified": bool(rng.random() < 0.7),
                 "age_days": int(rng.integers(60, 1500))} for b in bene_ids}
    benes["B017"] = {"name": "Settlement Account B017", "verified": False, "age_days": 9}
    benes["B042"] = {"name": "Enterprise Vendor B042", "verified": True, "age_days": 1200}
    benes["B077"] = {"name": "Settlement Account B077", "verified": False, "age_days": 40}
    pool = [b for b in bene_ids if b not in ("B017", "B042", "B077")]

    def ip_of(i):
        ip = f"{100 + i // 200}.{i % 200 + 20}.{(i * 7) % 250}.0/24"
        return ip if ip not in RESERVED_IPS else f"172.16.{i % 250}.0/24"

    prof, labels, last_shared = {}, {}, -5
    for i, c in enumerate(customers):
        dev, ip, inst = f"D{i + 1:04d}", ip_of(i), f"P{i + 1:05d}"
        if i > 0 and i - last_shared > 1 and rng.random() < 0.06:   # occasional benign pair-sharing
            dev = prof[customers[i - 1]]["device"] if rng.random() < 0.5 else dev
            ip = prof[customers[i - 1]]["ip"] if dev == f"D{i + 1:04d}" else ip
            last_shared = i
        prof[c] = {"device": dev, "ip": ip, "inst": inst}; labels[c] = "normal"

    pays = []

    def pay(c, t, ben, amt, merchant=None):
        pays.append({"customer_id": c, "merchant_id": merchant or pr.choice(merchants), "device_id": prof[c]["device"],
                     "ip24": prof[c]["ip"], "instrument_id": prof[c]["inst"], "beneficiary_id": ben,
                     "amount": round(float(amt), 2), "created_at": t})

    # ---- legitimate background activity
    for c in customers:
        fav = pr.sample(pool, 3)
        for _ in range(int(rng.poisson(17)) + 3):
            ben = pr.choice(fav) if rng.random() < 0.8 else pr.choice(pool)
            pay(c, NOW - timedelta(seconds=float(rng.uniform(0, 30 * 86400))), ben,
                min(max(rng.lognormal(7.5, 0.9), 50), 45000))

    def history(c, n=5):
        for _ in range(n):
            pay(c, NOW - timedelta(days=float(rng.uniform(8, 30))), pr.choice(pool), min(max(rng.lognormal(7.5, .9), 50), 45000))

    # ---- Scenario 1: coordinated ring (shared device/IP/instrument, burst to a new beneficiary)
    ring = [f"C{9001 + i}" for i in range(5)]
    for i, c in enumerate(ring):
        prof[c] = {"device": "D9001" if i < 4 else "D9002", "ip": "103.20.45.0/24",
                   "inst": "P9001" if i < 3 else f"P900{i + 2}"}
        labels[c] = "ring"; history(c)
    base = NOW - timedelta(hours=3)
    for i, c in enumerate(ring):
        for k in range(5 if i % 2 == 0 else 4):
            off = 0 if (i, k) == (0, 0) else 42 if (i, k) == (4, 4) else float(rng.uniform(0, 42))
            pay(c, base + timedelta(minutes=off), "B017", rng.uniform(48500, 49900), "M090")

    # ---- Scenario 2: legitimate merchant network (corporate office, verified vendor)
    emp = [f"C{9101 + i}" for i in range(7)]
    for i, c in enumerate(emp):
        prof[c] = {"device": "D9101" if i < 5 else "D9102", "ip": "14.140.22.0/24", "inst": f"P910{i + 1}"}
        labels[c] = "legit"
        for k in range(7):
            pay(c, NOW - timedelta(days=k * 4.2 + i * 0.3, hours=float(rng.uniform(0, 3))), "B042",
                12000 * (1 + rng.normal(0, 0.05)), "M055")

    # ---- Scenario 3: ambiguous (shared infrastructure, new unverified beneficiary, mixed signals)
    amb = [f"C{9201 + i}" for i in range(4)]
    for i, c in enumerate(amb):
        prof[c] = {"device": "D9201" if i < 3 else "D9202", "ip": "49.36.18.0/24", "inst": f"P920{i + 1}"}
        labels[c] = "ambiguous"; history(c)
    times = ([NOW - timedelta(hours=20) + timedelta(minutes=float(rng.uniform(0, 100))) for _ in range(8)]
             + [NOW - timedelta(hours=float(rng.uniform(7, 36))) for _ in range(9)]
             + [NOW - timedelta(hours=float(rng.uniform(0.5, 5))) for _ in range(3)])
    for j, t in enumerate(times):
        pay(amb[j % 4], t, "B077", rng.uniform(8000, 30000), "M071")

    # ---- events (revision-aware): some payments get a revision 2 (fee correction)
    pays.sort(key=lambda p: p["created_at"])
    events = []
    for n, p in enumerate(pays, 1):
        p["created_at"] = _iso(p["created_at"]) if not isinstance(p["created_at"], str) else p["created_at"]
        payload = {**p, "payment_id": f"PAY_{n:05d}", "currency": "INR", "status": "captured"}
        events.append({"source_type": "payment_event", "source_id": payload["payment_id"], "revision": 1,
                       "timestamp": payload["created_at"], "payload": payload})
        if labels[p["customer_id"]] == "normal" and rng.random() < 0.004:
            p2 = {**payload, "amount": round(payload["amount"] * 1.02, 2)}
            events.append({"source_type": "payment_event", "source_id": payload["payment_id"], "revision": 2,
                           "timestamp": payload["created_at"], "payload": p2})
    pay_events = list(events)
    for n in range(250):                                    # refunds & disputes (context entities)
        p = pay_events[int(rng.integers(0, len(pay_events)))]["payload"]
        events.append({"source_type": "refund_event", "source_id": f"REF_{n + 1:04d}", "revision": 1,
                       "timestamp": p["created_at"], "payload": {"payment_id": p["payment_id"], "amount": round(p["amount"] * .5, 2)}})
    for n in range(40):
        p = pay_events[int(rng.integers(0, len(pay_events)))]["payload"]
        events.append({"source_type": "dispute_event", "source_id": f"DSP_{n + 1:03d}", "revision": 1,
                       "timestamp": p["created_at"], "payload": {"payment_id": p["payment_id"], "reason": "item_not_received"}})

    benign = ["Settlement for yesterday has not arrived yet, please check.", "Customer asked for a refund on an undelivered order.",
              "Please update our registered support email address.", "Checkout page shows a timeout on UPI collect requests.",
              "Need a copy of last month's settlement report."]
    convs = [{"id": f"CONV_{100 + n}", "customer_id": pr.choice(customers), "ts": _iso(NOW - timedelta(days=pr.random() * 20)),
              "channel": "support_chat", "text": pr.choice(benign)} for n in range(40)]
    convs += [
        {"id": "CONV_17", "customer_id": "C9002", "ts": _iso(NOW - timedelta(hours=4)), "channel": "merchant_support",
         "text": "We need to move the remaining amount through the alternate merchant account before tonight."},
        {"id": "CONV_18", "customer_id": "C9004", "ts": _iso(NOW - timedelta(hours=3.5)), "channel": "merchant_support",
         "text": "Keep each transfer under the 50k limit so it does not get flagged."},
        {"id": "CONV_41", "customer_id": "C9103", "ts": _iso(NOW - timedelta(days=2)), "channel": "merchant_support",
         "text": "Our finance team batches monthly vendor invoices and pays them through the shared corporate device. Standard process."},
        {"id": "CONV_52", "customer_id": "C9202", "ts": _iso(NOW - timedelta(hours=22)), "channel": "support_chat",
         "text": "Can we send the remaining balance from the other account? Not sure if that is allowed."},
        {"id": "CONV_53", "customer_id": "C9203", "ts": _iso(NOW - timedelta(hours=21)), "channel": "support_chat",
         "text": "We are housemates sharing one home router and a tablet for payments."},
    ]
    return {"events": events, "beneficiaries": benes, "conversations": convs, "labels": labels,
            "counts": {"customers": N + 16, "merchants": len(merchants), "devices": len({p['device_id'] for p in pays}),
                       "ip_ranges": len({p['ip24'] for p in pays}), "beneficiaries": len(benes),
                       "transactions": len(pays)}}
