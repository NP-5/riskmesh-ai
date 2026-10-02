"""Entity graph (NetworkX), DBSCAN candidate-ring detection, structural risk and 6-hour escalation signal."""
import math
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(ts: pd.Series):
    """Seconds since epoch, independent of pandas datetime resolution."""
    return ((ts - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


FEATS = ["shared_dev", "shared_ip", "shared_inst", "ben_share", "burst60", "amt_log", "amt_cv", "recent6"]


def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for c in df.customer_id.unique():
        G.add_node(f"customer:{c}", type="customer", label=c)
        G.add_node(f"account:ACC_{c}", type="account", label=f"ACC_{c}")
        G.add_edge(f"customer:{c}", f"account:ACC_{c}", rel="owns")
    for col, typ, rel in [("device_id", "device", "uses_device"), ("ip24", "ip", "seen_on_ip"),
                          ("instrument_id", "instrument", "pays_with"), ("merchant_id", "merchant", "pays_merchant")]:
        for c, v in df[[ "customer_id", col]].drop_duplicates().itertuples(index=False):
            G.add_node(f"{typ}:{v}", type=typ, label=v)
            G.add_edge(f"customer:{c}", f"{typ}:{v}", rel=rel)
    agg = df.groupby(["customer_id", "beneficiary_id"]).amount.agg(["count", "sum"]).reset_index()
    for r in agg.itertuples(index=False):
        G.add_node(f"beneficiary:{r.beneficiary_id}", type="beneficiary", label=r.beneficiary_id)
        G.add_edge(f"customer:{r.customer_id}", f"beneficiary:{r.beneficiary_id}", rel="transfers_to",
                   txns=int(r.count), amount=float(r.sum))
    return G


def customer_features(df: pd.DataFrame, now) -> pd.DataFrame:
    def shared(col):
        n = df.groupby(col).customer_id.nunique()
        return (df[col].map(n) - 1).groupby(df.customer_id).max()
    f = pd.DataFrame({"shared_dev": shared("device_id"), "shared_ip": shared("ip24"), "shared_inst": shared("instrument_id")})
    amt = df.groupby(["customer_id", "beneficiary_id"]).amount.sum()
    f["ben_share"] = amt.groupby("customer_id").max() / amt.groupby("customer_id").sum()
    g = df.groupby("customer_id").amount
    f["amt_log"] = np.log1p(g.mean()); f["amt_cv"] = (g.std().fillna(0) / g.mean())
    f["recent6"] = df[df.ts >= now - pd.Timedelta(hours=6)].groupby("customer_id").size().reindex(f.index).fillna(0) / df.groupby("customer_id").size()
    burst = {}
    for c, grp in df.groupby("customer_id"):
        t = np.sort(secs(grp.ts))
        burst[c] = int((np.searchsorted(t, t + 3600, side="right") - np.arange(len(t))).max())
    f["burst60"] = pd.Series(burst)
    return f.fillna(0)


def _components(df, members):
    sub, H = df[df.customer_id.isin(members)], nx.Graph()
    H.add_nodes_from(members)
    for col in ("device_id", "ip24", "instrument_id"):
        for _, grp in sub.groupby(col).customer_id:
            u = list(grp.unique())
            H.add_edges_from((u[0], x) for x in u[1:])
    return [sorted(c) for c in nx.connected_components(H)]


def detect_rings(df, now, eps=2.0, min_samples=3):
    """DBSCAN over behavioural + shared-infrastructure features finds dense *seeds*; graph closure then
    recovers every customer connected to a seed through shared device / IP / instrument.
    Output is a list of *candidate* risk rings, never a fraud verdict."""
    f = customer_features(df, now)
    cand = f[(f.shared_dev > 0) | (f.shared_ip > 0) | (f.shared_inst > 0)].copy()
    if len(cand) < min_samples:
        return [], f
    X = StandardScaler().fit_transform(cand[FEATS].clip(upper=8))
    cand["cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    seeds = set(cand.index[cand.cluster != -1])
    rings = []
    for comp in _components(df, list(cand.index)):
        if len(comp) >= 3 and len(seeds & set(comp)) >= 2:
            rings.append(comp)
    return rings, f


def _clip(x, lo=0.0, hi=1.0): return float(min(max(x, lo), hi))


def ring_signals(df, members, now, benes) -> dict:
    d = df[df.customer_id.isin(members)]
    top = lambda col: (d.groupby(col).customer_id.nunique().sort_values(ascending=False))
    dv, ip, ins = top("device_id"), top("ip24"), top("instrument_id")
    r7 = d[d.ts >= now - pd.Timedelta(days=7)]
    r7 = r7 if len(r7) else d
    amt = r7.groupby("beneficiary_id").amount.sum().sort_values(ascending=False)
    tb = amt.index[0]; conc = float(amt.iloc[0] / amt.sum())
    t = np.sort(secs(r7[r7.beneficiary_id == tb].ts))
    window = None
    if len(t) >= 3:
        k = max(3, math.ceil(0.6 * len(t)))
        window = float(min(t[i + k - 1] - t[i] for i in range(len(t) - k + 1)) / 60)
    d24 = d[(d.ts >= now - pd.Timedelta(hours=24)) & (d.beneficiary_id == tb)].amount
    cv = float(d24.std() / d24.mean()) if len(d24) >= 5 else None
    old = d[d.ts < now - pd.Timedelta(days=7)]
    prior_n = int(len(old)); prior_share = float((old.beneficiary_id == tb).mean()) if prior_n else 0.0
    # --- 6-hour escalation signal (prototype heuristic)
    r6 = d[d.ts >= now - pd.Timedelta(hours=6)]
    base = d[(d.ts < now - pd.Timedelta(hours=6)) & (d.ts >= now - pd.Timedelta(days=14))].amount.sum() / 56
    amt6 = float(r6.amount.sum())
    ratio = amt6 / max(base, 1.0)
    rs = _clip(math.log2(max(ratio, 1)) / 6)
    rc = float(r6[r6.beneficiary_id == tb].amount.sum() / amt6) if amt6 > 0 else 0.0
    bs = _clip(1 - window / 360) if window is not None else 0.0
    esc = 100 * (0.4 * rs + 0.3 * rc * bs + 0.3 * bs)
    structural = 100 * (0.25 * _clip(dv.iloc[0] / 4) + 0.10 * _clip(ip.iloc[0] / 5) + 0.15 * _clip(ins.iloc[0] / 3)
                        + 0.25 * conc + 0.15 * bs + 0.10 * _clip(len(members) / 5))
    return {"members": sorted(members), "n_members": len(members),
            "max_on_device": int(dv.iloc[0]), "top_device": dv.index[0],
            "max_on_ip": int(ip.iloc[0]), "top_ip": ip.index[0],
            "max_on_inst": int(ins.iloc[0]), "top_inst": ins.index[0],
            "top_beneficiary": tb, "concentration": round(conc, 3),
            "exposure_inr": round(float(amt.iloc[0]), 2), "total_7d_inr": round(float(amt.sum()), 2),
            "n_top_txns": int(len(t)), "window_minutes": None if window is None else round(window, 1),
            "amount_cv": None if cv is None else round(cv, 4),
            "prior_share": round(prior_share, 3), "prior_n": prior_n,
            "structural": round(structural, 1), "escalation": round(esc, 1),
            "esc_parts": {"volume_ratio_score": round(rs, 2), "recent_concentration": round(rc, 2), "burst_score": round(bs, 2)},
            "recent6_inr": round(amt6, 2),
            "source_refs": [[r.payment_id, int(r.revision)] for r in d.itertuples()]}


def case_graph(df, ring, benes, max_txn_nodes=12) -> dict:
    d = df[df.customer_id.isin(ring["members"])]
    nodes, edges = {}, []
    def N(i, t, label, **k): nodes.setdefault(i, {"id": i, "type": t, "label": label, **k})
    for c in ring["members"]:
        N(f"customer:{c}", "customer", c); N(f"account:ACC_{c}", "account", f"ACC_{c}")
        edges.append({"source": f"customer:{c}", "target": f"account:ACC_{c}", "rel": "owns"})
    for col, t, rel in [("device_id", "device", "uses_device"), ("ip24", "ip", "seen_on_ip"), ("instrument_id", "instrument", "pays_with")]:
        for c, v in d[["customer_id", col]].drop_duplicates().itertuples(index=False):
            shared = d[d[col] == v].customer_id.nunique() > 1
            if shared or col == "instrument_id":
                N(f"{t}:{v}", t, v, shared=bool(shared)); edges.append({"source": f"customer:{c}", "target": f"{t}:{v}", "rel": rel})
    for b in d.groupby("beneficiary_id").amount.sum().sort_values(ascending=False).index[:3]:
        bi = benes.get(b, {}); N(f"beneficiary:{b}", "beneficiary", b, verified=bi.get("verified"), age_days=bi.get("age_days"))
        for c, s in d[d.beneficiary_id == b].groupby("customer_id").amount.sum().items():
            edges.append({"source": f"customer:{c}", "target": f"beneficiary:{b}", "rel": "transfers_to", "amount": round(float(s), 2)})
    big = d[d.beneficiary_id == ring["top_beneficiary"]].sort_values("amount", ascending=False).head(max_txn_nodes)
    for r in big.itertuples():
        N(f"txn:{r.payment_id}", "transaction", r.payment_id, amount=float(r.amount))
        edges.append({"source": f"customer:{r.customer_id}", "target": f"txn:{r.payment_id}", "rel": "initiated"})
        edges.append({"source": f"txn:{r.payment_id}", "target": f"beneficiary:{r.beneficiary_id}", "rel": "settles_to"})
    return {"nodes": list(nodes.values()), "edges": edges}


def timeline(df, ring, now, hours=72, limit=80):
    d = df[(df.customer_id.isin(ring["members"])) & (df.ts >= now - pd.Timedelta(hours=hours))].sort_values("ts").tail(limit)
    return [{"ts": r.ts.isoformat(), "payment_id": r.payment_id, "customer_id": r.customer_id,
             "beneficiary_id": r.beneficiary_id, "amount": float(r.amount)} for r in d.itertuples()]
