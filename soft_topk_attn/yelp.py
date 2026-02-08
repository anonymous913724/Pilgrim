#!/usr/bin/env python3
"""

- Build a unified (homogeneous) interaction graph per snapshot/month
- Unified indexing: businesses first [0..Nb-1], then users [Nb..Nb+Nu-1]
- Edges: user<->business review edges for that month (bidirectional by default)
- Edge attributes: rating stars (edge_attr[:,0]) if present

This reads the .pt produced by yelp_process.py

Provides a PyG-Temporal-like interface:
  - len(ds)
  - ds.get_hetero(t): HeteroData
  - ds.get_unified(t): Data (unified bipartite interaction graph)
  - ds.horizon_pair(t,h): (Data_t, Data_{t+h})
  - ds.iter_horizon(h): iterator

Extra helper (TGNN-friendly):
  get_event_list(t) -> (src, dst, edge_attr[, t_event])
returns *directed* events for that month (default user->business only).

- ds.to_dynamic_graph_temporal_signal(...) -> torch_geometric_temporal.signal.DynamicGraphTemporalSignal
  * builds semantic-aligned unified features (+optional type onehot)
  * builds undirected weighted edges via duplicating (u->b) and (b->u) with weight=rating
  * unified indexing: business first, then users (Nb+u)

  - Instead of padding business.x/user.x then concatenating, build a unified feature schema:
      [log1p_review_count, avg_stars, activity_score, popularity_score, log1p_degree] (+ type onehot)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

try:
    from torch_geometric.data import HeteroData, Data
except Exception as e:
    raise RuntimeError("This script requires torch_geometric.") from e

try:
    from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
except Exception as e:
    DynamicGraphTemporalSignal = None
    _TGT_IMPORT_ERROR = e


# ============================================================
# Helpers for semantic-aligned unified feature construction
# ============================================================

def _safe_idx_by_name(names: List[str], candidates: List[str]) -> Optional[int]:
    """
    Try to locate a channel index by fuzzy name matching.
    Returns index if found else None.
    """
    if not names:
        return None
    low = [str(n).lower() for n in names]
    for cand in candidates:
        c = cand.lower()
        for i, n in enumerate(low):
            if c in n:
                return i
    return None


def _pick_col(x: torch.Tensor, idx: Optional[int], fallback_idx: Optional[int] = None) -> torch.Tensor:
    """
    Returns x[:, idx] if idx is valid else fallback else zeros.
    """
    n = x.size(0)
    dev = x.device
    if idx is not None and 0 <= idx < x.size(1):
        return x[:, idx].to(dtype=torch.float32)
    if fallback_idx is not None and 0 <= fallback_idx < x.size(1):
        return x[:, fallback_idx].to(dtype=torch.float32)
    return torch.zeros((n,), dtype=torch.float32, device=dev)


def _log1p_safe(v: torch.Tensor) -> torch.Tensor:
    """
    Apply log1p if values look non-negative; if already looks like log-space,
    keep as-is. (Heuristic: if max is small-ish, assume it's already log.)
    """
    v = v.to(dtype=torch.float32)
    if v.numel() == 0:
        return v
    vmax = float(v.max().item())
    vmin = float(v.min().item())
    # If values are already in a log-ish range (e.g., 0..~10), don't log again.
    # If values look like raw counts (e.g., can be large), apply log1p.
    if vmin >= 0.0 and vmax > 20.0:
        return torch.log1p(v.clamp_min(0.0))
    return v


def _compute_log1p_degree(
    Nb: int,
    Nu: int,
    e_user_to_biz: torch.Tensor,  # [2, E] user->business with local ids
    device: torch.device,
) -> torch.Tensor:
    """
    Degree in the unified bipartite graph for this snapshot.
    compute degree from user->business edges, then mirror for bidirectional use.
    Return: log1p(deg) for all nodes [Nb+Nu]
    """
    N = Nb + Nu
    if e_user_to_biz.numel() == 0:
        return torch.zeros((N,), dtype=torch.float32, device=device)

    u = e_user_to_biz[0].to(dtype=torch.long, device=device) + Nb  # unified user ids
    b = e_user_to_biz[1].to(dtype=torch.long, device=device)       # unified business ids
    # Count each review edge once for degree (undirected conceptually)
    deg = torch.zeros((N,), dtype=torch.float32, device=device)
    ones = torch.ones((u.numel(),), dtype=torch.float32, device=device)
    deg.index_add_(0, u, ones)
    deg.index_add_(0, b, ones)
    return torch.log1p(deg)


def _build_unified_semantic_x(
    xb: torch.Tensor,
    xu: torch.Tensor,
    e_user_to_biz: torch.Tensor,
    biz_names: Optional[List[str]] = None,
    user_names: Optional[List[str]] = None,
    add_type_onehot: bool = True,
) -> torch.Tensor:
    """
    Build semantic-aligned unified node features for one snapshot.

    Output base features (5 dims):
      0) log1p_review_count
      1) avg_stars
      2) activity_score
      3) popularity_score
      4) log1p_degree (in bipartite review graph)
    then optionally append type onehot (2 dims) -> total 7 dims.

    try to pick columns using meta.channel_names_* if available; otherwise fall back to
    reasonable defaults based on common layouts:
      - business: review_count ~ col0, stars ~ col1, activity ~ col2, popularity ~ col3
      - user: review_count ~ col0, elite/fans/useful maybe ~ col1/col3, avg_stars ~ col2
    """
    if xb.dim() != 2 or xu.dim() != 2:
        raise ValueError("Expected business.x and user.x to be 2D tensors.")

    Nb = int(xb.size(0))
    Nu = int(xu.size(0))
    device = xb.device

    biz_names = biz_names or []
    user_names = user_names or []

    # --- find candidate indices by name (best-effort) ---
    # Business
    b_idx_reviews = _safe_idx_by_name(biz_names, ["review", "reviews", "review_count"])
    b_idx_stars   = _safe_idx_by_name(biz_names, ["stars", "avg_stars", "rating"])
    b_idx_checkin = _safe_idx_by_name(biz_names, ["checkin", "checkins"])
    b_idx_photos  = _safe_idx_by_name(biz_names, ["photo", "photos"])

    # User
    u_idx_reviews = _safe_idx_by_name(user_names, ["review", "reviews", "review_count"])
    u_idx_stars   = _safe_idx_by_name(user_names, ["stars", "avg_stars", "rating"])
    u_idx_fans    = _safe_idx_by_name(user_names, ["fan", "fans"])
    u_idx_useful  = _safe_idx_by_name(user_names, ["useful"])
    u_idx_elite   = _safe_idx_by_name(user_names, ["elite"])

    # --- pull raw columns with fallback positions ---
    # business: [col0 ~ log1p(reviews), col1 ~ stars, col2 ~ activity, col3 ~ popularity]
    # user:     [col0 ~ log1p(reviews), col2 ~ avg_stars] and other cols may be sparse.
    b_reviews = _pick_col(xb, b_idx_reviews, fallback_idx=0)
    b_stars   = _pick_col(xb, b_idx_stars,   fallback_idx=1)
    b_checkin = _pick_col(xb, b_idx_checkin, fallback_idx=2)
    b_photos  = _pick_col(xb, b_idx_photos,  fallback_idx=3)

    u_reviews = _pick_col(xu, u_idx_reviews, fallback_idx=0)
    # user avg stars: try by name, else fallback to col2
    u_stars   = _pick_col(xu, u_idx_stars,   fallback_idx=2 if xu.size(1) > 2 else None)

    # user popularity: fans/useful, else 0
    u_pop = None
    if u_idx_fans is not None:
        u_pop = _pick_col(xu, u_idx_fans, None)
    elif u_idx_useful is not None:
        u_pop = _pick_col(xu, u_idx_useful, None)
    else:
        u_pop = torch.zeros((Nu,), dtype=torch.float32, device=device)

    # user activity: just reviews (strong baseline)
    # business activity: checkins if present else reviews
    b_activity = b_checkin
    # if checkin column is missing/zeros, fallback to reviews
    if float(b_activity.abs().sum().item()) == 0.0:
        b_activity = b_reviews

    # business popularity: photos if present else reviews
    b_popularity = b_photos
    if float(b_popularity.abs().sum().item()) == 0.0:
        b_popularity = b_reviews

    # --- apply log1p heuristics on count-like signals ---
    b_reviews_l = _log1p_safe(b_reviews)
    u_reviews_l = _log1p_safe(u_reviews)

    b_activity_l = _log1p_safe(b_activity)
    u_activity_l = _log1p_safe(u_reviews)  # user activity: reviews

    b_pop_l = _log1p_safe(b_popularity)
    u_pop_l = _log1p_safe(u_pop)

    # --- degree feature from edges ---
    deg_l = _compute_log1p_degree(Nb, Nu, e_user_to_biz, device=device)
    deg_b = deg_l[:Nb]
    deg_u = deg_l[Nb:]

    # --- assemble base 5-d features for business and user ---
    xb5 = torch.stack([b_reviews_l, b_stars, b_activity_l, b_pop_l, deg_b], dim=1)  # [Nb, 5]
    xu5 = torch.stack([u_reviews_l, u_stars, u_activity_l, u_pop_l, deg_u], dim=1)  # [Nu, 5]

    x = torch.cat([xb5, xu5], dim=0)  # [Nb+Nu, 5]

    if add_type_onehot:
        tb = x.new_zeros((Nb, 2))
        tu = x.new_zeros((Nu, 2))
        tb[:, 0] = 1.0  # business
        tu[:, 1] = 1.0  # user
        x = torch.cat([x, torch.cat([tb, tu], dim=0)], dim=1)  # [N, 7]

    return x


# ============================================================
# Meta + dataset
# ============================================================

@dataclass
class YelpTemporalMeta:
    start_ym: str = ""
    months_count: int = 0
    months: List[str] = None
    biz_id_to_idx: Dict[str, int] = None
    user_id_to_idx: Dict[str, int] = None
    channel_names_business: List[str] = None
    channel_names_user: List[str] = None
    category_vocab: List[str] = None
    num_categories: int = 0
    edge_defs: List[str] = None

    @staticmethod
    def from_dict(d: Dict) -> "YelpTemporalMeta":
        return YelpTemporalMeta(
            start_ym=str(d.get("start_ym", "")),
            months_count=int(d.get("months_count", len(d.get("months", [])))),
            months=list(d.get("months", [])) if d.get("months", None) is not None else [],
            biz_id_to_idx=dict(d.get("biz_id_to_idx", {})),
            user_id_to_idx=dict(d.get("user_id_to_idx", {})),
            channel_names_business=list(d.get("channel_names_business", [])),
            channel_names_user=list(d.get("channel_names_user", [])),
            category_vocab=list(d.get("category_vocab", [])),
            num_categories=int(d.get("num_categories", 0)),
            edge_defs=list(d.get("edge_defs", [])),
        )


class YelpBipartiteTemporal:
    """
    Monthly snapshots:
      - raw snapshots are HeteroData with node types 'business' and 'user'
      - dynamic edges: ('user','rev','business') with edge_rating

    Unified graph per snapshot:
      - nodes: [business; user]
      - edges: user<->business review edges (bidirectional optional)
      - edge_attr: rating stars as (E,1) if available

    Event list per snapshot:
      - directed events (default: user->business only)
      - returns tensors ready for TGN/TGAT style encoders
    """

    def __init__(self, pt_path: str, map_location: str = "cpu"):
        obj = torch.load(pt_path, map_location=map_location)
        if not isinstance(obj, dict) or "data_list" not in obj:
            raise ValueError(
                f"Invalid .pt: expected dict with 'data_list'. Got {type(obj)} keys={list(obj.keys()) if isinstance(obj, dict) else None}"
            )
        self.data_list: List[HeteroData] = obj["data_list"]
        self.meta = YelpTemporalMeta.from_dict(obj.get("meta", {}))

        if len(self.data_list) == 0:
            raise ValueError("Empty data_list in .pt")

        d0 = self.data_list[0]
        if "business" not in d0.node_types or "user" not in d0.node_types:
            raise ValueError(f"Expected node types ['business','user'], got {d0.node_types}")
        if ("user", "rev", "business") not in d0.edge_types:
            raise ValueError(f"Expected edge type ('user','rev','business'), got {d0.edge_types}")

    def __len__(self) -> int:
        return len(self.data_list)

    def month_str(self, t: int) -> str:
        d = self.get_hetero(t)
        if "month" in d:
            return str(d["month"])
        if self.meta.months and t < len(self.meta.months):
            return str(self.meta.months[t])
        return str(t)

    def get_hetero(self, t: int) -> HeteroData:
        if t < 0 or t >= len(self.data_list):
            raise IndexError(f"t out of range: {t} (T={len(self.data_list)})")
        return self.data_list[t]

    def get_unified(
        self,
        t: int,
        bidirectional: bool = True,
        include_edge_attr: bool = True,
        add_type_onehot: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Data:
        """
        Build unified bipartite graph for month t.

        Unified indexing:
          business: 0..Nb-1
          user: Nb..Nb+Nu-1

        Build semantic-aligned unified features:
          [log1p_review_count, avg_stars, activity, popularity, log1p_degree] (+ type onehot)
        """
        d = self.get_hetero(t)

        xb = d["business"].x
        xu = d["user"].x
        Nb = int(xb.size(0))
        Nu = int(xu.size(0))

        store = d["user", "rev", "business"]
        e = store.edge_index  # [2, E]: user_idx -> business_idx

        # --- semantic-aligned features  ---
        x = _build_unified_semantic_x(
            xb=xb,
            xu=xu,
            e_user_to_biz=e,
            biz_names=self.meta.channel_names_business,
            user_names=self.meta.channel_names_user,
            add_type_onehot=add_type_onehot,
        )
        if device is not None:
            x = x.to(device)

        # --- edges ---
        if e.numel() == 0:
            g = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long, device=device))
            g.month = self.month_str(t)
            g.Nb, g.Nu = Nb, Nu
            return g

        u = e[0].to(dtype=torch.long)
        b = e[1].to(dtype=torch.long)

        # unified mapping
        u_unified = u + Nb
        b_unified = b

        edge_index = torch.stack([u_unified, b_unified], dim=0)  # user -> business

        edge_attr = None
        if include_edge_attr and hasattr(store, "edge_rating") and store.edge_rating is not None:
            r = store.edge_rating
            if r.numel() == edge_index.size(1):
                edge_attr = r.to(dtype=torch.float32).view(-1, 1)

        if bidirectional:
            rev_edge = torch.stack([b_unified, u_unified], dim=0)  # business -> user
            edge_index = torch.cat([edge_index, rev_edge], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        if device is not None:
            edge_index = edge_index.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

        g = Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            g.edge_attr = edge_attr

        g.month = self.month_str(t)
        g.Nb, g.Nu = Nb, Nu
        return g

    def get_event_list(
        self,
        t: int,
        direction: str = "u2b",
        include_edge_attr: bool = True,
        include_time: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    ]:
        """
        Return a TGNN-friendly event list for month t.

        direction:
          - "u2b": events are user -> business (recommended for TGN/TGAT)
          - "b2u": business -> user
          - "both": concatenated u2b + b2u (bidirectional)

        Output tensors are in unified indexing:
          business: 0..Nb-1
          user: Nb..Nb+Nu-1

        Returns:
          src: [E]
          dst: [E]
          edge_attr: [E, F] or None (rating stars as [E,1] if present)
          (optional) t_event: [E] all filled with the integer month index t
        """
        d = self.get_hetero(t)

        xb = d["business"].x
        Nb = int(xb.size(0))

        store = d["user", "rev", "business"]
        e = store.edge_index  # [2, E]: user_idx -> business_idx

        if e.numel() == 0:
            src = torch.empty((0,), dtype=torch.long)
            dst = torch.empty((0,), dtype=torch.long)
            edge_attr = None
            if include_time:
                t_event = torch.empty((0,), dtype=torch.long)
                if device is not None:
                    src, dst, t_event = src.to(device), dst.to(device), t_event.to(device)
                return src, dst, edge_attr, t_event
            if device is not None:
                src, dst = src.to(device), dst.to(device)
            return src, dst, edge_attr

        u = e[0].to(dtype=torch.long)
        b = e[1].to(dtype=torch.long)

        u_unified = u + Nb
        b_unified = b

        # edge attributes (rating)
        edge_attr = None
        if include_edge_attr and hasattr(store, "edge_rating") and store.edge_rating is not None:
            r = store.edge_rating
            if r.numel() == u.numel():
                edge_attr = r.to(dtype=torch.float32).view(-1, 1)

        if direction == "u2b":
            src = u_unified
            dst = b_unified
        elif direction == "b2u":
            src = b_unified
            dst = u_unified
        elif direction == "both":
            src = torch.cat([u_unified, b_unified], dim=0)
            dst = torch.cat([b_unified, u_unified], dim=0)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        else:
            raise ValueError("direction must be one of: 'u2b', 'b2u', 'both'")

        if include_time:
            t_event = torch.full((src.numel(),), int(t), dtype=torch.long)
            if device is not None:
                src, dst, t_event = src.to(device), dst.to(device), t_event.to(device)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)
            return src, dst, edge_attr, t_event

        if device is not None:
            src, dst = src.to(device), dst.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

        return src, dst, edge_attr

    def to_dynamic_graph_temporal_signal(
        self,
        bidirectional: bool = True,
        add_type_onehot: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "DynamicGraphTemporalSignal":
        """
        Build torch_geometric_temporal.signal.DynamicGraphTemporalSignal.

        Use semantic-aligned unified features:
          [log1p_review_count, avg_stars, activity, popularity, log1p_degree] (+ type onehot)
        """
        if DynamicGraphTemporalSignal is None:
            raise ImportError(
                "torch_geometric_temporal is not available; cannot create DynamicGraphTemporalSignal. "
                f"Original import error: {_TGT_IMPORT_ERROR}"
            )

        edge_indices: List[torch.Tensor] = []
        edge_weights: List[torch.Tensor] = []
        features: List[torch.Tensor] = []

        T = len(self)
        for t in range(T):
            d = self.get_hetero(t)
            xb = d["business"].x
            xu = d["user"].x
            Nb = int(xb.size(0))
            Nu = int(xu.size(0))

            store = d["user", "rev", "business"]
            e = store.edge_index  # [2, E] user -> business

            # --- semantic-aligned features ---
            x = _build_unified_semantic_x(
                xb=xb,
                xu=xu,
                e_user_to_biz=e,
                biz_names=self.meta.channel_names_business,
                user_names=self.meta.channel_names_user,
                add_type_onehot=add_type_onehot,
            )
            if device is not None:
                x = x.to(device)
            features.append(x)

            # --- edges ---
            if e.numel() == 0:
                edge_indices.append(torch.empty((2, 0), dtype=torch.long, device=device))
                edge_weights.append(torch.empty((0,), dtype=torch.float32, device=device))
                continue

            u = e[0].to(dtype=torch.long)
            b = e[1].to(dtype=torch.long)

            # unified mapping: user becomes Nb + u; business stays b
            u_uni = u + Nb
            b_uni = b

            ei_fwd = torch.stack([u_uni, b_uni], dim=0)  # user -> business

            # rating weights
            w = None
            if hasattr(store, "edge_rating") and store.edge_rating is not None and store.edge_rating.numel() > 0:
                w0 = store.edge_rating.view(-1).to(dtype=torch.float32)
                if w0.numel() == ei_fwd.size(1):
                    w = w0
            if w is None:
                w = torch.ones((ei_fwd.size(1),), dtype=torch.float32)

            if bidirectional:
                ei_rev = torch.stack([b_uni, u_uni], dim=0)  # business -> user
                ei = torch.cat([ei_fwd, ei_rev], dim=1)
                ew = torch.cat([w, w], dim=0)
            else:
                ei = ei_fwd
                ew = w

            if device is not None:
                ei = ei.to(device)
                ew = ew.to(device)

            edge_indices.append(ei)
            edge_weights.append(ew)

        # PyG Temporal requires targets to be a sequence of length T (can be None entries).
        targets = [None] * len(features)
        return DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets
        )

    def horizon_pair(self, t: int, h: int, **kwargs) -> Tuple[Data, Data]:
        if h <= 0:
            raise ValueError("h must be >= 1")
        t2 = t + h
        if t2 >= len(self):
            raise IndexError(f"t+h out of range: t={t}, h={h}, T={len(self)}")
        return self.get_unified(t, **kwargs), self.get_unified(t2, **kwargs)

    def iter_horizon(self, h: int, start: int = 0, end: Optional[int] = None, **kwargs) -> Iterator[Tuple[int, Data, Data]]:
        if h <= 0:
            raise ValueError("h must be >= 1")
        if end is None:
            end = len(self) - h
        end = min(end, len(self) - h)
        for t in range(start, end):
            yield t, self.get_unified(t, **kwargs), self.get_unified(t + h, **kwargs)


def demo(pt_path: str, t: int = 0):
    ds = YelpBipartiteTemporal(pt_path)
    print("=== YelpBipartiteTemporal Demo ===")
    print("Snapshots (months) T =", len(ds))
    print("Month[t] =", ds.month_str(t))

    d = ds.get_hetero(t)
    Nb = d["business"].num_nodes
    Nu = d["user"].num_nodes
    E = d["user", "rev", "business"].edge_index.size(1)
    print(f"Hetero snapshot: Nb={Nb} Nu={Nu} E_rev={E} (user->business)")

    g = ds.get_unified(t, bidirectional=True, include_edge_attr=True, add_type_onehot=True)
    print(f"Unified snapshot: N={g.num_nodes} (Nb+Nu={g.Nb}+{g.Nu})  E={g.edge_index.size(1)} (bidirectional=True)")
    print("x shape:", tuple(g.x.shape))

    # print feature meaning
    print("Feature columns (unified):")
    print("  [0]=log1p_review_count  [1]=avg_stars  [2]=activity  [3]=popularity  [4]=log1p_degree  [5:7]=type_onehot")

    if hasattr(g, "edge_attr"):
        ea = g.edge_attr
        print(
            "edge_attr shape:",
            tuple(ea.shape),
            "rating min/mean/max:",
            float(ea.min().item()),
            float(ea.mean().item()),
            float(ea.max().item()),
        )
    else:
        print("edge_attr: (none)")

    # event list demo
    src, dst, ea = ds.get_event_list(t, direction="u2b", include_edge_attr=True)
    print(f"Event list (u2b): E={src.numel()}")
    if src.numel() > 0:
        print("src range:", int(src.min().item()), "..", int(src.max().item()))
        print("dst range:", int(dst.min().item()), "..", int(dst.max().item()))
        if ea is not None:
            print(
                "event edge_attr (rating) min/mean/max:",
                float(ea.min().item()),
                float(ea.mean().item()),
                float(ea.max().item()),
            )
        print("first 5 events:", list(zip(src[:5].tolist(), dst[:5].tolist())))

    print("business idx range: [0,", g.Nb - 1, "]")
    print("user idx range:    [", g.Nb, ",", g.Nb + g.Nu - 1, "]")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to Yelp preprocessed .pt produced by yelp_process.py")
    ap.add_argument("--t", type=int, default=0, help="Snapshot index to demo")
    ap.add_argument("--demo", action="store_true", help="Run demo printing graph info")
    ap.add_argument("--dgts", action="store_true", help="Also build DynamicGraphTemporalSignal and print one snapshot")
    args = ap.parse_args()

    if args.demo:
        demo(args.pt, t=args.t)
        if args.dgts:
            ds = YelpBipartiteTemporal(args.pt)
            dgts = ds.to_dynamic_graph_temporal_signal(bidirectional=True, add_type_onehot=True)
            sample_snap = dgts[args.t]
            print("Sample snapshot {}:\n{}".format(args.t, sample_snap))
            print("=== DynamicGraphTemporalSignal snapshot ===")
            print(
                "x (first 5):\n", sample_snap.x[:5],
                "\nx (last 5):\n", sample_snap.x[-5:],
                "\nedge_index (first 10 edges):\n", sample_snap.edge_index[:, :10],
                "\nedge_attr/weights (first 10):\n", sample_snap.edge_attr[:10] if hasattr(sample_snap, "edge_attr") else None
            )
    else:
        ds = YelpBipartiteTemporal(args.pt)
        print("Loaded. T =", len(ds))
