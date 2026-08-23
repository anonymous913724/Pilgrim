"""
 
Build a PyTorch-Geometric-Temporal dataset:
  - Static spatial graph: U.S. county adjacency (Census county adjacency file)
  - Dynamic node features: 7-day snapshot features from NYT county COVID data
  - Date range: 2021-11-01 to 2022-04-30
  - Snapshot length: 7 days (each snapshot is a 7-day window)

Outputs:
  - dataset: StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
  - meta: dict with fips mappings and snapshot windows
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# torch_geometric_temporal
try:
    from torch_geometric_temporal.signal import StaticGraphTemporalSignal
except ImportError as e:
    raise ImportError(
        "torch_geometric_temporal is required. Install e.g. `pip install torch-geometric-temporal` "
    ) from e


@dataclass
class DatasetMeta:
    fips_to_idx: Dict[str, int]
    idx_to_fips: List[str]
    snapshot_starts: List[pd.Timestamp]
    snapshot_ends: List[pd.Timestamp]
    date_start: str
    date_end: str
    snapshot_days: int


def _load_and_clean_nyt_county_data(
    nyt_us_counties_csv: str,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Load NYT us-counties.csv and clean:
      - parse date
      - drop missing fips
      - keep date range
      - ensure fips is string (5 digits with leading zeros)
      - sort for diff
    """
    if not os.path.exists(nyt_us_counties_csv):
        raise FileNotFoundError(f"NYT file not found: {nyt_us_counties_csv}")

    df = pd.read_csv(nyt_us_counties_csv, dtype={"fips": "string"})
    if "date" not in df.columns or "cases" not in df.columns or "deaths" not in df.columns:
        raise ValueError("NYT file must contain columns: date, cases, deaths (and usually county, state, fips).")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Remove invalid / missing entries: drop missing fips
    df = df.dropna(subset=["fips"])
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    # Filter date range
    start_ts = pd.to_datetime(date_start)
    end_ts = pd.to_datetime(date_end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    # Sort for diff computations
    df = df.sort_values(["fips", "date"]).reset_index(drop=True)

    # Ensure numeric
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df = df.dropna(subset=["cases", "deaths"])

    return df


def _compute_daily_new_from_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily new cases/deaths per county from cumulative counts.
    Handles retrospective corrections by clipping negative diffs to 0.
    """
    df = df.copy()
    df["new_cases"] = df.groupby("fips")["cases"].diff()
    df["new_deaths"] = df.groupby("fips")["deaths"].diff()

    # Fill first day diffs with the cumulative (often 0 or small)
    df["new_cases"] = df["new_cases"].fillna(df["cases"])
    df["new_deaths"] = df["new_deaths"].fillna(df["deaths"])

    # Clip negatives due to corrections
    df["new_cases"] = df["new_cases"].clip(lower=0)
    df["new_deaths"] = df["new_deaths"].clip(lower=0)

    return df


def _build_snapshot_windows(
    date_start: str,
    date_end: str,
    snapshot_days: int = 7,
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Build non-overlapping snapshot windows:
      [start, start+snapshot_days-1], [next_start, ...]
    """
    start_ts = pd.to_datetime(date_start)
    end_ts = pd.to_datetime(date_end)

    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []

    cur = start_ts
    delta = pd.Timedelta(days=snapshot_days - 1)

    while cur <= end_ts:
        window_end = cur + delta
        if window_end > end_ts:
            break  # drop last partial window to keep consistent 7-day snapshots
        starts.append(cur)
        ends.append(window_end)
        cur = cur + pd.Timedelta(days=snapshot_days)

    if len(starts) == 0:
        raise ValueError("No full snapshot windows were created. Check date range / snapshot_days.")

    return starts, ends


def _load_and_clean_census_adjacency(census_adjacency_txt: str) -> pd.DataFrame:
    """
    Load Census county adjacency file in pipe-delimited format:
      County Name|County GEOID|Neighbor Name|Neighbor GEOID|Length
    Returns DataFrame with columns: fips, neighbor_fips  (and length if present)
    """
    if not os.path.exists(census_adjacency_txt):
        raise FileNotFoundError(f"Census adjacency file not found: {census_adjacency_txt}")

    adj = pd.read_csv(
        census_adjacency_txt,
        sep="|",
        dtype=str,
        engine="python",
    )

    # Normalize column names (in case of minor variations)
    cols = {c.strip(): c for c in adj.columns}
    required = ["County GEOID", "Neighbor GEOID"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required column '{r}'. Found columns: {list(adj.columns)}")

    adj = adj.rename(columns={
        cols["County GEOID"]: "fips",
        cols["Neighbor GEOID"]: "neighbor_fips",
    })

    # Clean: drop missing, enforce 5-digit strings
    adj = adj.dropna(subset=["fips", "neighbor_fips"]).copy()
    adj["fips"] = adj["fips"].astype(str).str.strip().str.zfill(5)
    adj["neighbor_fips"] = adj["neighbor_fips"].astype(str).str.strip().str.zfill(5)

    # Remove self-loops
    adj = adj[adj["fips"] != adj["neighbor_fips"]].copy()

    # Keep only needed columns (+ optional Length)
    if "Length" in adj.columns:
        adj["Length"] = pd.to_numeric(adj["Length"], errors="coerce")
        # Length can be NaN; keep it if you want weights later
        return adj[["fips", "neighbor_fips", "Length"]].drop_duplicates().reset_index(drop=True)

    return adj[["fips", "neighbor_fips"]].drop_duplicates().reset_index(drop=True)



def build_nyt_covid_static_graph_temporal_signal(
    nyt_us_counties_csv: str,
    census_adjacency_txt: str,
    date_start: str = "2021-11-01",
    date_end: str = "2022-04-30",
    snapshot_days: int = 7,
    feature_mode: str = "cases_only",
) -> Tuple[StaticGraphTemporalSignal, DatasetMeta]:
    """
    Build StaticGraphTemporalSignal dataset.

    feature_mode:
      - "cases_only": feature dim=1 (7-day avg daily new cases)
      - "cases_deaths": feature dim=2 (7-day avg daily new cases, 7-day avg daily new deaths)
    """
    if feature_mode not in {"cases_only", "cases_deaths"}:
        raise ValueError("feature_mode must be one of: {'cases_only','cases_deaths'}")

    # Load NYT
    df = _load_and_clean_nyt_county_data(nyt_us_counties_csv, date_start, date_end)
    df = _compute_daily_new_from_cumulative(df)

    # Build snapshot windows
    starts, ends = _build_snapshot_windows(date_start, date_end, snapshot_days=snapshot_days)

    # Build node universe from NYT FIPS (after dropping missing)
    all_fips = sorted(df["fips"].unique().tolist())
    fips_to_idx = {f: i for i, f in enumerate(all_fips)}
    idx_to_fips = all_fips

    # Load adjacency and align to NYT node universe
    adj = _load_and_clean_census_adjacency(census_adjacency_txt)

    valid = set(all_fips)
    adj = adj[adj["fips"].isin(valid) & adj["neighbor_fips"].isin(valid)].copy()

    # Build undirected unique edges
    undirected = set()
    for a, b in zip(adj["fips"].tolist(), adj["neighbor_fips"].tolist()):
        i, j = fips_to_idx[a], fips_to_idx[b]
        if i == j:
            continue
        if i < j:
            undirected.add((i, j))
        else:
            undirected.add((j, i))

    # Convert to edge_index with both directions
    # PyG Temporal expects edge_index as numpy array shape [2, num_edges_directed]
    edges_i = []
    edges_j = []
    for i, j in sorted(undirected):
        edges_i.extend([i, j])
        edges_j.extend([j, i])

    edge_index = np.array([edges_i, edges_j], dtype=np.int64)
    edge_weight = None  # unweighted adjacency

    # Pre-index NYT data for fast window filtering
    # compute per-county mean(new_cases) over each 7-day window.
    df_small = df[["date", "fips", "new_cases", "new_deaths"]].copy()

    # Prepare features per snapshot
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []  # placeholder; not used in your task setup

    # Make sure it have complete county-day coverage; missing days -> treat as 0 new cases
    # pivot inside each window to fill missing counties.
    for w_start, w_end in zip(starts, ends):
        win = df_small[(df_small["date"] >= w_start) & (df_small["date"] <= w_end)].copy()

        # Aggregate: mean over the 7 days (7-day avg of daily new)
        g = win.groupby("fips", as_index=True).agg(
            new_cases_avg=("new_cases", "mean"),
            new_deaths_avg=("new_deaths", "mean"),
        )

        # Reindex to all counties; missing -> 0
        g = g.reindex(idx_to_fips).fillna(0.0)

        if feature_mode == "cases_only":
            x = g[["new_cases_avg"]].to_numpy(dtype=np.float32)
        else:
            x = g[["new_cases_avg", "new_deaths_avg"]].to_numpy(dtype=np.float32)

        features.append(x)

        # Placeholder targets (not used). Keep same shape to satisfy the dataset container.
        targets.append(np.zeros((x.shape[0],), dtype=np.float32))

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    meta = DatasetMeta(
        fips_to_idx=fips_to_idx,
        idx_to_fips=idx_to_fips,
        snapshot_starts=starts,
        snapshot_ends=ends,
        date_start=date_start,
        date_end=date_end,
        snapshot_days=snapshot_days,
    )

    return dataset, meta
