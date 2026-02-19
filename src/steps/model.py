"""
src/steps/model.py
──────────────────
notebooks/v3_MODEL.ipynb 섹션 2.x, 4-6 로직 모듈화.

I/O 없음 – 순수 pandas/numpy/sklearn 계산만.

공개 API:
  add_gate_and_context(df, config)                          -> pd.DataFrame
  compute_z_scores(df, config)                              -> Tuple[pd.DataFrame, List[str]]
  train_isolation_forest(df, z_use_cols, config)            -> IsolationForest
  compute_anomaly_score(df, iso, z_use_cols, score_mask)    -> pd.DataFrame
  compute_risk_score(df, config)                            -> pd.DataFrame
  run_model(df, config, iso_model=None, *, debug=False)     -> Tuple[pd.DataFrame, IsolationForest]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sklearn.ensemble import IsolationForest

from .baseline import (
    BASE_COLS_CANDIDATES,
    STD_FLOOR_MAP,
    DEFAULT_FLOOR,
    winsorize_series,
)

# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────

# 2.2  Context thresholds
PARTIAL_THR = 1.0
TRAVEL_THR  = 2.0
TZ_THR      = 1.0

# 4.x  Z-score
Z_CLIP   = 4.0
Z_NA_THR = 0.60

Z_BASE_CANDIDATES: List[str] = [
    "daily_event_cnt",
    "Screen", "Notif", "UserAct",
    "night_ratio", "hour_entropy",
    "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h", "gap_long_ratio", "overnight_gap",
    "session_total_sec", "session_cnt",
    "step_sum",
]

# 5.x  IsolationForest defaults
IF_CONTAMINATION = 0.05
IF_ESTIMATORS    = 300
IF_SEED          = 42

# 6.x  Risk score defaults
W_Z          = 0.85
W_A          = 0.15
TAU_Q        = 0.90
EMA_ALPHA    = 0.25
HOLD_WINDOW  = 14
HOLD_MINP    = 7
BAD_THR      = 0.70
STABLE_THR   = 0.45
ANCHOR_START = 16
ANCHOR_END   = 30
DRIFT_Z_THR  = 2.0
Z_CLIP_EARLY = 4.0

DISCOUNT = {"NORMAL": 1.00, "PARTIAL": 0.60, "TRAVEL": 0.70}


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _trim_mean(row: pd.Series, keep_ratio: float = 0.8) -> float:
    s = row.dropna().sort_values()
    if len(s) == 0:
        return np.nan
    k = max(1, int(len(s) * keep_ratio))
    return float(s.iloc[:k].mean())


def _resolve_discount(cm: str) -> float:
    """context_mode (예: "PARTIAL|TRAVEL") → 가장 낮은 discount."""
    if pd.isna(cm) or cm == "NORMAL":
        return 1.0
    parts = str(cm).split("|")
    return min(DISCOUNT.get(p.strip(), 1.0) for p in parts)


# ──────────────────────────────────────────────────────────────────────────────
# 2.x  Gate + Context  (notebook 섹션 2.x)
# ──────────────────────────────────────────────────────────────────────────────

def add_gate_and_context(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    notebook 섹션 2.x (cell 8 / 11 / 13 / 15 / 17 / 19):
      1. 타입 표준화
      2. QC 표준화 → qc_any_low_cov, qc_missing_any
      3. Context signals → partial_flag, travel_flag, tz_flag
      4. quality_state: GOOD / LOW_CONF
      5. context_mode
      6. baseline_fit_mask, z_score_eligible

    선행 조건: add_time_context() 완료 (is_weekend, day_idx, cold_stage)
    """
    df = df.copy()
    mc = config.get("model", {})

    # ── cell 8: 타입 표준화 ─────────────────────────────────────────────────
    BOOL_COLS = [
        "has_activity", "is_weekend",
        "qc_core_low_cov", "qc_rhythm_low_cov", "qc_gap_low_cov",
        "qc_meta_low_heartbeat", "qc_meta_retry_warn", "qc_meta_queue_warn",
    ]
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("boolean")

    NUM_COLS = [
        "daily_event_cnt", "unlock_cnt",
        "partial_signal_raw", "travel_signal_raw", "tz_change_signal",
        "night_ratio", "hour_entropy",
        "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h", "gap_long_ratio", "overnight_gap",
        "session_total_sec", "session_cnt",
        "step_sum", "cell_change_cnt", "wifi_change_cnt_est",
        "heartbeat_cnt", "retry_max", "queue_max",
    ]
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ZERO_FILL = ["daily_event_cnt", "unlock_cnt", "step_sum", "session_cnt", "heartbeat_cnt"]
    for c in ZERO_FILL:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if "gap_max" in df.columns:
        df.loc[df["daily_event_cnt"] == 0, "gap_max"] = df["gap_max"].fillna(24.0)

    # ── cell 11: QC 표준화 ──────────────────────────────────────────────────
    QC_COLS = ["qc_core_low_cov", "qc_rhythm_low_cov", "qc_gap_low_cov"]
    df["qc_missing_any"] = df[QC_COLS].isna().any(axis=1)
    for c in QC_COLS:
        df[c] = df[c].astype("boolean").fillna(False)

    META_QC = ["qc_meta_low_heartbeat", "qc_meta_retry_warn", "qc_meta_queue_warn"]
    for c in META_QC:
        if c in df.columns:
            df[c] = df[c].astype("boolean")
        else:
            df[c] = pd.array([pd.NA] * len(df), dtype="boolean")

    df["qc_any_low_cov"] = (
        df["qc_core_low_cov"] | df["qc_rhythm_low_cov"] | df["qc_gap_low_cov"]
    )

    # ── cell 13: Context signals ─────────────────────────────────────────────
    partial_thr = float(mc.get("partial_thr", PARTIAL_THR))
    travel_thr  = float(mc.get("travel_thr",  TRAVEL_THR))
    tz_thr      = float(mc.get("tz_thr",      TZ_THR))

    df["partial_lv"] = pd.to_numeric(df.get("partial_signal_raw", 0), errors="coerce").fillna(0.0)
    df["travel_lv"]  = pd.to_numeric(df.get("travel_signal_raw",  0), errors="coerce").fillna(0.0)
    df["tz_lv"]      = pd.to_numeric(df.get("tz_change_signal",   0), errors="coerce").fillna(0.0)

    df["partial_flag"] = (df["partial_lv"] >= partial_thr).astype("boolean")
    df["tz_flag"]      = (df["tz_lv"]      >= tz_thr).astype("boolean")
    df["travel_flag"]  = (
        (df["travel_lv"] >= travel_thr) | (df["tz_flag"])
    ).astype("boolean")

    # ── cell 15: quality_state ───────────────────────────────────────────────
    qs = np.full(len(df), "GOOD", dtype=object)
    mask_lowcov = df["qc_any_low_cov"].to_numpy() | df["qc_missing_any"].to_numpy()
    qs[mask_lowcov] = "LOW_CONF"
    df["quality_state"] = qs

    # ── cell 17: context_mode ────────────────────────────────────────────────
    m_partial = df["partial_flag"].fillna(False).to_numpy()
    m_travel  = df["travel_flag"].fillna(False).to_numpy()
    cm_list = []
    for i in range(len(df)):
        flags = []
        if m_partial[i]: flags.append("PARTIAL")
        if m_travel[i]:  flags.append("TRAVEL")
        cm_list.append("NORMAL" if not flags else "|".join(flags))
    df["context_mode"]     = cm_list
    df["context_any_warn"] = m_partial | m_travel

    # ── cell 19: baseline_fit_mask / z_score_eligible ────────────────────────
    has_act = df["has_activity"].astype("boolean").fillna(False).to_numpy()
    df["baseline_fit_mask"] = (
        has_act
        & (~df["qc_any_low_cov"].to_numpy())
        & (~df["qc_missing_any"].to_numpy())
        & (~df["partial_flag"].fillna(False).to_numpy())
        & (~df["travel_flag"].fillna(False).to_numpy())
    )
    df["z_score_eligible"] = has_act

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.x  Z-score  (notebook 섹션 4.x)
# ──────────────────────────────────────────────────────────────────────────────

def compute_z_scores(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    notebook 섹션 4.x: baseline 기반 Z-score 계산.

    선행 조건: compute_baseline() 완료
      (baseline_ready, {c}_mean_st_s, {c}_std_st_s)

    Returns: (df_with_z_cols, Z_COLS)
    """
    df = df.copy()
    mc = config.get("model", {})
    z_clip = float(mc.get("z_clip", Z_CLIP))

    z_mask = df["baseline_ready"].fillna(False) & (df["quality_state"] == "GOOD")

    z_base_cols = [
        c for c in Z_BASE_CANDIDATES
        if c in df.columns
        and f"{c}_mean_st_s" in df.columns
        and f"{c}_std_st_s" in df.columns
    ]
    if not z_base_cols:
        raise ValueError("[Z] No Z features with ST baseline. Check compute_baseline().")

    Z_COLS: List[str] = []
    for c in z_base_cols:
        x  = pd.to_numeric(df[c], errors="coerce")
        mu = pd.to_numeric(df[f"{c}_mean_st_s"], errors="coerce")
        sd = pd.to_numeric(df[f"{c}_std_st_s"],  errors="coerce")
        z  = (x - mu) / sd
        z  = z.replace([np.inf, -np.inf], np.nan)
        z  = z_clip * np.tanh(z / z_clip)
        z_col = f"{c}_z"
        df[z_col] = z.where(z_mask, np.nan)
        Z_COLS.append(z_col)

    df.attrs["Z_COLS"] = Z_COLS
    return df, Z_COLS


# ──────────────────────────────────────────────────────────────────────────────
# 5.x  IsolationForest  (notebook 섹션 5.x)
# ──────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(
    df: pd.DataFrame,
    z_use_cols: List[str],
    config: dict,
) -> IsolationForest:
    """notebook 섹션 5.4-5.5: IsolationForest 학습."""
    mc = config.get("model", {})
    contamination = float(mc.get("if_contamination", IF_CONTAMINATION))
    n_estimators  = int(mc.get("if_estimators",    IF_ESTIMATORS))
    seed          = int(mc.get("if_seed",           IF_SEED))

    train_mask = (
        df["baseline_fit_mask"].fillna(False)
        & df["baseline_ready"].fillna(False)
        & (df["quality_state"] == "GOOD")
    )
    X_train = df.loc[train_mask, z_use_cols].fillna(0.0)
    if len(X_train) < 10:
        raise ValueError(f"[IForest] X_train too small: {len(X_train)} rows.")

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_train)
    return iso


def compute_anomaly_score(
    df: pd.DataFrame,
    iso: IsolationForest,
    z_use_cols: List[str],
    score_mask: pd.Series,
) -> pd.DataFrame:
    """notebook 섹션 5.5-5.6: anomaly_score (per-user quantile scaling)."""
    df = df.copy()

    X_score = df.loc[score_mask, z_use_cols].fillna(0.0)
    raw     = -iso.decision_function(X_score)

    df["anomaly_score_raw"] = np.nan
    df.loc[score_mask, "anomaly_score_raw"] = raw

    Q_LOW  = 0.80
    Q_HIGH = 0.99

    df["anomaly_score"] = np.nan
    df["anomaly_flag"]  = pd.array([pd.NA] * len(df), dtype="boolean")

    tmp = df.loc[score_mask, ["uuid", "anomaly_score_raw"]].copy()
    q   = tmp.groupby("uuid")["anomaly_score_raw"].quantile([Q_LOW, Q_HIGH]).unstack()
    q.columns = ["q_low", "q_high"]
    tmp = tmp.join(q, on="uuid")

    den = (tmp["q_high"] - tmp["q_low"]).replace(0, np.nan)
    tmp["anomaly_score"] = ((tmp["anomaly_score_raw"] - tmp["q_low"]) / den).clip(0, 1).fillna(0.0)
    tmp["anomaly_flag"]  = (tmp["anomaly_score_raw"] >= tmp["q_high"]).astype("boolean")

    df.loc[score_mask, "anomaly_score"] = tmp["anomaly_score"].values
    df.loc[score_mask, "anomaly_flag"]  = tmp["anomaly_flag"].values

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6.x  Risk Score  (notebook 섹션 6.x)
# ──────────────────────────────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    notebook 섹션 6.x:
      6.2  risk_mask
      6.3  Z 그룹 대표값 → z_score
      6.4  risk_pre = W_Z*z + W_A*a
      6.5  risk_adj (context discount)
      6.6  risk_score (EMA)
      6.8  early_risk, final_risk, bad_history_flag, stable_hold_flag, drift_flag
    """
    df = df.copy().sort_values(["uuid", "date"]).reset_index(drop=True)
    mc = config.get("model", {})

    w_z          = float(mc.get("w_z",          W_Z))
    w_a          = float(mc.get("w_a",          W_A))
    tau_q        = float(mc.get("tau_q",        TAU_Q))
    ema_alpha    = float(mc.get("ema_alpha",    EMA_ALPHA))
    hold_window  = int(mc.get("hold_window",    HOLD_WINDOW))
    hold_minp    = int(mc.get("hold_minp",      HOLD_MINP))
    bad_thr      = float(mc.get("bad_thr",      BAD_THR))
    stable_thr   = float(mc.get("stable_thr",   STABLE_THR))
    anchor_start = int(mc.get("anchor_start",   ANCHOR_START))
    anchor_end   = int(mc.get("anchor_end",     ANCHOR_END))
    drift_z_thr  = float(mc.get("drift_z_thr",  DRIFT_Z_THR))
    z_clip_early = float(mc.get("z_clip_early", Z_CLIP_EARLY))

    base_cols = [c for c in BASE_COLS_CANDIDATES if c in df.columns]
    floor_map = dict(STD_FLOOR_MAP)
    z_cols    = [c for c in df.columns if c.endswith("_z") and not c.endswith("_z_early")]

    # ── 6.2  risk_mask ──────────────────────────────────────────────────────
    df["risk_mask"] = (
        (df["quality_state"] == "GOOD")
        & df["baseline_ready"].fillna(False)
        & (~df["qc_any_low_cov"].fillna(False))
        & (~df["qc_missing_any"].fillna(False))
    )
    risk_mask = df["risk_mask"].astype(bool)

    # ── 6.3  Z 그룹 대표값 ──────────────────────────────────────────────────
    Z_ABS = df[z_cols].abs()

    Z_GROUPS = {
        "core":    [c for c in ["Screen_z", "Notif_z", "UserAct_z", "daily_event_cnt_z"] if c in z_cols],
        "rhythm":  [c for c in ["night_ratio_z", "hour_entropy_z"] if c in z_cols],
        "gap":     [c for c in ["gap_max_z", "gap_p95_z", "gap_cnt_2h_z", "gap_cnt_6h_z",
                                "gap_long_ratio_z", "overnight_gap_z"] if c in z_cols],
        "session": [c for c in ["session_total_sec_z", "session_cnt_z"] if c in z_cols],
        "mob":     [c for c in ["step_sum_z"] if c in z_cols],
    }
    Z_GROUPS = {k: v for k, v in Z_GROUPS.items() if v}

    for gname, cols in Z_GROUPS.items():
        q = 0.85 if gname == "core" else 0.90
        df[f"z_{gname}_rep"] = Z_ABS[cols].quantile(q=q, axis=1, interpolation="linear")

    group_rep_cols       = [f"z_{g}_rep" for g in Z_GROUPS]
    df["z_group_trim_mean"] = df[group_rep_cols].apply(_trim_mean, axis=1)
    df["z_trim_mean_abs"]   = df["z_group_trim_mean"]
    df["z_max_abs"]         = Z_ABS.max(axis=1)

    ref = df.loc[risk_mask, "z_group_trim_mean"].dropna()
    TAU = max(float(ref.quantile(tau_q)) if len(ref) else 1.0, 1e-6)
    df["z_score"] = (1.0 - np.exp(-df["z_group_trim_mean"] / TAU)).clip(0, 0.999)

    df["anomaly_score"] = pd.to_numeric(
        df.get("anomaly_score", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).clip(0, 1)
    df["a_score"] = df["anomaly_score"].clip(0, 0.999)

    # ── 6.4  risk_pre ────────────────────────────────────────────────────────
    df["risk_pre"] = (w_z * df["z_score"] + w_a * df["a_score"]).clip(0, 1)

    # ── 6.5  risk_adj ────────────────────────────────────────────────────────
    df["risk_adj"] = (
        df["risk_pre"] * df["context_mode"].apply(_resolve_discount)
    ).astype(float)
    df.loc[~risk_mask, ["risk_pre", "risk_adj"]] = np.nan

    # ── 6.6  risk_score (EMA) ────────────────────────────────────────────────
    df["risk_score"] = (
        df.groupby("uuid", sort=False)["risk_adj"]
          .transform(lambda s: s.ewm(alpha=ema_alpha, adjust=False, ignore_na=True).mean())
    )
    df.loc[~risk_mask, "risk_score"] = np.nan
    df = df.copy()

    # ── 6.8a  early_risk ─────────────────────────────────────────────────────
    early_mask = (
        (df["quality_state"] == "GOOD")
        & df["cold_stage"].isin(["WARMUP", "SEMI_READY"])
        & df["early_ready"].fillna(False)
        & (~df["qc_any_low_cov"].fillna(False))
        & (~df["qc_missing_any"].fillna(False))
    )

    early_z_cols: List[str] = []
    for c in base_cols:
        if f"{c}_mean_early_s" not in df.columns:
            continue
        mu = df[f"{c}_mean_early_s"]
        sd = df[f"{c}_std_early_s"]
        z  = (df[c] - mu) / sd
        z  = z.replace([np.inf, -np.inf], np.nan)
        z  = z_clip_early * np.tanh(z / z_clip_early)
        col = f"{c}_z_early"
        df[col] = z.where(early_mask, np.nan)
        early_z_cols.append(col)

    df["early_risk"] = np.nan
    if early_z_cols:
        early_ref_med = df.loc[early_mask, early_z_cols].abs().median()
        early_ref_med = early_ref_med.replace(0, np.nan).fillna(1.0)
        abs_early_norm = df[early_z_cols].abs().div(early_ref_med, axis=1)
        df["z_trim_mean_abs_early"] = abs_early_norm.apply(_trim_mean, axis=1)
        ref_e = df.loc[early_mask, "z_trim_mean_abs_early"].dropna()
        TAU_E = max(float(ref_e.quantile(0.90)) if len(ref_e) else 1.0, 1e-6)
        raw_e     = 1.0 - np.exp(-df["z_trim_mean_abs_early"] / TAU_E)
        raw_e_max = 1.0 - np.exp(-z_clip_early / TAU_E)
        df.loc[early_mask, "early_risk"] = (raw_e / (raw_e_max + 1e-6)).clip(0, 1)

    df["final_risk"] = df["risk_score"]
    df.loc[df["final_risk"].isna(), "final_risk"] = df["early_risk"]

    # ── 6.8b  Recovery flags ─────────────────────────────────────────────────
    df["bad_history_flag"] = (
        df.groupby("uuid", sort=False)["final_risk"]
          .transform(lambda s: s.shift(1).rolling(hold_window, min_periods=hold_minp).max() >= bad_thr)
    ).astype("boolean")

    df["stable_hold_flag"] = (
        df["bad_history_flag"].fillna(False)
        & df["final_risk"].notna()
        & (df["final_risk"] < stable_thr)
    ).astype("boolean")

    # ── 6.8c  Drift flags ────────────────────────────────────────────────────
    DRIFT_WATCH = [c for c in ["daily_event_cnt", "night_ratio", "gap_max"] if c in base_cols]
    anchor_mask = df["day_idx"].between(anchor_start, anchor_end) & df["baseline_fit_mask"]

    for c in DRIFT_WATCH:
        anchor_stats = (
            df.loc[anchor_mask].groupby("uuid")[c]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{c}_anchor_mean", "std": f"{c}_anchor_std"})
        )
        floor = float(floor_map.get(c, DEFAULT_FLOOR))
        anchor_stats[f"{c}_anchor_std"] = anchor_stats[f"{c}_anchor_std"].clip(lower=floor)
        df = df.join(anchor_stats, on="uuid")
        mean_col = f"{c}_mean_final"
        if mean_col in df.columns:
            df[f"{c}_drift_z"] = (df[mean_col] - df[f"{c}_anchor_mean"]) / df[f"{c}_anchor_std"]
        else:
            df[f"{c}_drift_z"] = np.nan

    drift_z_cols = [f"{c}_drift_z" for c in DRIFT_WATCH]
    drift_z_abs  = df[drift_z_cols].abs()
    df["drift_flag"] = (
        (df["day_idx"] > anchor_end)
        & (drift_z_abs.max(axis=1) >= drift_z_thr)
    ).astype("boolean").fillna(False)

    df["drift_top_feature"] = ""
    dv = df["drift_flag"].fillna(False)
    if dv.any():
        df.loc[dv, "drift_top_feature"] = (
            drift_z_abs.loc[dv].idxmax(axis=1)
            .str.replace("_drift_z", "", regex=False)
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 전체 파이프라인 (orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

def run_model(
    df: pd.DataFrame,
    config: dict,
    iso_model: Optional[IsolationForest] = None,
    *,
    debug: bool = False,
) -> Tuple[pd.DataFrame, IsolationForest]:
    """
    model 전체 파이프라인.

    선행 조건:
      1. build_daily_features() 완료
      2. add_time_context()     완료
      3. add_gate_and_context() 완료  (이 함수에서 포함)
      4. compute_baseline()     완료

    iso_model=None  → IForest 신규 학습
    iso_model=<obj> → 기존 모델로 scoring만

    Returns: (df_with_all_model_cols, iso)
    """
    mc = config.get("model", {})
    z_na_thr = float(mc.get("z_na_thr", Z_NA_THR))

    df, z_cols = compute_z_scores(df, config)
    if debug:
        print(f"[MODEL] z_cols={len(z_cols)}")

    z_na_ratio = df[z_cols].isna().mean()
    z_use_cols = [c for c in z_cols if z_na_ratio[c] < z_na_thr]
    if len(z_use_cols) < 3:
        z_use_cols = list(z_cols)

    score_mask = df["baseline_ready"].fillna(False) & (df["quality_state"] == "GOOD")

    if iso_model is None:
        iso = train_isolation_forest(df, z_use_cols, config)
    else:
        iso = iso_model

    df = compute_anomaly_score(df, iso, z_use_cols, score_mask)
    if debug:
        print(f"[MODEL] anomaly_score valid={df['anomaly_score'].notna().sum()}")

    df = compute_risk_score(df, config)
    if debug:
        print(f"[MODEL] final_risk valid={df['final_risk'].notna().sum()}")

    return df, iso
