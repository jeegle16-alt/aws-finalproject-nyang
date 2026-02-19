"""
src/steps/baseline.py
─────────────────────
notebooks/v3_MODEL.ipynb 섹션 1.4, 3.1-3.7 로직 모듈화.

I/O 없음 – 순수 pandas/numpy 계산만.

공개 API:
  add_time_context(df)          -> pd.DataFrame
  compute_baseline(df, config)  -> pd.DataFrame
  winsorize_series(s, lo, hi)   -> pd.Series
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# 상수 (notebook 3.1 그대로)
# ──────────────────────────────────────────────────────────────────────────────
ST_WINDOW         = 14
LT_WINDOW         = 56
ST_MINP_SPLIT     = 5
ST_MINP_GLOBAL    = 7
LT_MINP           = 14
EARLY_WINDOW      = 7
EARLY_MINP_GLOBAL = 3
EARLY_MINP_SPLIT  = 3
DEFAULT_FLOOR     = 0.1

STD_FLOOR_MAP: Dict[str, float] = {
    "daily_event_cnt":   10.0,
    "night_ratio":        0.05,
    "hour_entropy":       0.05,
    "Notif":              5.0,
    "UserAct":            5.0,
    "Screen":             3.0,
    "gap_max":            0.5,
    "gap_cnt_2h":         1.0,
    "gap_long_ratio":     0.05,
    "overnight_gap":      0.5,
    "gap_p95":            0.5,
    "gap_cnt_6h":         0.5,
    "session_total_sec":  60.0,
    "session_cnt":        1.0,
    "step_sum":           50.0,
}

# notebook 3.1 그대로 – 실제 사용 시 df에 있는 것만 필터링
BASE_COLS_CANDIDATES: List[str] = [
    "Screen", "Notif", "UserAct",
    "daily_event_cnt", "night_ratio", "hour_entropy",
    "gap_max", "gap_cnt_2h", "gap_long_ratio", "overnight_gap",
    "gap_p95", "gap_cnt_6h", "session_total_sec", "session_cnt", "step_sum",
]


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼 (노트북 공유)
# ──────────────────────────────────────────────────────────────────────────────

def winsorize_series(
    s: pd.Series,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> pd.Series:
    """유효한 값 기준으로 quantile clip (NaN 보존) – notebook 3.3 그대로."""
    valid = s.dropna()
    if len(valid) < 10:
        return s
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────

def add_time_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    notebook 1.4: is_weekend, day_idx, cold_stage 컬럼 추가.

    baseline / context 계산 전에 반드시 먼저 호출.
    """
    df = df.copy()
    df["date"]        = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["is_weekend"]  = df["date"].dt.weekday >= 5
    first_day         = df.groupby("uuid")["date"].transform("min")
    df["day_idx"]     = (df["date"] - first_day).dt.days + 1
    df["cold_stage"]  = np.select(
        [
            df["day_idx"].between(1, 4),
            df["day_idx"].between(5, 8),
            df["day_idx"].between(9, 15),
            df["day_idx"] >= 16,
        ],
        ["ONBOARD", "WARMUP", "SEMI_READY", "READY"],
        default="ONBOARD",
    )
    return df


def compute_baseline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    notebook 3.1-3.7: LT / ST / Early baseline 계산 후 컬럼 추가.

    선행 조건:
      add_time_context()        완료  (is_weekend, cold_stage)
      add_gate_and_context()    완료  (baseline_fit_mask)

    추가 컬럼:
      {c}_mean_lt_g / std_lt_g
      {c}_mean_st_g / std_st_g
      {c}_mean_st_s / std_st_s   (split + fallback chain)
      {c}_mean_early_s / std_early_s
      {c}_mean_final / std_final
      baseline_ready, early_ready, analysis_ready
    """
    df = df.copy().sort_values(["uuid", "date"]).reset_index(drop=True)

    bc             = config.get("baseline", {})
    st_window      = int(bc.get("st_window",       ST_WINDOW))
    lt_window      = int(bc.get("lt_window",       LT_WINDOW))
    st_minp_split  = int(bc.get("st_minp_split",   ST_MINP_SPLIT))
    st_minp_global = int(bc.get("st_minp_global",  ST_MINP_GLOBAL))
    lt_minp        = int(bc.get("lt_minp",         LT_MINP))
    early_window   = int(bc.get("early_window",    EARLY_WINDOW))
    early_minp     = int(bc.get("early_minp",      EARLY_MINP_GLOBAL))
    default_floor  = float(bc.get("default_floor", DEFAULT_FLOOR))
    w_lower        = float(bc.get("winsorize_lower", 0.005))
    w_upper        = float(bc.get("winsorize_upper", 0.995))

    base_cols: List[str] = [c for c in BASE_COLS_CANDIDATES if c in df.columns]
    if not base_cols:
        raise ValueError("[baseline] No BASE_COLS found in df. Check FE output.")

    floor_map: Dict[str, float] = dict(STD_FLOOR_MAP)
    for c in base_cols:
        if c not in floor_map:
            floor_map[c] = default_floor

    for c in base_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "baseline_fit_mask" not in df.columns:
        raise ValueError("[baseline] baseline_fit_mask missing. Call add_gate_and_context first.")
    fit_mask = pd.Series(df["baseline_fit_mask"]).astype("boolean").fillna(False)

    # ── 3.3  LT Global baseline ─────────────────────────────────────────────
    for c in base_cols:
        x       = pd.to_numeric(df[c], errors="coerce")
        x_fit   = winsorize_series(x.where(fit_mask, np.nan), w_lower, w_upper)
        x_shift = x_fit.groupby(df["uuid"], sort=False).shift(1)
        roll    = x_shift.groupby(df["uuid"], sort=False).rolling(lt_window, min_periods=lt_minp)
        mean_lt = roll.mean().reset_index(level=0, drop=True).sort_index()
        std_lt  = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        floor   = float(floor_map.get(c, default_floor))
        std_lt  = std_lt.replace(0, np.nan).fillna(floor).clip(lower=floor)
        df[f"{c}_mean_lt_g"] = mean_lt
        df[f"{c}_std_lt_g"]  = std_lt

    # ── 3.4.1  ST Global ────────────────────────────────────────────────────
    for c in base_cols:
        x         = pd.to_numeric(df[c], errors="coerce")
        x_fit     = winsorize_series(x.where(fit_mask, np.nan), w_lower, w_upper)
        x_shift   = x_fit.groupby(df["uuid"], sort=False).shift(1)
        roll      = x_shift.groupby(df["uuid"], sort=False).rolling(st_window, min_periods=st_minp_global)
        mean_st_g = roll.mean().reset_index(level=0, drop=True).sort_index()
        std_st_g  = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        floor     = float(floor_map.get(c, default_floor))
        std_st_g  = std_st_g.replace(0, np.nan).fillna(floor).clip(lower=floor)
        df[f"{c}_mean_st_g"] = mean_st_g
        df[f"{c}_std_st_g"]  = std_st_g

    # ── 3.4.2  ST Split + fallback chain ────────────────────────────────────
    for c in base_cols:
        x     = pd.to_numeric(df[c], errors="coerce")
        x_fit = winsorize_series(x.where(fit_mask, np.nan), w_lower, w_upper)
        grp   = [df["uuid"], df["is_weekend"]]
        xs    = x_fit.groupby(grp, sort=False).shift(1)
        roll  = xs.groupby(grp, sort=False).rolling(st_window, min_periods=st_minp_split)
        mu_s_raw = roll.mean().reset_index(level=[0, 1], drop=True).sort_index()
        sd_s_raw = roll.std(ddof=0).reset_index(level=[0, 1], drop=True).sort_index()
        floor    = float(floor_map.get(c, default_floor))
        sd_s_raw = sd_s_raw.replace(0, np.nan).fillna(floor).clip(lower=floor)
        # fallback: split → ST global → LT global
        mean_final = mu_s_raw.fillna(df[f"{c}_mean_st_g"]).fillna(df[f"{c}_mean_lt_g"])
        std_final  = sd_s_raw.fillna(df[f"{c}_std_st_g"]).fillna(df[f"{c}_std_lt_g"])
        df[f"{c}_mean_st_s"]        = mean_final
        df[f"{c}_std_st_s"]         = std_final
        df[f"{c}_fallback_stsplit"] = mu_s_raw.isna()

    # ── 3.5  baseline_ready ──────────────────────────────────────────────────
    mean_cols_st = [f"{c}_mean_st_s" for c in base_cols]
    std_cols_st  = [f"{c}_std_st_s"  for c in base_cols]
    df["baseline_ready"] = df[mean_cols_st + std_cols_st].notna().all(axis=1)

    # ── 3.6  Early baseline ──────────────────────────────────────────────────
    for c in base_cols:
        x     = winsorize_series(df[c].where(fit_mask, np.nan), w_lower, w_upper)
        xs    = x.groupby(df["uuid"], sort=False).shift(1)
        roll  = xs.groupby(df["uuid"], sort=False).rolling(early_window, min_periods=early_minp)
        mu_g  = roll.mean().reset_index(level=0, drop=True).sort_index()
        sd_g  = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        floor = float(floor_map.get(c, default_floor))
        sd_g  = sd_g.replace(0, np.nan).fillna(floor).clip(lower=floor)
        df[f"{c}_mean_early_g"] = mu_g
        df[f"{c}_std_early_g"]  = sd_g

        grp  = [df["uuid"], df["is_weekend"]]
        xs2  = x.groupby(grp, sort=False).shift(1)
        roll2 = xs2.groupby(grp, sort=False).rolling(early_window, min_periods=early_minp)
        mu_s  = roll2.mean().reset_index(level=[0, 1], drop=True).sort_index()
        sd_s  = roll2.std(ddof=0).reset_index(level=[0, 1], drop=True).sort_index()
        sd_s  = sd_s.replace(0, np.nan).fillna(floor).clip(lower=floor)
        df[f"{c}_mean_early_s"] = mu_s.fillna(df[f"{c}_mean_early_g"])
        df[f"{c}_std_early_s"]  = sd_s.fillna(df[f"{c}_std_early_g"])

    early_mean_cols = [f"{c}_mean_early_s" for c in base_cols]
    early_std_cols  = [f"{c}_std_early_s"  for c in base_cols]
    df["early_ready"] = df[early_mean_cols + early_std_cols].notna().all(axis=1)

    # ── 3.7  최종 baseline 통합 ──────────────────────────────────────────────
    for c in base_cols:
        df[f"{c}_mean_final"] = df[f"{c}_mean_st_s"].fillna(df[f"{c}_mean_early_s"])
        df[f"{c}_std_final"]  = df[f"{c}_std_st_s"].fillna(df[f"{c}_std_early_s"])

    df["analysis_ready"] = (
        df[[f"{c}_mean_final" for c in base_cols] + [f"{c}_std_final" for c in base_cols]]
        .notna().all(axis=1)
    )

    df.attrs["BASE_COLS"] = base_cols
    return df
