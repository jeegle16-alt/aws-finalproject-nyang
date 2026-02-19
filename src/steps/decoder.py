"""
src/steps/decoder.py
─────────────────────
notebooks/v3_DECODER.ipynb 전체 로직 모듈화.

I/O 없음 – 순수 pandas/numpy 계산만.

공개 API:
  decode(df, config, *, debug=False) -> pd.DataFrame
    model_out DataFrame → cat_state / notify_level / decoder_quality
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 상수 (notebook 0.3 / cell 6 그대로)
# ──────────────────────────────────────────────────────────────────────────────

RISK_LOW    = 0.45
RISK_ALERT  = 0.60
RISK_SEVERE = 0.70

STATE_PRIORITY: Dict[str, int] = {
    "NO_DATA":  100,
    "TRAVEL":    90,
    "CHAOS":     70,
    "LETHARGY":  60,
    "SLEEP":     50,
    "STABLE":    10,
}

SLEEP_FEATURES    = ["night_ratio_z",      "overnight_gap_z"]
LETHARGY_FEATURES = ["daily_event_cnt_z",  "UserAct_z",     "Screen_z"]
CHAOS_FEATURES    = ["hour_entropy_z",     "gap_max_z",     "gap_cnt_6h_z"]

Z_THRESHOLD       = 2.0
GROUP_Z_THRESHOLD = 1.5

GROUP_REP_MAP: Dict[str, str] = {
    "rhythm":  "z_rhythm_rep",
    "core":    "z_core_rep",
    "gap":     "z_gap_rep",
    "session": "z_session_rep",
}

STAGE_NOTIFY_MAP: Dict[str, str] = {
    "ONBOARD":    "NONE",
    "WARMUP":     "NONE",
    "SEMI_READY": "LOW",
    "READY":      "NORMAL",
}

OUTPUT_COLS = [
    "uuid", "date", "cat_state", "notify_level", "decoder_quality",
    "risk_used", "risk_score", "final_risk", "risk_band",
    "top_z_feature", "top_z_value",
]


# ──────────────────────────────────────────────────────────────────────────────
# 내부 함수 (notebook 3.1 / 5.1 / 5.2 / 5.3 / 6.1 / 6.2 / 6.3 / 7.1 / 8.1)
# ──────────────────────────────────────────────────────────────────────────────

def _get_risk_band(risk: float, risk_low: float, risk_alert: float, risk_severe: float) -> str:
    """notebook 3.1 get_risk_band."""
    if pd.isna(risk):
        return "UNKNOWN"
    if risk < risk_low:
        return "SAFE"
    elif risk < risk_alert:
        return "WATCH"
    elif risk < risk_severe:
        return "ALERT"
    else:
        return "SEVERE"


def _get_top_z_feature(
    row: pd.Series,
    z_cols: List[str],
) -> Tuple[str, float]:
    """notebook 5.1 get_top_z_feature."""
    zvals = row[z_cols].dropna()
    if len(zvals) == 0:
        return "none", 0.0
    top_col = zvals.abs().idxmax()
    return top_col, float(zvals[top_col])


def _decode_pattern_state(
    row: pd.Series,
    group_z_threshold: float,
    z_threshold: float,
) -> str:
    """notebook 5.2 decode_pattern_state – 그룹별 대표 Z + Top-Z 기반."""
    top_feature = row["top_z_feature"]
    top_value   = row["top_z_value"]

    rhythm_rep  = float(row.get("z_rhythm_rep",  0) or 0)
    core_rep    = float(row.get("z_core_rep",    0) or 0)
    gap_rep     = float(row.get("z_gap_rep",     0) or 0)

    candidates: Dict[str, float] = {}

    if rhythm_rep >= group_z_threshold:
        candidates["SLEEP"] = rhythm_rep

    if core_rep >= group_z_threshold:
        if top_feature in LETHARGY_FEATURES and top_value < 0:
            candidates["LETHARGY"] = core_rep
        elif top_feature not in LETHARGY_FEATURES:
            pass
        else:
            candidates["LETHARGY"] = core_rep

    if gap_rep >= group_z_threshold:
        candidates["CHAOS"] = gap_rep

    if candidates:
        return max(candidates, key=candidates.get)

    if abs(top_value) < z_threshold:
        return "STABLE"
    if top_feature in SLEEP_FEATURES:
        return "SLEEP"
    if top_feature in LETHARGY_FEATURES:
        return "LETHARGY" if top_value < 0 else "STABLE"
    if top_feature in CHAOS_FEATURES:
        return "CHAOS"
    return "STABLE"


def _determine_cat_state(row: pd.Series) -> str:
    """notebook 5.3 determine_cat_state."""
    if row["is_no_data"]:
        return "NO_DATA"
    if row["is_travel"]:
        return "TRAVEL"
    return row["pattern_state"]


def _get_base_notify(risk_band: str) -> str:
    """notebook 6.1 get_base_notify."""
    if risk_band in ("SAFE", "WATCH"):
        return "NONE"
    elif risk_band == "ALERT":
        return "LOW"
    elif risk_band == "SEVERE":
        return "HIGH"
    return "NONE"


def _apply_context_downshift(row: pd.Series) -> str:
    """notebook 6.2 apply_context_downshift."""
    base = row["base_notify"]
    if row["is_travel"]:
        return "NONE"
    if bool(row.get("partial_flag", False)) or bool(row.get("tz_flag", False)):
        return "NONE"
    return base


def _determine_notify_level(row: pd.Series) -> str:
    """notebook 6.3 determine_notify_level."""
    if row["cat_state"] in ("NO_DATA", "TRAVEL"):
        return "NONE"
    if row["cat_state"] == "STABLE":
        if bool(row.get("stable_hold_flag", False)):
            return "LOW"
        if bool(row.get("drift_flag", False)):
            return "LOW"
        return "NONE"
    return row["notify_after_context"]


def _apply_cold_start_policy(row: pd.Series) -> str:
    """notebook 7.1 apply_cold_start_policy."""
    stage  = row["cold_stage"]
    notify = row["notify_level"]
    if pd.isna(stage):
        return notify
    stage = str(stage).upper()
    if stage in ("ONBOARD", "WARMUP"):
        return "NONE"
    elif stage == "SEMI_READY":
        return "LOW" if notify == "HIGH" else notify
    return notify  # READY


def _get_decoder_quality(row: pd.Series) -> str:
    """notebook 8.1 get_decoder_quality."""
    if row["cat_state"] == "NO_DATA":
        return "LOW_CONF" if row["quality_state"] == "LOW_CONF" else "NO_DATA"
    return "OK"


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────

def decode(
    df: pd.DataFrame,
    config: dict,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    model_out DataFrame → cat_state / notify_level / decoder_quality.

    선행 조건: run_model() 완료
      (risk_score, final_risk, quality_state, cold_stage, travel_flag, z_*_rep, …)

    Returns
    -------
    pd.DataFrame  (OUTPUT_COLS 컬럼 포함)
    """
    df = df.copy()
    dc = config.get("decoder", {})

    risk_low       = float(dc.get("risk_low",        RISK_LOW))
    risk_alert     = float(dc.get("risk_alert",      RISK_ALERT))
    risk_severe    = float(dc.get("risk_severe",     RISK_SEVERE))
    z_threshold    = float(dc.get("z_threshold",     Z_THRESHOLD))
    group_z_thr    = float(dc.get("group_z_threshold", GROUP_Z_THRESHOLD))

    # ── 1.3  타입 표준화 ────────────────────────────────────────────────────
    BOOL_COLS = [
        "travel_flag", "partial_flag", "tz_flag",
        "baseline_ready", "early_ready", "has_activity",
        "stable_hold_flag", "drift_flag",
    ]
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("boolean").fillna(False)

    for c in ("risk_score", "final_risk", "early_risk"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("quality_state", "context_mode", "cold_stage"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    # optional defaults
    OPTIONAL_DEFAULTS = {
        "final_risk":         np.nan,
        "early_risk":         np.nan,
        "baseline_ready":     False,
        "early_ready":        False,
        "has_activity":       False,
        "partial_flag":       False,
        "tz_flag":            False,
        "stable_hold_flag":   False,
        "drift_flag":         False,
        "drift_top_feature":  "",
    }
    for c, v in OPTIONAL_DEFAULTS.items():
        if c not in df.columns:
            df[c] = v

    # risk_used (final_risk 우선)
    df["risk_used"] = df["final_risk"]
    df.loc[df["risk_used"].isna(), "risk_used"] = df.loc[df["risk_used"].isna(), "risk_score"]

    # ── 2.1  NO_DATA gate ───────────────────────────────────────────────────
    df["quality_fail"] = (df["quality_state"] == "LOW_CONF")
    df["analysis_ready"] = (
        (~df["quality_fail"])
        & df["risk_used"].notna()
        & (df["baseline_ready"] | df["early_ready"])
    )
    df["is_no_data"] = ~df["analysis_ready"]
    df["analysis_ok"] = df["analysis_ready"]

    if debug:
        print(f"[DECODER] NO_DATA={df['is_no_data'].sum()}, analysis_ok={df['analysis_ok'].sum()}")

    # ── 3.1  risk_band ──────────────────────────────────────────────────────
    df["risk_band"] = df["risk_used"].apply(
        lambda r: _get_risk_band(r, risk_low, risk_alert, risk_severe)
    )

    # ── 4.1  TRAVEL override ────────────────────────────────────────────────
    df["is_travel"] = df["travel_flag"].fillna(False)

    # ── 5.1  Top-Z feature ──────────────────────────────────────────────────
    z_cols = [c for c in df.columns if c.endswith("_z") and not c.endswith("_z_early")]
    df["top_z_feature"] = "none"
    df["top_z_value"]   = 0.0

    mask = df["analysis_ok"]
    if mask.any() and z_cols:
        topz = df.loc[mask].apply(
            lambda r: _get_top_z_feature(r, z_cols), axis=1
        )
        df.loc[mask, "top_z_feature"] = topz.apply(lambda x: x[0]).astype("string")
        df.loc[mask, "top_z_value"]   = topz.apply(lambda x: x[1]).astype(float)

    # ── 5.2  pattern_state ──────────────────────────────────────────────────
    df["pattern_state"] = "UNKNOWN"
    if mask.any():
        df.loc[mask, "pattern_state"] = df.loc[mask].apply(
            lambda r: _decode_pattern_state(r, group_z_thr, z_threshold), axis=1
        )

    # ── 5.3  cat_state ──────────────────────────────────────────────────────
    df["cat_state"] = df.apply(_determine_cat_state, axis=1)

    # ── 6.1  base_notify ────────────────────────────────────────────────────
    df["base_notify"] = df["risk_band"].apply(_get_base_notify)

    # ── 6.2  context downshift ───────────────────────────────────────────────
    df["notify_after_context"] = df.apply(_apply_context_downshift, axis=1)

    # ── 6.3  notify_level ────────────────────────────────────────────────────
    df["notify_level"] = df.apply(_determine_notify_level, axis=1)

    # ── 7.1  cold-start policy ───────────────────────────────────────────────
    df["notify_final"] = df.apply(_apply_cold_start_policy, axis=1)
    df["notify_level"] = df["notify_final"]   # 최종

    # ── 8.1  decoder_quality ─────────────────────────────────────────────────
    df["decoder_quality"] = df.apply(_get_decoder_quality, axis=1)

    if debug:
        print(f"[DECODER] cat_state:\n{df['cat_state'].value_counts()}")
        print(f"[DECODER] notify_level:\n{df['notify_level'].value_counts()}")

    # ── 8.2  출력 컬럼 선택 ────────────────────────────────────────────────
    # OUTPUT_COLS에 없는 컬럼도 df에 유지 (batch_runner가 더 넓게 저장할 수 있도록)
    return df
