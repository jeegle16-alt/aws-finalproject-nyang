"""
src/steps/decoder.py
====================
v3_DECODER.ipynb 로직을 그대로 포팅.

[변경점 - 로직 외]
- 입력: model_out_v3 DataFrame
- 출력: state_out_v3 DataFrame
- 로직: 100% 동일
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 0. 상수  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

RISK_LOW    = 0.45
RISK_ALERT  = 0.60
RISK_SEVERE = 0.70

STATE_PRIORITY = {
    "NO_DATA":  100,
    "TRAVEL":    90,
    "CHAOS":     70,
    "LETHARGY":  60,
    "SLEEP":     50,
    "STABLE":    10,
}

SLEEP_FEATURES = ["night_ratio_z", "overnight_gap_z"]
LETHARGY_FEATURES = ["daily_event_cnt_z", "UserAct_z", "Screen_z"]
CHAOS_FEATURES = ["hour_entropy_z", "gap_max_z", "gap_cnt_6h_z"]

Z_THRESHOLD = 2.0
GROUP_Z_THRESHOLD = 1.5

GROUP_REP_MAP = {
    "rhythm":  "z_rhythm_rep",
    "core":    "z_core_rep",
    "gap":     "z_gap_rep",
    "session": "z_session_rep",
}

STAGE_NOTIFY_MAP = {
    "ONBOARD":    "NONE",
    "WARMUP":     "NONE",
    "SEMI_READY": "LOW",
    "READY":      "NORMAL",
}

OPTIONAL_DEFAULTS = {
    "final_risk":        np.nan,
    "early_risk":        np.nan,
    "baseline_ready":    False,
    "early_ready":       False,
    "has_activity":      False,
    "partial_flag":      False,
    "tz_flag":           False,
    "stable_hold_flag":  False,
    "drift_flag":        False,
    "drift_top_feature": "",
}


# ---------------------------------------------------------------------------
# 1. Utility  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

def get_risk_band(risk: float) -> str:
    if pd.isna(risk):
        return "UNKNOWN"
    if risk < RISK_LOW:
        return "SAFE"
    elif risk < RISK_ALERT:
        return "WATCH"
    elif risk < RISK_SEVERE:
        return "ALERT"
    else:
        return "SEVERE"


def get_base_notify(risk_band: str) -> str:
    mapping = {"SAFE": "NONE", "WATCH": "NONE", "ALERT": "LOW", "SEVERE": "HIGH"}
    return mapping.get(risk_band, "NONE")


# ---------------------------------------------------------------------------
# 2. 전처리  (v3_DECODER Section 1-2 그대로)
# ---------------------------------------------------------------------------

def _preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # optional 컬럼 없으면 default 생성
    for c, v in OPTIONAL_DEFAULTS.items():
        if c not in df.columns:
            df[c] = v

    # 타입 정규화
    bool_cols = ["travel_flag", "partial_flag", "tz_flag", "baseline_ready",
                 "early_ready", "has_activity", "stable_hold_flag", "drift_flag"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype("boolean").fillna(False)

    for c in ["risk_score", "final_risk", "early_risk"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["quality_state", "context_mode", "cold_stage"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # risk_used: final_risk 우선, 없으면 risk_score
    df["risk_used"] = df["final_risk"]
    df.loc[df["risk_used"].isna(), "risk_used"] = df.loc[df["risk_used"].isna(), "risk_score"]

    # Z 컬럼 목록
    z_cols = [c for c in df.columns if c.endswith("_z") and not c.endswith("_z_early")]

    return df, z_cols


# ---------------------------------------------------------------------------
# 3. Gate  (v3_DECODER Section 3 그대로)
# ---------------------------------------------------------------------------

def _compute_gate(df: pd.DataFrame) -> pd.DataFrame:
    df["quality_fail"] = (df["quality_state"] == "LOW_CONF")
    df["analysis_ready"] = (
        (~df["quality_fail"])
        & (df["risk_used"].notna())
        & (df["baseline_ready"] | df["early_ready"])
    )
    df["is_no_data"] = ~df["analysis_ready"]
    df["analysis_ok"] = df["analysis_ready"]
    return df


# ---------------------------------------------------------------------------
# 4. Risk band + TRAVEL  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

def _compute_risk_band_travel(df: pd.DataFrame) -> pd.DataFrame:
    df["risk_band"] = df["risk_used"].apply(get_risk_band)
    df["is_travel"] = df["travel_flag"].fillna(False)
    return df


# ---------------------------------------------------------------------------
# 5. Top-Z feature  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

def _get_top_z_feature(row: pd.Series, z_cols: List[str]) -> Tuple[str, float]:
    zvals = row[z_cols]
    zvals = zvals[pd.notna(zvals)]
    if len(zvals) == 0:
        return ("none", 0.0)
    top_col = zvals.abs().idxmax()
    return (top_col, float(zvals[top_col]))


def _compute_top_z(df: pd.DataFrame, z_cols: List[str]) -> pd.DataFrame:
    df["top_z_feature"] = "none"
    df["top_z_value"]   = 0.0

    mask = df["analysis_ok"]
    if mask.any() and z_cols:
        results = df.loc[mask].apply(lambda r: _get_top_z_feature(r, z_cols), axis=1)
        df.loc[mask, "top_z_feature"] = results.apply(lambda x: x[0]).astype("string")
        df.loc[mask, "top_z_value"]   = results.apply(lambda x: x[1]).astype(float)

    return df


# ---------------------------------------------------------------------------
# 6. Pattern state  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

def decode_pattern_state(row: pd.Series) -> str:
    """
    [v3] 그룹별 대표 Z + Top-Z feature 기반 상태 판정
    1차: 그룹 대표값(z_rhythm_rep, z_core_rep, z_gap_rep)
    2차: Top-Z feature fallback
    """
    top_feature = row["top_z_feature"]
    top_value = row["top_z_value"]

    rhythm_rep = float(row.get("z_rhythm_rep",  0) or 0)
    core_rep = float(row.get("z_core_rep",    0) or 0)
    gap_rep = float(row.get("z_gap_rep",     0) or 0)

    candidates: Dict[str, float] = {}
    if rhythm_rep >= GROUP_Z_THRESHOLD:
        candidates["SLEEP"] = rhythm_rep
    if core_rep >= GROUP_Z_THRESHOLD:
        if top_feature in LETHARGY_FEATURES and top_value < 0:
            candidates["LETHARGY"] = core_rep
        elif top_feature not in LETHARGY_FEATURES:
            pass
        else:
            candidates["LETHARGY"] = core_rep
    if gap_rep >= GROUP_Z_THRESHOLD:
        candidates["CHAOS"] = gap_rep

    if candidates:
        return max(candidates, key=candidates.get)

    # fallback: Top-Z
    if abs(top_value) < Z_THRESHOLD:
        return "STABLE"
    if top_feature in SLEEP_FEATURES:
        return "SLEEP"
    if top_feature in LETHARGY_FEATURES:
        return "LETHARGY" if top_value < 0 else "STABLE"
    if top_feature in CHAOS_FEATURES:
        return "CHAOS"
    return "STABLE"


def _compute_pattern_state(df: pd.DataFrame) -> pd.DataFrame:
    df["pattern_state"] = "UNKNOWN"
    mask = df["analysis_ok"]
    if mask.any():
        df.loc[mask, "pattern_state"] = df.loc[mask].apply(decode_pattern_state, axis=1)
    return df


# ---------------------------------------------------------------------------
# 7. Cat state (최종)  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

def determine_cat_state(row: pd.Series) -> str:
    """우선순위: NO_DATA > TRAVEL > Pattern"""
    if row["is_no_data"]:
        return "NO_DATA"
    if row["is_travel"]:
        return "TRAVEL"
    return row["pattern_state"]


def _compute_cat_state(df: pd.DataFrame) -> pd.DataFrame:
    df["cat_state"] = df.apply(determine_cat_state, axis=1)
    return df


# ---------------------------------------------------------------------------
# 8. Notify  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

def apply_context_downshift(row: pd.Series) -> str:
    base = row["base_notify"]
    if row["is_travel"]:
        return "NONE"
    if bool(row.get("partial_flag", False)) or bool(row.get("tz_flag", False)):
        return "NONE"
    return base


def determine_notify_level(row: pd.Series) -> str:
    if row["cat_state"] in ["NO_DATA", "TRAVEL"]:
        return "NONE"
    if row["cat_state"] == "STABLE":
        if bool(row.get("stable_hold_flag", False)):
            return "LOW"
        if bool(row.get("drift_flag", False)):
            return "LOW"
        return "NONE"
    return row["notify_after_context"]


def apply_cold_start_policy(row: pd.Series) -> str:
    stage  = row["cold_stage"]
    notify = row["notify_level"]
    if pd.isna(stage):
        return notify
    stage = str(stage).upper()
    if stage in ["ONBOARD", "WARMUP"]:
        return "NONE"
    elif stage == "SEMI_READY":
        return "LOW" if notify == "HIGH" else notify
    else:
        return notify


def get_decoder_quality(row: pd.Series) -> str:
    if row["cat_state"] == "NO_DATA":
        return "LOW_CONF" if row["quality_state"] == "LOW_CONF" else "NO_DATA"
    return "OK"


def _compute_notify(df: pd.DataFrame) -> pd.DataFrame:
    df["base_notify"] = df["risk_band"].apply(get_base_notify)
    df["notify_after_context"] = df.apply(apply_context_downshift, axis=1)
    df["notify_level"] = df.apply(determine_notify_level, axis=1)
    df["notify_final"] = df.apply(apply_cold_start_policy, axis=1)
    df["decoder_quality"] = df.apply(get_decoder_quality, axis=1)
    return df


# ---------------------------------------------------------------------------
# 9. Output  (v3_DECODER 그대로)
# ---------------------------------------------------------------------------

OUTPUT_COLS = [
    "uuid", "date", "cat_state", "notify_final", "decoder_quality", "risk_used", "risk_score", "final_risk", "risk_band", "top_z_feature", "top_z_value",
]


def _build_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    # notify_final을 notify_level로 노출 (v3_DECODER 출력 스키마)
    output["notify_level"] = output["notify_final"]
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")

    cols = [c for c in OUTPUT_COLS if c in output.columns]
    # notify_level 없으면 추가
    if "notify_level" in output.columns and "notify_level" not in cols:
        cols.insert(cols.index("notify_final") + 1, "notify_level")

    return output[cols].copy()


# ---------------------------------------------------------------------------
# 10. 메인 진입점
# ---------------------------------------------------------------------------

def run(model_out_df: pd.DataFrame) -> pd.DataFrame:
    """
    DECODER Step 메인 함수.

    Parameters
    ----------
    model_out_df : pd.DataFrame
        model.run() 출력 (model_out_v3)

    Returns
    -------
    pd.DataFrame
        state_out_v3 — uuid, date, cat_state, notify_level, decoder_quality 포함
    """
    print(f"[DECODER] input: {model_out_df.shape}")

    df, z_cols = _preprocess(model_out_df)
    df = _compute_gate(df)
    df = _compute_risk_band_travel(df)
    df = _compute_top_z(df, z_cols)
    df = _compute_pattern_state(df)
    df = _compute_cat_state(df)
    df = _compute_notify(df)

    state_out = _build_output(df)

    print(f"[DECODER] output: {state_out.shape}")
    print("[DECODER] cat_state dist:")
    print(state_out["cat_state"].value_counts().to_dict())

    return state_out
