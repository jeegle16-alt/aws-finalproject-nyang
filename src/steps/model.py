"""
src/steps/model.py
==================
v3_MODEL.ipynb 로직을 그대로 포팅.

[변경점 - 로직 외]
- 입력:  daily_feature_v3 DataFrame  +  iso (학습된 IsolationForest 모델)
- 출력:  model_out_v3 DataFrame
- 로직:  100% 동일 (베이스라인 계산, Z-score, Anomaly, Risk, Drift 모두 유지)

[실행 시나리오]
  train  → run(feature_df, mode="train") : IF 학습 → 모델 반환
  infer  → run(feature_df, mode="infer", iso=loaded_model) : 추론 전용
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# 0. 상수  (v3_MODEL 그대로)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ST_WINDOW      = 14
LT_WINDOW      = 56
ST_MINP_SPLIT  = 5
ST_MINP_GLOBAL = 7
LT_MINP        = 14
DEFAULT_FLOOR  = 0.1

STD_FLOOR_MAP = {
    "daily_event_cnt": 10.0,
    "night_ratio":     0.05,
    "hour_entropy":    0.05,
    "Notif":           5.0,
    "UserAct":         5.0,
    "Screen":          3.0,
    "gap_max":         0.5,
    "gap_cnt_2h":      1.0,
    "gap_long_ratio":  0.05,
    "overnight_gap":   0.5,
    "gap_p95":         0.5,
    "gap_cnt_6h":      0.5,
    "session_total_sec": 60.0,
    "session_cnt":     1.0,
    "step_sum":        50.0,
}

BASE_COLS_CANDIDATES = [
    "Screen", "Notif", "UserAct",
    "daily_event_cnt", "night_ratio", "hour_entropy",
    "gap_max", "gap_cnt_2h", "gap_long_ratio", "overnight_gap",
    "gap_p95", "gap_cnt_6h", "session_total_sec", "session_cnt", "step_sum",
]

Z_BASE_CANDIDATES = [
    "daily_event_cnt",
    "Screen", "Notif", "UserAct",
    "night_ratio", "hour_entropy",
    "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h", "gap_long_ratio", "overnight_gap",
    "session_total_sec", "session_cnt",
    "step_sum",
]

# IsolationForest 하이퍼파라미터
IF_CONTAM    = 0.05
IF_ESTIMATORS = 300
IF_MAX_SAMPLES = "auto"
IF_SEED      = 42

# Risk 가중치
W_Z = 0.85
W_A = 0.15

DISCOUNT = {"NORMAL": 1.00, "PARTIAL": 0.60, "TRAVEL": 0.70}
EMA_ALPHA = 0.25

# Z-score clip
Z_CLIP       = 4.0
Z_CLIP_EARLY = 4.0

# Early baseline
EARLY_WINDOW      = 7
EARLY_MINP_GLOBAL = 3
EARLY_MINP_SPLIT  = 3

# Recovery / Drift
HOLD_WINDOW = 14
HOLD_MINP   = 7
BAD_THR     = 0.70
STABLE_THR  = 0.45

ANCHOR_START = 16
ANCHOR_END   = 30
DRIFT_Z_THR  = 2.0
DRIFT_WATCH_COLS_CANDIDATES = ["daily_event_cnt", "night_ratio", "gap_max"]

# Anomaly scaling
Q_LOW  = 0.80
Q_HIGH = 0.99

PARTIAL_THR = 1.0
TRAVEL_THR  = 2.0
TZ_THR      = 1.0

# Z 그룹 정의
Z_GROUP_DEFS = {
    "core":    ["Screen_z", "Notif_z", "UserAct_z", "daily_event_cnt_z"],
    "rhythm":  ["night_ratio_z", "hour_entropy_z"],
    "gap":     ["gap_max_z", "gap_p95_z", "gap_cnt_2h_z", "gap_cnt_6h_z", "gap_long_ratio_z", "overnight_gap_z"],
    "session": ["session_total_sec_z", "session_cnt_z"],
    "mob":     ["step_sum_z"],
}
CORE_Q  = 0.85
OTHER_Q = 0.90


# ---------------------------------------------------------------------------
# 1. Utility  (v3_MODEL 그대로)
# ---------------------------------------------------------------------------

def winsorize_series(s: pd.Series, lower_q: float = 0.005, upper_q: float = 0.995) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 10:
        return s
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def _trim_mean(row: pd.Series, keep_ratio: float = 0.8) -> float:
    s = row.dropna().sort_values()
    if len(s) == 0:
        return np.nan
    k = max(1, int(len(s) * keep_ratio))
    return float(s.iloc[:k].mean())


def _resolve_discount(cm: str) -> float:
    if pd.isna(cm) or cm == "NORMAL":
        return 1.0
    parts = str(cm).split("|")
    vals  = [DISCOUNT.get(p.strip(), 1.0) for p in parts]
    return min(vals)


# ---------------------------------------------------------------------------
# 2. 공통 전처리  (v3_MODEL Section 1-2 그대로)
# ---------------------------------------------------------------------------

def _preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """타입 정규화 + cold_stage + BASE_COLS 확정."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["uuid", "date"]).sort_values(["uuid", "date"]).reset_index(drop=True)

    df["is_weekend"] = df["date"].dt.weekday >= 5

    first_day    = df.groupby("uuid")["date"].transform("min")
    df["day_idx"] = (df["date"] - first_day).dt.days + 1
    df["cold_stage"] = np.select(
        [df["day_idx"].between(1, 4), df["day_idx"].between(5, 8),
         df["day_idx"].between(9, 15), df["day_idx"] >= 16],
        ["ONBOARD", "WARMUP", "SEMI_READY", "READY"],
        default="ONBOARD",
    )

    # bool 컬럼
    bool_cols = ["has_activity", "is_weekend",
                 "qc_core_low_cov", "qc_rhythm_low_cov", "qc_gap_low_cov",
                 "qc_meta_low_heartbeat", "qc_meta_retry_warn", "qc_meta_queue_warn"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype("boolean")

    # 수치 컬럼
    num_cols = ["daily_event_cnt", "unlock_cnt", "partial_signal_raw", "travel_signal_raw",
                "tz_change_signal", "night_ratio", "hour_entropy",
                "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h", "gap_long_ratio", "overnight_gap",
                "session_total_sec", "session_cnt", "step_sum", "cell_change_cnt",
                "wifi_change_cnt_est", "heartbeat_cnt", "retry_max", "queue_max"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["daily_event_cnt", "unlock_cnt", "step_sum", "session_cnt", "heartbeat_cnt"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if "gap_max" in df.columns:
        df.loc[df["daily_event_cnt"] == 0, "gap_max"] = df["gap_max"].fillna(24.0)

    # QC
    qc_cols = ["qc_core_low_cov", "qc_rhythm_low_cov", "qc_gap_low_cov"]
    df["qc_missing_any"] = df[qc_cols].isna().any(axis=1)
    for c in qc_cols:
        df[c] = df[c].astype("boolean").fillna(False)
    for c in ["qc_meta_low_heartbeat", "qc_meta_retry_warn", "qc_meta_queue_warn"]:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df), dtype="boolean")
        else:
            df[c] = df[c].astype("boolean")
    df["qc_any_low_cov"] = df["qc_core_low_cov"] | df["qc_rhythm_low_cov"] | df["qc_gap_low_cov"]

    # context flags
    df["partial_lv"] = pd.to_numeric(df.get("partial_signal_raw", 0), errors="coerce").fillna(0.0)
    df["travel_lv"]  = pd.to_numeric(df.get("travel_signal_raw",  0), errors="coerce").fillna(0.0)
    df["tz_lv"]      = pd.to_numeric(df.get("tz_change_signal",   0), errors="coerce").fillna(0.0)
    df["partial_flag"] = (df["partial_lv"] >= PARTIAL_THR).astype("boolean")
    df["tz_flag"]      = (df["tz_lv"]      >= TZ_THR).astype("boolean")
    df["travel_flag"]  = ((df["travel_lv"] >= TRAVEL_THR) | (df["tz_flag"])).astype("boolean")

    # context_mode 문자열
    temp_list = []
    m_partial = df["partial_flag"].fillna(False).to_numpy()
    m_travel  = df["travel_flag"].fillna(False).to_numpy()
    for i in range(len(df)):
        flags = []
        if m_partial[i]: flags.append("PARTIAL")
        if m_travel[i]:  flags.append("TRAVEL")
        temp_list.append("|".join(flags) if flags else "NORMAL")
    df["context_mode"]     = temp_list
    df["context_any_warn"] = (m_partial | m_travel)

    # quality_state
    qs = np.full(len(df), "GOOD", dtype=object)
    mask_lowcov = df["qc_any_low_cov"].to_numpy() | df["qc_missing_any"].to_numpy()
    qs[mask_lowcov] = "LOW_CONF"
    df["quality_state"] = qs

    # baseline_fit_mask
    has_act = df["has_activity"].astype("boolean").fillna(False).to_numpy()
    df["baseline_fit_mask"] = (
        has_act
        & (~df["qc_any_low_cov"].to_numpy())
        & (~df["qc_missing_any"].to_numpy())
        & (~df["partial_flag"].fillna(False).to_numpy())
        & (~df["travel_flag"].fillna(False).to_numpy())
    )

    # BASE_COLS (있는 것만)
    base_cols = [c for c in BASE_COLS_CANDIDATES if c in df.columns]
    if not base_cols:
        raise ValueError("[MODEL] BASE_COLS empty. Check FE output columns.")
    for c in base_cols:
        if c not in STD_FLOOR_MAP:
            STD_FLOOR_MAP[c] = DEFAULT_FLOOR
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, base_cols


# ---------------------------------------------------------------------------
# 3. Baseline 계산  (v3_MODEL Section 3 그대로)
# ---------------------------------------------------------------------------

def _compute_baseline(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    """LT / ST Global / ST Split / Early baseline 계산."""
    BASELINE_FIT_MASK = pd.Series(df["baseline_fit_mask"]).astype("boolean").fillna(False)
    new_cols: dict = {}

    # --- LT global ---
    for c in base_cols:
        x      = pd.to_numeric(df[c], errors="coerce")
        x_fit  = winsorize_series(x.where(BASELINE_FIT_MASK, np.nan))
        x_shift = x_fit.groupby(df["uuid"], sort=False).shift(1)
        roll   = x_shift.groupby(df["uuid"], sort=False).rolling(LT_WINDOW, min_periods=LT_MINP)
        mu_lt  = roll.mean().reset_index(level=0, drop=True).sort_index()
        sd_lt  = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        floor  = float(STD_FLOOR_MAP.get(c, DEFAULT_FLOOR))
        sd_lt  = sd_lt.replace(0, np.nan).fillna(floor).clip(lower=floor)
        new_cols[f"{c}_mean_lt_g"] = mu_lt
        new_cols[f"{c}_std_lt_g"]  = sd_lt

    # --- ST global ---
    for c in base_cols:
        x       = pd.to_numeric(df[c], errors="coerce")
        x_fit   = winsorize_series(x.where(BASELINE_FIT_MASK, np.nan))
        x_shift = x_fit.groupby(df["uuid"], sort=False).shift(1)
        roll    = x_shift.groupby(df["uuid"], sort=False).rolling(ST_WINDOW, min_periods=ST_MINP_GLOBAL)
        mu_stg  = roll.mean().reset_index(level=0, drop=True).sort_index()
        sd_stg  = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        floor   = float(STD_FLOOR_MAP.get(c, DEFAULT_FLOOR))
        sd_stg  = sd_stg.replace(0, np.nan).fillna(floor).clip(lower=floor)
        new_cols[f"{c}_mean_st_g"] = mu_stg
        new_cols[f"{c}_std_st_g"]  = sd_stg

    # --- ST split (weekday/weekend) + fallback chain ---
    fallback_flags = []
    for c in base_cols:
        x       = pd.to_numeric(df[c], errors="coerce")
        x_fit   = winsorize_series(x.where(BASELINE_FIT_MASK, np.nan))
        grp     = [df["uuid"], df["is_weekend"]]
        x_shift = x_fit.groupby(grp, sort=False).shift(1)
        roll    = x_shift.groupby(grp, sort=False).rolling(ST_WINDOW, min_periods=ST_MINP_SPLIT)
        mu_sts  = roll.mean().reset_index(level=[0, 1], drop=True).sort_index()
        sd_sts  = roll.std(ddof=0).reset_index(level=[0, 1], drop=True).sort_index()
        floor   = float(STD_FLOOR_MAP.get(c, DEFAULT_FLOOR))
        sd_sts  = sd_sts.replace(0, np.nan).fillna(floor).clip(lower=floor)

        # fallback: ST split → ST global → LT
        mu_final = mu_sts.fillna(new_cols[f"{c}_mean_st_g"]).fillna(new_cols[f"{c}_mean_lt_g"])
        sd_final = sd_sts.fillna(new_cols[f"{c}_std_st_g"]).fillna(new_cols[f"{c}_std_lt_g"])
        new_cols[f"{c}_mean_st_s"] = mu_final
        new_cols[f"{c}_std_st_s"]  = sd_final

        flag_col = f"{c}_fallback_stsplit"
        new_cols[flag_col] = mu_sts.isna()
        fallback_flags.append(flag_col)

    # --- Early baseline (cold-start) ---
    fit_mask = BASELINE_FIT_MASK
    for c in base_cols:
        x       = winsorize_series(df[c].where(fit_mask, np.nan))
        x_shift = x.groupby(df["uuid"], sort=False).shift(1)

        roll = x_shift.groupby(df["uuid"], sort=False).rolling(EARLY_WINDOW, min_periods=EARLY_MINP_GLOBAL)
        mu_eg = roll.mean().reset_index(level=0, drop=True).sort_index()
        sd_eg = roll.std(ddof=0).reset_index(level=0, drop=True).sort_index()
        floor = STD_FLOOR_MAP.get(c, DEFAULT_FLOOR)
        sd_eg = sd_eg.replace(0, np.nan).fillna(floor).clip(lower=floor)
        new_cols[f"{c}_mean_early_g"] = mu_eg
        new_cols[f"{c}_std_early_g"]  = sd_eg

        grp = [df["uuid"], df["is_weekend"]]
        x_shift2 = x.groupby(grp, sort=False).shift(1)
        roll2 = x_shift2.groupby(grp, sort=False).rolling(EARLY_WINDOW, min_periods=EARLY_MINP_SPLIT)
        mu_es = roll2.mean().reset_index(level=[0, 1], drop=True).sort_index()
        sd_es = roll2.std(ddof=0).reset_index(level=[0, 1], drop=True).sort_index()
        sd_es = sd_es.replace(0, np.nan).fillna(floor).clip(lower=floor)
        new_cols[f"{c}_mean_early_s"] = mu_es.fillna(new_cols[f"{c}_mean_early_g"])
        new_cols[f"{c}_std_early_s"]  = sd_es.fillna(new_cols[f"{c}_std_early_g"])

    # final baseline: ST 우선, 없으면 Early
    for c in base_cols:
        new_cols[f"{c}_mean_final"] = new_cols[f"{c}_mean_st_s"].fillna(new_cols[f"{c}_mean_early_s"])
        new_cols[f"{c}_std_final"]  = new_cols[f"{c}_std_st_s"].fillna(new_cols[f"{c}_std_early_s"])

    # 한번에 concat → PerformanceWarning 제거
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    df["fallback_any_stsplit"] = df[fallback_flags].any(axis=1)

    mean_cols_st = [f"{c}_mean_st_s" for c in base_cols]
    std_cols_st  = [f"{c}_std_st_s"  for c in base_cols]
    df["baseline_ready"] = df[mean_cols_st + std_cols_st].notna().all(axis=1)

    early_mean_cols = [f"{c}_mean_early_s" for c in base_cols]
    early_std_cols  = [f"{c}_std_early_s"  for c in base_cols]
    df["early_ready"] = df[early_mean_cols + early_std_cols].notna().all(axis=1)

    mean_final_cols = [f"{c}_mean_final" for c in base_cols]
    std_final_cols  = [f"{c}_std_final"  for c in base_cols]
    df["analysis_ready"] = df[mean_final_cols + std_final_cols].notna().all(axis=1)

    return df


# ---------------------------------------------------------------------------
# 4. Z-score  (v3_MODEL Section 4 그대로)
# ---------------------------------------------------------------------------

def _compute_z_scores(df: pd.DataFrame, base_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    z_mask = df["baseline_ready"].fillna(False) & (df["quality_state"] == "GOOD")
    df["z_score_eligible"] = df["has_activity"].astype("boolean").fillna(False)

    z_base_cols = [c for c in Z_BASE_CANDIDATES if c in df.columns]
    usable = [c for c in z_base_cols if
              f"{c}_mean_st_s" in df.columns and f"{c}_std_st_s" in df.columns]

    Z_COLS: List[str] = []
    z_new_cols: dict = {}
    for c in usable:
        x = pd.to_numeric(df[c], errors="coerce")
        mu = pd.to_numeric(df[f"{c}_mean_st_s"], errors="coerce")
        sd = pd.to_numeric(df[f"{c}_std_st_s"],  errors="coerce")
        z = (x - mu) / sd
        z = z.replace([np.inf, -np.inf], np.nan)
        z = Z_CLIP * np.tanh(z / Z_CLIP)
        z_new_cols[f"{c}_z"] = z.where(z_mask, np.nan)
        Z_COLS.append(f"{c}_z")

    df = pd.concat([df, pd.DataFrame(z_new_cols, index=df.index)], axis=1)
    return df, Z_COLS


# ---------------------------------------------------------------------------
# 5. Anomaly  (v3_MODEL Section 5 그대로)
# ---------------------------------------------------------------------------

def _train_isolation_forest(df: pd.DataFrame, Z_COLS: List[str]) -> IsolationForest:
    """IsolationForest 학습. train.py에서만 호출."""
    train_mask = (
        df["baseline_fit_mask"].fillna(False)
        & df["baseline_ready"].fillna(False)
        & (df["quality_state"] == "GOOD")
    )
    z_na_ratio = df[Z_COLS].isna().mean()
    Z_USE = [c for c in Z_COLS if z_na_ratio[c] < 0.60] or Z_COLS[:]

    X_train = df.loc[train_mask, Z_USE].fillna(0.0)

    # train_mask가 0이면 (더미 데이터 cold-start) 순차 fallback
    if len(X_train) == 0:
        score_mask_tmp = df["baseline_ready"].fillna(False) & (df["quality_state"] == "GOOD")
        X_train = df.loc[score_mask_tmp, Z_USE].fillna(0.0)
        print(f"⚠️ [IF] train_mask=0 → fallback to score_mask ({len(X_train)} rows)")
    if len(X_train) == 0:
        X_train = df[Z_USE].fillna(0.0)
        print(f"⚠️ [IF] still 0 → using all rows ({len(X_train)} rows)")
    if len(X_train) < 200:
        print(f"⚠️ [IF] X_train={len(X_train)} rows. Consider relaxing train_mask.")

    iso = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination=IF_CONTAM,
        max_samples=IF_MAX_SAMPLES,
        random_state=IF_SEED,
        n_jobs=-1,
    )
    iso.fit(X_train)
    return iso


def _apply_anomaly_score(df: pd.DataFrame, iso: IsolationForest, Z_COLS: List[str]) -> pd.DataFrame:
    """학습된 모델로 anomaly_score 계산."""
    z_na_ratio = df[Z_COLS].isna().mean()
    Z_USE      = [c for c in Z_COLS if z_na_ratio[c] < 0.60] or Z_COLS[:]

    score_mask = df["baseline_ready"].fillna(False) & (df["quality_state"] == "GOOD")
    X_score    = df.loc[score_mask, Z_USE].fillna(0.0)

    df["anomaly_score_raw"] = np.nan
    if len(X_score) > 0:
        raw = -iso.decision_function(X_score)
        df.loc[score_mask, "anomaly_score_raw"] = raw

    # per-user quantile scaling (0~1)
    df["anomaly_score"] = np.nan
    df["anomaly_flag"] = pd.Series([pd.NA] * len(df), dtype="boolean")

    tmp = df.loc[score_mask, ["uuid", "anomaly_score_raw"]].copy()
    if tmp.empty:
        return df

    q = tmp.groupby("uuid")["anomaly_score_raw"].quantile([Q_LOW, Q_HIGH]).unstack()
    q.columns = ["q_low", "q_high"]
    tmp = tmp.join(q, on="uuid")
    den = (tmp["q_high"] - tmp["q_low"]).replace(0, np.nan)
    tmp["anomaly_score"] = ((tmp["anomaly_score_raw"] - tmp["q_low"]) / den).clip(0, 1).fillna(0.0)
    tmp["anomaly_flag"] = (tmp["anomaly_score_raw"] >= tmp["q_high"]).astype("boolean")

    df.loc[score_mask, "anomaly_score"] = tmp["anomaly_score"].values
    df.loc[score_mask, "anomaly_flag"]  = tmp["anomaly_flag"].values
    return df


# ---------------------------------------------------------------------------
# 6. Risk  (v3_MODEL Section 6 그대로)
# ---------------------------------------------------------------------------

def _compute_risk(df: pd.DataFrame, Z_COLS: List[str], base_cols: List[str]) -> pd.DataFrame:
    risk_mask = (
        (df["quality_state"] == "GOOD")
        & df["baseline_ready"].fillna(False)
        & (~df["qc_any_low_cov"].fillna(False))
        & (~df["qc_missing_any"].fillna(False))
    )
    df["risk_mask"] = risk_mask

    # Z 그룹 대표값
    Z_ABS = df[Z_COLS].abs()
    Z_GROUPS = {k: [c for c in v if c in Z_COLS] for k, v in Z_GROUP_DEFS.items()}
    Z_GROUPS = {k: v for k, v in Z_GROUPS.items() if v}

    risk_new_cols: dict = {}
    for gname, cols in Z_GROUPS.items():
        q = CORE_Q if gname == "core" else OTHER_Q
        risk_new_cols[f"z_{gname}_rep"] = Z_ABS[cols].quantile(q=q, axis=1, interpolation="linear")

    group_rep_cols = [f"z_{g}_rep" for g in Z_GROUPS]
    tmp_rep = pd.DataFrame(risk_new_cols, index=df.index)
    risk_new_cols["z_group_trim_mean"] = tmp_rep[group_rep_cols].apply(_trim_mean, axis=1)
    risk_new_cols["z_trim_mean_abs"]   = risk_new_cols["z_group_trim_mean"]
    risk_new_cols["z_max_abs"]         = Z_ABS.max(axis=1)
    df = pd.concat([df, pd.DataFrame(risk_new_cols, index=df.index)], axis=1)

    ref  = df.loc[risk_mask, "z_group_trim_mean"].dropna()
    TAU  = max(float(ref.quantile(0.90)) if len(ref) else 1.0, 1e-6)
    df["z_score"] = (1.0 - np.exp(-df["z_group_trim_mean"] / (TAU + 1e-6))).clip(0, 0.999)

    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").clip(0, 1)
    df["a_score"]       = df["anomaly_score"].clip(0, 0.999)

    df["risk_pre"] = (W_Z * df["z_score"] + W_A * df["a_score"]).clip(0, 1)
    df["risk_adj"] = (df["risk_pre"] * df["context_mode"].apply(_resolve_discount)).astype(float)
    df.loc[~risk_mask, ["risk_pre", "risk_adj"]] = np.nan

    # EMA smoothing
    df["risk_score"] = (
        df.groupby("uuid", sort=False)["risk_adj"]
        .transform(lambda s: s.ewm(alpha=EMA_ALPHA, adjust=False, ignore_na=True).mean())
    )
    df.loc[~risk_mask, "risk_score"] = np.nan
    df = df.copy()

    # Early risk (cold-start)
    early_mask = (
        (df["quality_state"] == "GOOD")
        & df["cold_stage"].isin(["WARMUP", "SEMI_READY"])
        & df["early_ready"].fillna(False)
        & (~df["qc_any_low_cov"].fillna(False))
        & (~df["qc_missing_any"].fillna(False))
    )
    early_z_cols: List[str] = []
    early_z_new: dict = {}
    for c in base_cols:
        mu = df[f"{c}_mean_early_s"]
        sd = df[f"{c}_std_early_s"]
        z  = (df[c] - mu) / sd
        z  = z.replace([np.inf, -np.inf], np.nan)
        z  = Z_CLIP_EARLY * np.tanh(z / Z_CLIP_EARLY)
        col = f"{c}_z_early"
        early_z_new[col] = z.where(early_mask, np.nan)
        early_z_cols.append(col)
    df = pd.concat([df, pd.DataFrame(early_z_new, index=df.index)], axis=1)

    early_ref_med = df.loc[early_mask, early_z_cols].abs().median()
    early_ref_med = early_ref_med.replace(0, np.nan).fillna(1.0)
    abs_early_norm = df[early_z_cols].abs().div(early_ref_med, axis=1)
    df["z_trim_mean_abs_early"] = abs_early_norm.apply(_trim_mean, axis=1)

    ref_e = df.loc[early_mask, "z_trim_mean_abs_early"].dropna()
    TAU_E = max(float(ref_e.quantile(0.90)) if len(ref_e) else 1.0, 1e-6)
    raw_e = 1.0 - np.exp(-df["z_trim_mean_abs_early"] / (TAU_E + 1e-6))
    raw_e_max = 1.0 - np.exp(-(Z_CLIP_EARLY) / (TAU_E + 1e-6))
    df["early_risk"] = np.nan
    df.loc[early_mask, "early_risk"] = (raw_e / (raw_e_max + 1e-6)).clip(0, 1)

    # final_risk = risk_score 우선, 없으면 early_risk
    df["final_risk"] = df["risk_score"]
    df.loc[df["final_risk"].isna(), "final_risk"] = df["early_risk"]

    return df


# ---------------------------------------------------------------------------
# 7. Recovery / Drift  (v3_MODEL Section 7 그대로)
# ---------------------------------------------------------------------------

def _compute_recovery_drift(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    df = df.sort_values(["uuid", "date"]).reset_index(drop=True)

    # bad_history / stable_hold
    df["bad_history_flag"] = (
        df.groupby("uuid", sort=False)["final_risk"]
        .transform(lambda s: s.shift(1).rolling(HOLD_WINDOW, min_periods=HOLD_MINP).max() >= BAD_THR)
    ).astype("boolean")

    df["stable_hold_flag"] = (
        df["bad_history_flag"].fillna(False)
        & df["final_risk"].notna()
        & (df["final_risk"] < STABLE_THR)
    ).astype("boolean")

    # Drift
    drift_watch = [c for c in DRIFT_WATCH_COLS_CANDIDATES if c in base_cols]
    anchor_mask = df["day_idx"].between(ANCHOR_START, ANCHOR_END) & df["baseline_fit_mask"]

    for c in drift_watch:
        anchor_stats = (
            df.loc[anchor_mask].groupby("uuid")[c]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{c}_anchor_mean", "std": f"{c}_anchor_std"})
        )
        floor = STD_FLOOR_MAP.get(c, DEFAULT_FLOOR)
        anchor_stats[f"{c}_anchor_std"] = anchor_stats[f"{c}_anchor_std"].clip(lower=floor)
        df = df.join(anchor_stats, on="uuid")
        mean_col = f"{c}_mean_final"
        if mean_col in df.columns:
            drift_z = (df[mean_col] - df[f"{c}_anchor_mean"]) / df[f"{c}_anchor_std"]
            df[f"{c}_drift_z"] = drift_z
        else:
            df[f"{c}_drift_z"] = np.nan

    drift_z_cols = [f"{c}_drift_z" for c in drift_watch]
    drift_z_abs  = df[drift_z_cols].abs() if drift_z_cols else pd.DataFrame(index=df.index)

    df["drift_flag"] = (
        (df["day_idx"] > ANCHOR_END)
        & (drift_z_abs.max(axis=1) >= DRIFT_Z_THR if not drift_z_abs.empty else False)
    ).astype("boolean").fillna(False)

    df["drift_top_feature"] = ""
    drift_valid = df["drift_flag"].fillna(False)
    if drift_valid.any() and not drift_z_abs.empty:
        df.loc[drift_valid, "drift_top_feature"] = (
            drift_z_abs.loc[drift_valid].idxmax(axis=1).str.replace("_drift_z", "")
        )

    return df


# ---------------------------------------------------------------------------
# 8. Output 컬럼 정리  (v3_MODEL Section 8 그대로)
# ---------------------------------------------------------------------------

def _build_output(df: pd.DataFrame, Z_COLS: List[str]) -> pd.DataFrame:
    Z_GROUP_REP_COLS = [c for c in df.columns if c.startswith("z_") and c.endswith("_rep")]

    out_cols = [
        "uuid", "date", "day_idx", "cold_stage",
        "early_ready", "early_risk", "final_risk",
        "quality_state", "context_mode", "baseline_ready",
        "risk_score", "risk_pre", "risk_adj",
        "z_score", "a_score",
        "z_trim_mean_abs", "z_max_abs",
        "anomaly_flag", "anomaly_score",
        "partial_flag", "travel_flag", "tz_flag",
        "bad_history_flag", "stable_hold_flag", "drift_flag", "drift_top_feature",
    ] + Z_COLS + Z_GROUP_REP_COLS

    out_cols = [c for c in dict.fromkeys(out_cols) if c in df.columns]
    export   = df.copy()
    export["date"] = pd.to_datetime(export["date"]).dt.strftime("%Y-%m-%d")
    return export[out_cols]


# ---------------------------------------------------------------------------
# 9. 메인 진입점
# ---------------------------------------------------------------------------

def run(
    feature_df: pd.DataFrame,
    mode: str = "infer",
    iso: Optional[IsolationForest] = None,
) -> Tuple[pd.DataFrame, Optional[IsolationForest]]:
    """
    MODEL Step 메인 함수.

    Parameters
    ----------
    feature_df : pd.DataFrame
        features.run() 출력 (daily_feature_v3)
    mode : str
        "train"  → IsolationForest 학습 포함 (train.py 용)
        "infer"  → 학습된 iso 로 추론만   (batch_runner.py 용)
    iso : IsolationForest, optional
        mode="infer" 일 때 반드시 전달

    Returns
    -------
    model_out : pd.DataFrame
        model_out_v3 스키마
    iso : IsolationForest or None
        mode="train" 이면 학습된 모델, "infer" 이면 None
    """
    print(f"[MODEL] mode={mode}, input: {feature_df.shape}")

    df, base_cols = _preprocess(feature_df)
    df            = _compute_baseline(df, base_cols)
    df, Z_COLS    = _compute_z_scores(df, base_cols)

    if mode == "train":
        iso = _train_isolation_forest(df, Z_COLS)
    else:
        if iso is None:
            raise ValueError("[MODEL] mode='infer' requires iso to be passed.")

    df = _apply_anomaly_score(df, iso, Z_COLS)
    df = _compute_risk(df, Z_COLS, base_cols)
    df = _compute_recovery_drift(df, base_cols)

    model_out = _build_output(df, Z_COLS)

    print(f"[MODEL] output: {model_out.shape}")
    if mode == "train":
        return model_out, iso
    return model_out, None