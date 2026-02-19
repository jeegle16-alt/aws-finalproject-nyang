"""
src/steps/features.py
─────────────────────
notebooks/v3_FE.ipynb 의 피처 엔지니어링 로직을 모듈화한 파일.

주요 변경사항 (notebook → 모듈):
  - CSV 루프          → groupby("uuid", "date") 방식
  - 청크 처리          → 불필요 (DataFrame 통째로 처리)
  - local_ts 계산     → timestamp + tz_offset_minutes * 60 * 1000 명시 적용
  - step_count        → silver_events.step_count 컬럼 직접 합산
  - I/O               → 없음 (순수 pandas 로직만)

공개 API:
  build_daily_features(silver_events, config, *, debug=False) -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 0.1  센서 / 이벤트 매핑  (notebook 0.1 그대로)
# ──────────────────────────────────────────────────────────────────────────────
CORE_SENSORS: List[str]    = ["Screen", "UserAct"]
PASSIVE_SENSORS: List[str] = ["Notif"]
AUX_SENSORS: List[str]     = ["Unlock"]

EVENT_MAP: Dict[str, str] = {
    "USER_INTERACTION":       "UserAct",
    "SCREEN_INTERACTIVE":     "Screen",
    "SCREEN_NON_INTERACTIVE": "Screen",
    "NOTIF_INTERRUPTION":     "Notif",
    "KEYGUARD_HIDDEN":        "Unlock",
    # 공개데이터 aliases
    "SCREEN":   "Screen",
    "USERACT":  "UserAct",
    "NOTIF":    "Notif",
    "UNLOCK":   "Unlock",
}

RHYTHM_SENSORS: List[str] = ["Screen", "UserAct", "Unlock"]
GAP_SENSORS:    List[str] = ["Screen", "UserAct", "Unlock"]
SESSION_EVENTS: List[str] = ["SCREEN_INTERACTIVE", "SCREEN_NON_INTERACTIVE"]
MOBILITY_EVENTS: List[str] = ["CELL_CHANGE", "WIFI_SSID"]
META_EVENT_TYPE = "HEARTBEAT"

STEP_SENSOR_ALIASES: List[str] = [
    "Step_Count", "Step", "Steps", "STEP_COUNT", "STEP", "Acc_Avg",
]

# ──────────────────────────────────────────────────────────────────────────────
# 0.2  QC 임계값  (notebook 0.2 그대로)
# ──────────────────────────────────────────────────────────────────────────────
MIN_DAILY_EVENTS        = 50
MIN_RHYTHM_EVENTS       = 50
MIN_GAP_EVENTS          = 50
GAP_THR_HOURS           = 2.0
GAP_THR_6HOURS          = 6.0
MIN_HEARTBEAT_PER_DAY   = 1
RETRY_WARN_THR          = 3
QUEUE_WARN_THR          = 20
TZ_CHANGE_THR_MINUTES   = 60
PARTIAL_CORE_MIN_EVENTS = 10
WINSORIZE_LOWER         = 0.005
WINSORIZE_UPPER         = 0.995

# baseline.py / model.py 와 공유할 상수
STD_FLOOR_MAP: Dict[str, float] = {}   # downstream에서 채움
BASE_COLS: List[str] = CORE_SENSORS + PASSIVE_SENSORS + ["unlock_cnt", "daily_event_cnt"]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  수치 유틸  (notebook 1.1 그대로)
# ──────────────────────────────────────────────────────────────────────────────
def _entropy_from_counts(counts: np.ndarray) -> float:
    s = float(np.nansum(counts))
    if s <= 0:
        return np.nan
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  전처리  (모듈 전용 – notebook의 preprocess_chunk + 시간변환 통합)
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess_silver_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    silver_events DataFrame을 분석 가능한 형태로 전처리.

    notebook의 preprocess_chunk + to_dt_date_hour_from_ms + map_event_to_cat 통합.
    ★ 핵심 차이: local_ts = timestamp + tz_offset_minutes * 60 * 1000
    """
    df = df.copy()

    # 1) 중복 제거 (uuid, timestamp, event_name 기준)
    if {"uuid", "timestamp", "event_name"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["uuid", "timestamp", "event_name"])

    # 2) cell_lac: "unknown" → NaN
    if "cell_lac" in df.columns:
        df["cell_lac"] = df["cell_lac"].replace("unknown", np.nan)
        df["cell_lac"] = pd.to_numeric(df["cell_lac"], errors="coerce")

    # 3) local_ts 계산 (★ notebook과 다른 부분: tz_offset 명시 적용)
    tz_offset_ms = (
        pd.to_numeric(df.get("tz_offset_minutes", 0), errors="coerce").fillna(0) * 60 * 1000
    )
    df["_local_ts_ms"] = pd.to_numeric(df["timestamp"], errors="coerce") + tz_offset_ms
    df["dt"]   = pd.to_datetime(df["_local_ts_ms"], unit="ms", errors="coerce")
    df         = df.dropna(subset=["dt"])
    df["date"] = df["dt"].dt.normalize()
    df["hour"] = df["dt"].dt.hour

    # 4) event_std 표준화
    df["event_std"] = df["event_name"].astype("string").str.strip().str.upper()

    # 5) event_cat 매핑
    df["event_cat"] = df["event_std"].map(EVENT_MAP).astype("string")

    # 6) uuid 보장
    if "uuid" not in df.columns:
        raise ValueError("silver_events must have 'uuid' column")
    df["uuid"] = df["uuid"].astype("string")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  피처 빌더들  (notebook 2.x → groupby 방식으로 변환)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 2.1 build_daily_activity → groupby 방식."""
    activity_sensors = CORE_SENSORS + PASSIVE_SENSORS + AUX_SENSORS
    sub = df[df["event_cat"].isin(activity_sensors)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["uuid", "date"] + activity_sensors)

    daily = (
        sub.groupby(["uuid", "date", "event_cat"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for c in activity_sensors:
        if c not in daily.columns:
            daily[c] = 0
    return daily[["uuid", "date"] + activity_sensors]


def _build_daily_rhythm(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 2.2 build_daily_rhythm → groupby 방식."""
    NIGHT_RANGE = (0, 6)
    DAY_RANGE   = (6, 18)
    EVE_RANGE   = (18, 24)

    _EMPTY_COLS = [
        "uuid", "date", "rhythm_event_cnt", "rhythm_low_coverage",
        "night_ratio", "hour_entropy", "day_ratio", "evening_ratio",
        "peak_hour", "peak_ratio",
    ]

    sub = df[df["event_cat"].isin(RHYTHM_SENSORS)].copy()
    if sub.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    hour_grp = sub.groupby(["uuid", "date", "hour"]).size().reset_index(name="cnt")

    out_rows: List[Dict[str, Any]] = []
    for (uid, d), g in hour_grp.groupby(["uuid", "date"]):
        counts24 = np.zeros(24, dtype=np.int64)
        counts24[g["hour"].to_numpy()] = g["cnt"].to_numpy()
        total = int(counts24.sum())

        if total < MIN_RHYTHM_EVENTS:
            out_rows.append({
                "uuid": uid, "date": d,
                "rhythm_event_cnt": total, "rhythm_low_coverage": True,
                **{k: np.nan for k in [
                    "night_ratio", "hour_entropy",
                    "day_ratio", "evening_ratio", "peak_hour", "peak_ratio",
                ]},
            })
            continue

        max_val   = int(np.max(counts24))
        peak_h    = int(np.argmax(counts24)) if max_val > 0 else np.nan
        peak_r    = float(max_val) / total if total > 0 else np.nan
        h_ent     = _entropy_from_counts(counts24.astype(float))
        night_cnt = int(counts24[NIGHT_RANGE[0]:NIGHT_RANGE[1]].sum())
        day_cnt   = int(counts24[DAY_RANGE[0]:DAY_RANGE[1]].sum())
        eve_cnt   = int(counts24[EVE_RANGE[0]:EVE_RANGE[1]].sum())

        out_rows.append({
            "uuid": uid, "date": d,
            "rhythm_event_cnt":  total,
            "night_ratio":       (night_cnt + 1) / (total + 24),
            "hour_entropy":      h_ent if np.isfinite(h_ent) else np.nan,
            "day_ratio":         float(day_cnt / total),
            "evening_ratio":     float(eve_cnt / total),
            "peak_hour":         float(peak_h),
            "peak_ratio":        float(peak_r),
            "rhythm_low_coverage": False,
        })

    return pd.DataFrame(out_rows) if out_rows else pd.DataFrame(columns=_EMPTY_COLS)


def _build_daily_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    notebook 2.3 build_daily_gap → groupby 방식.

    DF 전체가 메모리에 있으므로 청크 간 연속성 처리 불필요.
    overnight_gap: 연속 날짜에서 last_dt[d-1] → first_dt[d] 자동 계산.
    """
    _EMPTY_COLS = [
        "uuid", "date", "gap_event_cnt", "gap_low_coverage",
        "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h",
        "gap_long_ratio", "overnight_gap", "first_hour", "last_hour",
    ]

    sub = df[df["event_cat"].isin(GAP_SENSORS)].sort_values(["uuid", "date", "dt"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    out_rows: List[Dict[str, Any]] = []

    for uid, u_df in sub.groupby("uuid"):
        u_df = u_df.sort_values("dt")

        first_dt_by_date: Dict = {}
        last_dt_by_date:  Dict = {}
        gaps_by_date:     Dict = {}
        event_cnt_by_date: Dict = {}

        for d, d_df in u_df.groupby("date"):
            d_idx = pd.Timestamp(d)
            event_cnt_by_date[d_idx] = len(d_df)
            first_dt_by_date[d_idx]  = d_df["dt"].iloc[0]
            last_dt_by_date[d_idx]   = d_df["dt"].iloc[-1]

            if len(d_df) > 1:
                diffs = (
                    d_df["dt"].diff().dt.total_seconds().dropna().to_numpy() / 3600.0
                )
                gaps_by_date[d_idx] = diffs[diffs > 0].tolist()

        # overnight gap
        overnight_gap_by_date: Dict = {}
        sorted_days = sorted(first_dt_by_date.keys())
        for i in range(1, len(sorted_days)):
            prev_d, curr_d = sorted_days[i - 1], sorted_days[i]
            if (curr_d - prev_d).days == 1:
                og = (
                    first_dt_by_date[curr_d] - last_dt_by_date[prev_d]
                ).total_seconds() / 3600.0
                if og >= 0:
                    overnight_gap_by_date[curr_d] = og

        for d in sorted_days:
            total   = event_cnt_by_date[d]
            og_val  = overnight_gap_by_date.get(d, np.nan)
            first_h = float(first_dt_by_date[d].hour)
            last_h  = float(last_dt_by_date[d].hour)

            res: Dict[str, Any] = {
                "uuid": uid, "date": d,
                "gap_event_cnt":    total,
                "overnight_gap":    og_val,
                "first_hour":       first_h,
                "last_hour":        last_h,
                "gap_low_coverage": False,
            }

            if total < MIN_GAP_EVENTS:
                res.update({
                    "gap_low_coverage": True,
                    "gap_max": np.nan, "gap_p95": np.nan,
                    "gap_cnt_2h": np.nan, "gap_cnt_6h": np.nan,
                    "gap_long_ratio": np.nan,
                })
            else:
                combined = gaps_by_date.get(d, []).copy()
                if not np.isnan(og_val):
                    combined.append(og_val)
                gaps = np.array(combined, dtype=float)

                if gaps.size == 0:
                    stats: Dict[str, Any] = {
                        "gap_max": np.nan, "gap_p95": np.nan,
                        "gap_cnt_2h": 0, "gap_cnt_6h": 0, "gap_long_ratio": 0.0,
                    }
                else:
                    l_sum = float(np.sum(gaps[gaps >= GAP_THR_HOURS]))
                    stats = {
                        "gap_max":        float(np.max(gaps)),
                        "gap_p95":        float(_safe_quantile(gaps, 0.95)),
                        "gap_cnt_2h":     int(np.sum(gaps >= GAP_THR_HOURS)),
                        "gap_cnt_6h":     int(np.sum(gaps >= GAP_THR_6HOURS)),
                        "gap_long_ratio": min(l_sum / 24.0, 1.0),
                    }
                res.update(stats)

            out_rows.append(res)

    return pd.DataFrame(out_rows) if out_rows else pd.DataFrame(columns=_EMPTY_COLS)


def _build_daily_session(df: pd.DataFrame, long_session_sec: int = 1800) -> pd.DataFrame:
    """notebook 2.4 build_daily_session → groupby 방식."""
    START = "SCREEN_INTERACTIVE"
    END   = "SCREEN_NON_INTERACTIVE"

    _EMPTY_COLS = [
        "uuid", "date", "session_cnt", "session_total_sec",
        "session_mean_sec", "long_session_cnt",
    ]

    sub = df[df["event_std"].isin(SESSION_EVENTS)].sort_values(["uuid", "date", "dt"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    out_rows: List[Dict[str, Any]] = []

    for (uid, d), g in sub.groupby(["uuid", "date"]):
        seq = list(zip(g["dt"], g["event_std"]))
        durations: List[float] = []
        open_start: Optional[pd.Timestamp] = None

        for t, ev in seq:
            if ev == START:
                if open_start is None:
                    open_start = t
            elif ev == END:
                if open_start is not None:
                    dur = (t - open_start).total_seconds()
                    if 0 < dur < (6 * 3600):
                        durations.append(dur)
                    open_start = None

        session_cnt = len(durations)
        if session_cnt == 0:
            out_rows.append({
                "uuid": uid, "date": d,
                "session_cnt": 0, "session_total_sec": 0.0,
                "session_mean_sec": np.nan, "long_session_cnt": 0,
            })
        else:
            dur_arr = np.array(durations)
            out_rows.append({
                "uuid": uid, "date": d,
                "session_cnt":       int(session_cnt),
                "session_total_sec": float(np.sum(dur_arr)),
                "session_mean_sec":  float(np.mean(dur_arr)),
                "long_session_cnt":  int(np.sum(dur_arr >= long_session_sec)),
            })

    return pd.DataFrame(out_rows) if out_rows else pd.DataFrame(columns=_EMPTY_COLS)


def _build_daily_mobility(df: pd.DataFrame) -> pd.DataFrame:
    """
    notebook 2.5 build_daily_mobility → groupby 방식.

    step_count: silver_events.step_count 컬럼 직접 합산 (★ notebook과 다른 부분).
    """
    _EMPTY_COLS = [
        "uuid", "date", "cell_change_cnt", "wifi_change_cnt_est",
        "unique_wifi_cnt", "unique_cell_cnt", "step_sum",
    ]

    # --- CELL / WIFI ---
    mob = df[df["event_std"].isin(MOBILITY_EVENTS)].sort_values(["uuid", "date", "dt"]).copy()

    cell_cnt:     Dict = {}
    cell_uniqs:   Dict = {}
    wifi_uniqs:   Dict = {}
    wifi_changes: Dict = {}

    if not mob.empty:
        for (uid, d), g in mob.groupby(["uuid", "date"]):
            key = (uid, pd.Timestamp(d))

            c_sub = g[g["event_std"] == "CELL_CHANGE"]
            cell_cnt[key] = len(c_sub)
            if "cell_lac" in c_sub.columns:
                cells = c_sub["cell_lac"].dropna().astype(str).tolist()
                cell_uniqs[key] = set(cells)

            w_sub = g[g["event_std"] == "WIFI_SSID"].copy()
            if not w_sub.empty and "wifi_ssid" in w_sub.columns:
                w_sub["wifi_ssid"] = w_sub["wifi_ssid"].astype(str)
                ssids = w_sub["wifi_ssid"].values
                wifi_uniqs[key] = set(ssids.tolist())
                changes = int(np.sum(ssids[1:] != ssids[:-1]))
                wifi_changes[key] = changes

    # --- STEP COUNT (★ 모듈 전용: step_count 컬럼 직접 합산) ---
    step_sums: Dict = {}
    if "step_count" in df.columns:
        step_df = df.copy()
        step_df["step_count"] = pd.to_numeric(step_df["step_count"], errors="coerce")
        for (uid, d), g in step_df.groupby(["uuid", "date"]):
            key = (uid, pd.Timestamp(d))
            s = float(g["step_count"].sum(min_count=1))
            if not np.isnan(s):
                step_sums[key] = s

    all_keys = set(cell_cnt) | set(step_sums) | set(wifi_uniqs)
    if not all_keys:
        return pd.DataFrame(columns=_EMPTY_COLS)

    out_rows: List[Dict[str, Any]] = []
    for (uid, d) in sorted(all_keys):
        key = (uid, d)
        out_rows.append({
            "uuid":               uid,
            "date":               d,
            "cell_change_cnt":    int(cell_cnt.get(key, 0)),
            "wifi_change_cnt_est": int(wifi_changes.get(key, 0)),
            "unique_wifi_cnt":    float(len(wifi_uniqs.get(key, set()))),
            "unique_cell_cnt":    float(len(cell_uniqs.get(key, set()))),
            "step_sum":           step_sums.get(key, np.nan),
        })

    return pd.DataFrame(out_rows)


def _build_daily_meta_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    notebook 2.6 build_daily_meta_qc → groupby 방식.
    silver_events의 type 컬럼으로 heartbeat 행 직접 필터링.
    """
    _EMPTY_COLS = [
        "uuid", "date", "heartbeat_cnt", "retry_max", "queue_max",
        "tz_offset_min", "tz_offset_max", "tz_changed", "qc_last_ts_max",
    ]

    hb = pd.DataFrame()
    if "type" in df.columns:
        hb = df[df["type"].astype("string").str.lower() == "heartbeat"].copy()

    if hb.empty:
        hb = df[df["event_std"].str.lower() == "heartbeat"].copy()

    if hb.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    out_rows: List[Dict[str, Any]] = []
    for (uid, d), g in hb.groupby(["uuid", "date"]):
        tmin  = pd.to_numeric(g["tz_offset_minutes"],    errors="coerce").min()  if "tz_offset_minutes"    in g.columns else np.nan
        tmax  = pd.to_numeric(g["tz_offset_minutes"],    errors="coerce").max()  if "tz_offset_minutes"    in g.columns else np.nan
        rmax  = pd.to_numeric(g["retry_count"],          errors="coerce").max()  if "retry_count"          in g.columns else np.nan
        qmax  = pd.to_numeric(g["queue_size"],           errors="coerce").max()  if "queue_size"           in g.columns else np.nan
        lemax = pd.to_numeric(g["client_last_event_ts"], errors="coerce").max()  if "client_last_event_ts" in g.columns else np.nan

        tz_changed = False
        if pd.notna(tmin) and pd.notna(tmax):
            tz_changed = bool(abs(tmax - tmin) >= TZ_CHANGE_THR_MINUTES)

        out_rows.append({
            "uuid":           uid,
            "date":           d,
            "heartbeat_cnt":  int(len(g)),
            "retry_max":      rmax,
            "queue_max":      qmax,
            "tz_offset_min":  tmin,
            "tz_offset_max":  tmax,
            "tz_changed":     tz_changed,
            "qc_last_ts_max": lemax,
        })

    return pd.DataFrame(out_rows)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  병합 유틸  (notebook 3.0 그대로)
# ──────────────────────────────────────────────────────────────────────────────

def _assert_or_dedup_uuid_date(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not {"uuid", "date"}.issubset(df.columns):
        raise ValueError(f"{name}: missing uuid/date columns")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    dup = int(df.duplicated(["uuid", "date"]).sum())
    if dup == 0:
        return df
    return (
        df.sort_values(["uuid", "date"])
        .drop_duplicates(["uuid", "date"], keep="last")
        .reset_index(drop=True)
    )


def _outer_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy()
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, on=["uuid", "date"], how="outer")


def _to01_or_nan(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    x = s.copy()
    x = x.where(~pd.isna(x), np.nan)
    if x.dtype == bool:
        return x.astype(float)
    if pd.api.types.is_object_dtype(x):
        x = x.replace({"True": 1, "False": 0, True: 1, False: 0})
    return pd.to_numeric(x, errors="coerce")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  파생 피처  (notebook 3.2 / 3.3 / 3.4 그대로)
# ──────────────────────────────────────────────────────────────────────────────

def _add_qc_flags_v3(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 3.2 add_qc_flags_v3."""
    df = df.copy()

    df["qc_core_low_cov"] = df["daily_event_cnt"] < MIN_DAILY_EVENTS
    df["qc_core_very_low_activity"] = (
        (df["daily_event_cnt"] > 0) & (df["daily_event_cnt"] < PARTIAL_CORE_MIN_EVENTS)
    )

    if "rhythm_low_coverage" not in df.columns:
        df["rhythm_low_coverage"] = True
    df["rhythm_low_coverage"] = (
        df["rhythm_low_coverage"].fillna(True).infer_objects(copy=False).astype(bool)
    )
    df["qc_rhythm_low_cov"] = df["rhythm_low_coverage"]

    if "gap_low_coverage" not in df.columns:
        df["gap_low_coverage"] = True
    df["gap_low_coverage"] = (
        df["gap_low_coverage"].fillna(True).infer_objects(copy=False).astype(bool)
    )
    df["qc_gap_low_cov"] = df["gap_low_coverage"]

    if "heartbeat_cnt" in df.columns:
        df["qc_meta_low_heartbeat"] = (
            pd.to_numeric(df["heartbeat_cnt"], errors="coerce").fillna(0).astype(int)
            < MIN_HEARTBEAT_PER_DAY
        )
    else:
        df["qc_meta_low_heartbeat"] = np.nan

    if "retry_max" in df.columns:
        df["qc_meta_retry_warn"] = (
            pd.to_numeric(df["retry_max"], errors="coerce").fillna(-np.inf) >= RETRY_WARN_THR
        )
    else:
        df["qc_meta_retry_warn"] = np.nan

    if "queue_max" in df.columns:
        df["qc_meta_queue_warn"] = (
            pd.to_numeric(df["queue_max"], errors="coerce").fillna(-np.inf) >= QUEUE_WARN_THR
        )
    else:
        df["qc_meta_queue_warn"] = np.nan

    if "tz_changed" not in df.columns:
        if "tz_offset_min" in df.columns and "tz_offset_max" in df.columns:
            tmin = pd.to_numeric(df["tz_offset_min"], errors="coerce")
            tmax = pd.to_numeric(df["tz_offset_max"], errors="coerce")
            df["tz_changed"] = (tmax - tmin).abs() >= TZ_CHANGE_THR_MINUTES

    return df


def _add_soft_context_signals_v3(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 3.3 add_soft_context_signals_v3."""
    df = df.copy()

    # tz 변화 신호
    if "tz_changed" in df.columns:
        df["tz_change_signal"] = _to01_or_nan(df["tz_changed"].astype("boolean"))
    else:
        df["tz_change_signal"] = 0.0

    # partial 신호
    core_very_low = df.get("qc_core_very_low_activity", pd.Series(False, index=df.index))
    core_very_low = core_very_low.fillna(False).astype(bool)
    retry01 = _to01_or_nan(df.get("qc_meta_retry_warn", pd.Series(np.nan, index=df.index)))
    queue01 = _to01_or_nan(df.get("qc_meta_queue_warn", pd.Series(np.nan, index=df.index)))

    base = core_very_low.astype(int)
    df["partial_signal_raw"] = (
        base
        + pd.to_numeric(retry01, errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(queue01, errors="coerce").fillna(0).astype(int)
    ).astype(float)
    df.loc[~core_very_low, "partial_signal_raw"] = 0.0

    # travel 신호
    travel = pd.Series(0, index=df.index, dtype=float)
    tz01 = pd.to_numeric(df["tz_change_signal"], errors="coerce").fillna(0).astype(int)
    travel += (tz01 * 2).astype(float)

    if "cell_change_cnt" in df.columns:
        c = pd.to_numeric(df["cell_change_cnt"], errors="coerce")
        c_prev = c.groupby(df["uuid"]).shift(1)
        travel += ((c_prev.notna()) & (c >= (c_prev * 2 + 10))).astype(float)

    if "wifi_change_cnt_est" in df.columns:
        w = pd.to_numeric(df["wifi_change_cnt_est"], errors="coerce")
        w_prev = w.groupby(df["uuid"]).shift(1)
        travel += ((w_prev.notna()) & (w >= (w_prev * 2 + 10))).astype(float)

    if "unique_cell_cnt" in df.columns:
        u1 = pd.to_numeric(
            df.get("unique_cell_cnt", pd.Series(np.nan, index=df.index)), errors="coerce"
        ).fillna(0)
        u1_prev = u1.groupby(df["uuid"]).shift(1)
        travel += ((u1_prev.notna()) & (u1 >= (u1_prev + 3))).astype(float)

    df["travel_signal_raw"] = travel
    return df


def _add_delta_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 3.4 add_delta_features_v3."""
    df = df.copy().sort_values(["uuid", "date"]).reset_index(drop=True)

    delta_cols = [
        "daily_event_cnt", "night_ratio", "hour_entropy",
        "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h",
        "session_total_sec", "session_cnt", "step_sum",
        "cell_change_cnt", "wifi_change_cnt_est",
        "travel_signal_raw", "partial_signal_raw", "tz_change_signal",
    ]
    delta_cols = [c for c in delta_cols if c in df.columns]

    for c in delta_cols:
        x    = pd.to_numeric(df[c], errors="coerce")
        prev = x.groupby(df["uuid"]).shift(1)
        df[f"{c}_d1"] = x - prev

    ratio_cols = [c for c in ["daily_event_cnt", "session_total_sec", "step_sum"] if c in df.columns]
    eps = 1e-6
    for c in ratio_cols:
        x    = pd.to_numeric(df[c], errors="coerce")
        prev = x.groupby(df["uuid"]).shift(1)
        df[f"{c}_r1"] = (x - prev) / (prev.abs() + eps)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6.  병합 + 파생  (notebook 3.1 build_daily_feature_table_v3 그대로)
# ──────────────────────────────────────────────────────────────────────────────

def _merge_features(
    activity_df: pd.DataFrame,
    rhythm_df:   pd.DataFrame,
    gap_df:      pd.DataFrame,
    session_df:  pd.DataFrame,
    mob_df:      pd.DataFrame,
    meta_df:     pd.DataFrame,
) -> pd.DataFrame:
    activity_df = _assert_or_dedup_uuid_date(activity_df, "activity")
    rhythm_df   = _assert_or_dedup_uuid_date(rhythm_df,   "rhythm")
    gap_df      = _assert_or_dedup_uuid_date(gap_df,      "gap")
    session_df  = _assert_or_dedup_uuid_date(session_df,  "session")
    mob_df      = _assert_or_dedup_uuid_date(mob_df,      "mobility")
    meta_df     = _assert_or_dedup_uuid_date(meta_df,     "meta")

    df = activity_df.copy()
    df = _outer_merge(df, rhythm_df)
    df = _outer_merge(df, gap_df)
    df = _outer_merge(df, session_df)
    df = _outer_merge(df, mob_df)
    df = _outer_merge(df, meta_df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = (
        df.dropna(subset=["uuid", "date"])
        .sort_values(["uuid", "date"])
        .reset_index(drop=True)
    )

    # Unlock 표준화
    if "Unlock" not in df.columns:
        df["Unlock"] = 0
    df["Unlock"]     = pd.to_numeric(df["Unlock"], errors="coerce").fillna(0).astype(int)
    df["unlock_cnt"] = df["Unlock"]

    for c in CORE_SENSORS + PASSIVE_SENSORS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["unlock_cnt"] = pd.to_numeric(df["unlock_cnt"], errors="coerce").fillna(0).astype(int)

    core_cols_present = [c for c in CORE_SENSORS if c in df.columns]
    df["daily_event_cnt"] = df[core_cols_present].sum(axis=1).astype(int)
    df["has_activity"]    = (df["daily_event_cnt"] > 0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 7.  공개 API
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_features(
    silver_events: pd.DataFrame,
    config: dict,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    silver_events DataFrame → 유저별 일별 피처 DataFrame 반환.

    Parameters
    ----------
    silver_events : pd.DataFrame
        ETL 출력. 최소 컬럼: uuid, timestamp (ms epoch), event_name,
        tz_offset_minutes.
        선택 컬럼: type, step_count, cell_lac, wifi_ssid,
                   retry_count, queue_size, client_last_event_ts.
    config : dict
        config.yaml 로드 결과 (fe.long_session_sec 참조).
    debug : bool
        True이면 중간 단계 shape 출력.

    Returns
    -------
    pd.DataFrame  (uuid, date 기준 1행)
    """
    long_session_sec = int(
        config.get("fe", {}).get("long_session_sec", 1800)
    )

    # 1) 전처리
    df = _preprocess_silver_events(silver_events)
    if debug:
        print(f"[FE] preprocessed: {df.shape}")

    # 2) 피처 빌더 (notebook 2.x 각 함수)
    activity_df = _build_daily_activity(df)
    rhythm_df   = _build_daily_rhythm(df)
    gap_df      = _build_daily_gap(df)
    session_df  = _build_daily_session(df, long_session_sec=long_session_sec)
    mob_df      = _build_daily_mobility(df)
    meta_df     = _build_daily_meta_qc(df)

    if debug:
        for name, sub in [
            ("activity", activity_df), ("rhythm", rhythm_df),
            ("gap", gap_df),           ("session", session_df),
            ("mobility", mob_df),      ("meta", meta_df),
        ]:
            print(f"[FE] {name}: {sub.shape}")

    # 3) 병합 (notebook 3.1)
    merged = _merge_features(activity_df, rhythm_df, gap_df, session_df, mob_df, meta_df)

    # 4) 파생 피처 (notebook 3.2 / 3.3 / 3.4)
    merged = _add_qc_flags_v3(merged)
    merged = _add_soft_context_signals_v3(merged)
    merged = _add_delta_features_v3(merged)

    if debug:
        print(f"[FE] final: {merged.shape}")
        print(f"[FE] columns: {merged.columns.tolist()}")

    return merged
