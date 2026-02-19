"""
src/steps/features.py
─────────────────────
notebooks/v3_FE.ipynb 전체 로직 모듈화.

notebook → 모듈 핵심 변환:
  - CSV 루프 + chunk          → silver_events DataFrame groupby
  - to_dt_date_hour_from_ms   → local_ts = timestamp + tz_offset_minutes*60*1000
  - Step_Count 이벤트 행 추출  → silver_events.step_count 컬럼 직접 합산

I/O 없음 – 순수 pandas/numpy 계산만.

공개 API:
  build_daily_features(silver_events, config, *, debug=False) -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 0.1  센서 / 이벤트 매핑  (notebook 0.1 그대로)
# ──────────────────────────────────────────────────────────────────────────────

CORE_SENSORS:    List[str] = ["Screen", "UserAct"]
PASSIVE_SENSORS: List[str] = ["Notif"]
AUX_SENSORS:     List[str] = ["Unlock"]

EVENT_MAP: Dict[str, str] = {
    "USER_INTERACTION":       "UserAct",
    "SCREEN_INTERACTIVE":     "Screen",
    "SCREEN_NON_INTERACTIVE": "Screen",
    "NOTIF_INTERRUPTION":     "Notif",
    "KEYGUARD_HIDDEN":        "Unlock",
    # 공개 데이터 aliases
    "SCREEN":   "Screen",
    "USERACT":  "UserAct",
    "NOTIF":    "Notif",
    "UNLOCK":   "Unlock",
}

RHYTHM_SENSORS:  List[str] = ["Screen", "UserAct", "Unlock"]
GAP_SENSORS:     List[str] = ["Screen", "UserAct", "Unlock"]
SESSION_EVENTS:  List[str] = ["SCREEN_INTERACTIVE", "SCREEN_NON_INTERACTIVE"]
MOBILITY_EVENTS: List[str] = ["CELL_CHANGE", "WIFI_SSID"]
META_EVENT_TYPE: str       = "HEARTBEAT"

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


# ──────────────────────────────────────────────────────────────────────────────
# 1.1  수치 유틸  (notebook 1.1 그대로)
# ──────────────────────────────────────────────────────────────────────────────

def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy. notebook entropy_from_counts."""
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
# 1.4  전처리  (notebook preprocess_chunk + to_dt_date_hour_from_ms 통합)
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_silver_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    silver_events DataFrame 전처리.

    ★ 핵심 차이: local_ts = timestamp + tz_offset_minutes * 60 * 1000
       (notebook은 raw ts 그대로 사용)
    """
    df = df.copy()

    # 1) 중복 제거
    if {"uuid", "timestamp", "event_name"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["uuid", "timestamp", "event_name"])

    # 2) cell_lac 정리
    if "cell_lac" in df.columns:
        df["cell_lac"] = df["cell_lac"].replace("unknown", np.nan)
        df["cell_lac"] = pd.to_numeric(df["cell_lac"], errors="coerce")

    # 3) local_ts 계산 (★)
    tz_ms = (
        pd.to_numeric(df.get("tz_offset_minutes", 0), errors="coerce").fillna(0)
        * 60 * 1000
    )
    df["_local_ts_ms"] = pd.to_numeric(df["timestamp"], errors="coerce") + tz_ms
    df["dt"]   = pd.to_datetime(df["_local_ts_ms"], unit="ms", errors="coerce")
    df         = df.dropna(subset=["dt"])
    df["date"] = df["dt"].dt.normalize()
    df["hour"] = df["dt"].dt.hour

    # 4) event_std (raw 이벤트명 표준화)
    df["event_std"] = (
        df["event_name"].astype("string").str.strip().str.upper()
    )

    # 5) event_cat 매핑
    df["event_cat"] = df["event_std"].map(EVENT_MAP).astype("string")

    # 6) uuid
    if "uuid" not in df.columns:
        raise ValueError("silver_events must have 'uuid' column")
    df["uuid"] = df["uuid"].astype("string")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.1  활동 피처  (notebook build_daily_activity)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 2.1 → groupby."""
    activity_sensors = CORE_SENSORS + PASSIVE_SENSORS + AUX_SENSORS  # Screen, UserAct, Notif, Unlock

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

    return daily[["uuid", "date"] + activity_sensors].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 2.2  리듬 피처  (notebook build_daily_rhythm)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_rhythm(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 2.2 → groupby."""
    NIGHT = (0, 6)
    DAY   = (6, 18)
    EVE   = (18, 24)

    EMPTY_COLS = [
        "uuid", "date", "rhythm_event_cnt", "rhythm_low_coverage",
        "night_ratio", "hour_entropy", "day_ratio", "evening_ratio",
        "peak_hour", "peak_ratio",
    ]

    sub = df[df["event_cat"].isin(RHYTHM_SENSORS)].copy()
    if sub.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    # 24시간 bincount per (uuid, date)
    hour_grp = (
        sub.groupby(["uuid", "date", "hour"])
           .size()
           .reset_index(name="cnt")
    )

    out: List[Dict[str, Any]] = []
    for (uid, d), g in hour_grp.groupby(["uuid", "date"]):
        counts24 = np.zeros(24, dtype=np.int64)
        counts24[g["hour"].to_numpy()] = g["cnt"].to_numpy()
        total = int(counts24.sum())

        if total < MIN_RHYTHM_EVENTS:
            out.append({
                "uuid": uid, "date": d,
                "rhythm_event_cnt": total,
                "rhythm_low_coverage": True,
                **{k: np.nan for k in [
                    "night_ratio", "hour_entropy",
                    "day_ratio", "evening_ratio", "peak_hour", "peak_ratio",
                ]},
            })
            continue

        max_val = int(np.max(counts24))
        peak_h  = int(np.argmax(counts24)) if max_val > 0 else np.nan
        peak_r  = float(max_val) / total if total > 0 else np.nan
        h_ent   = _entropy_from_counts(counts24.astype(float))

        night_cnt = int(counts24[NIGHT[0]:NIGHT[1]].sum())
        day_cnt   = int(counts24[DAY[0]:DAY[1]].sum())
        eve_cnt   = int(counts24[EVE[0]:EVE[1]].sum())

        out.append({
            "uuid":               uid,
            "date":               d,
            "rhythm_event_cnt":   total,
            "rhythm_low_coverage": False,
            "night_ratio":        (night_cnt + 1) / (total + 24),   # smoothed
            "hour_entropy":       h_ent if np.isfinite(h_ent) else np.nan,
            "day_ratio":          float(day_cnt / total),
            "evening_ratio":      float(eve_cnt / total),
            "peak_hour":          float(peak_h),
            "peak_ratio":         float(peak_r),
        })

    return pd.DataFrame(out) if out else pd.DataFrame(columns=EMPTY_COLS)


# ──────────────────────────────────────────────────────────────────────────────
# 2.3  공백 피처  (notebook build_daily_gap)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    notebook 2.3 → groupby.

    DF 전체를 한 번에 처리하므로 청크 간 연속성 처리 불필요.
    overnight_gap: 전일 last_dt → 당일 first_dt (연속 날짜일 때만).
    """
    EMPTY_COLS = [
        "uuid", "date", "gap_event_cnt", "gap_low_coverage",
        "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h",
        "gap_long_ratio", "overnight_gap", "first_hour", "last_hour",
    ]

    sub = df[df["event_cat"].isin(GAP_SENSORS)].sort_values(["uuid", "dt"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    out: List[Dict[str, Any]] = []

    for uid, u_df in sub.groupby("uuid"):
        u_df = u_df.sort_values("dt")

        # 날짜별 메타 수집
        event_cnt:  Dict[pd.Timestamp, int]          = {}
        first_dt:   Dict[pd.Timestamp, pd.Timestamp] = {}
        last_dt:    Dict[pd.Timestamp, pd.Timestamp] = {}
        gaps_intra: Dict[pd.Timestamp, list]         = {}

        for d, d_df in u_df.groupby("date"):
            d_key = pd.Timestamp(d)
            event_cnt[d_key] = len(d_df)
            first_dt[d_key]  = d_df["dt"].iloc[0]
            last_dt[d_key]   = d_df["dt"].iloc[-1]
            if len(d_df) > 1:
                diffs = (
                    d_df["dt"].diff().dt.total_seconds().dropna().to_numpy() / 3600.0
                )
                gaps_intra[d_key] = diffs[diffs > 0].tolist()

        # overnight_gap
        overnight: Dict[pd.Timestamp, float] = {}
        sorted_days = sorted(first_dt.keys())
        for i in range(1, len(sorted_days)):
            prev_d, curr_d = sorted_days[i - 1], sorted_days[i]
            if (curr_d - prev_d).days == 1:
                og = (first_dt[curr_d] - last_dt[prev_d]).total_seconds() / 3600.0
                if og >= 0:
                    overnight[curr_d] = og

        for d in sorted_days:
            total  = event_cnt[d]
            og_val = overnight.get(d, np.nan)
            res: Dict[str, Any] = {
                "uuid":            uid,
                "date":            d,
                "gap_event_cnt":   total,
                "overnight_gap":   og_val,
                "first_hour":      float(first_dt[d].hour),
                "last_hour":       float(last_dt[d].hour),
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
                combined = gaps_intra.get(d, []).copy()
                if not np.isnan(og_val):
                    combined.append(og_val)
                gaps = np.array(combined, dtype=float)

                if gaps.size == 0:
                    res.update({
                        "gap_max": np.nan, "gap_p95": np.nan,
                        "gap_cnt_2h": 0, "gap_cnt_6h": 0, "gap_long_ratio": 0.0,
                    })
                else:
                    l_sum = float(np.sum(gaps[gaps >= GAP_THR_HOURS]))
                    res.update({
                        "gap_max":        float(np.max(gaps)),
                        "gap_p95":        float(_safe_quantile(gaps, 0.95)),
                        "gap_cnt_2h":     int(np.sum(gaps >= GAP_THR_HOURS)),
                        "gap_cnt_6h":     int(np.sum(gaps >= GAP_THR_6HOURS)),
                        "gap_long_ratio": min(l_sum / 24.0, 1.0),
                    })
            out.append(res)

    return pd.DataFrame(out) if out else pd.DataFrame(columns=EMPTY_COLS)


# ──────────────────────────────────────────────────────────────────────────────
# 2.4  세션 피처  (notebook build_daily_session)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_session(df: pd.DataFrame, long_session_sec: int = 1800) -> pd.DataFrame:
    """notebook 2.4 → groupby."""
    START = "SCREEN_INTERACTIVE"
    END   = "SCREEN_NON_INTERACTIVE"

    EMPTY_COLS = [
        "uuid", "date", "session_cnt", "session_total_sec",
        "session_mean_sec", "long_session_cnt",
    ]

    sub = df[df["event_std"].isin(SESSION_EVENTS)].sort_values(["uuid", "date", "dt"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    out: List[Dict[str, Any]] = []

    for (uid, d), g in sub.groupby(["uuid", "date"]):
        seq = sorted(zip(g["dt"], g["event_std"]), key=lambda x: x[0])
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

        if not durations:
            out.append({
                "uuid": uid, "date": d,
                "session_cnt": 0, "session_total_sec": 0.0,
                "session_mean_sec": np.nan, "long_session_cnt": 0,
            })
        else:
            arr = np.array(durations)
            out.append({
                "uuid":             uid,
                "date":             d,
                "session_cnt":      int(len(arr)),
                "session_total_sec": float(arr.sum()),
                "session_mean_sec": float(arr.mean()),
                "long_session_cnt": int((arr >= long_session_sec).sum()),
            })

    return pd.DataFrame(out) if out else pd.DataFrame(columns=EMPTY_COLS)


# ──────────────────────────────────────────────────────────────────────────────
# 2.5  이동성 피처  (notebook build_daily_mobility)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_mobility(df: pd.DataFrame) -> pd.DataFrame:
    """
    notebook 2.5 → groupby.

    step_count: silver_events.step_count 컬럼 직접 합산 (★ notebook과 다른 부분).
    결과는 동일 – non-step 이벤트의 step_count=0 이므로.
    """
    EMPTY_COLS = [
        "uuid", "date", "cell_change_cnt", "wifi_change_cnt_est",
        "unique_wifi_cnt", "unique_cell_cnt", "step_sum",
    ]

    cell_cnt:     Dict[Tuple, int]   = {}
    cell_uniqs:   Dict[Tuple, set]   = {}
    wifi_uniqs:   Dict[Tuple, set]   = {}
    wifi_changes: Dict[Tuple, int]   = {}
    step_sums:    Dict[Tuple, float] = {}

    # --- CELL / WIFI ---
    mob = df[df["event_std"].isin(MOBILITY_EVENTS)].sort_values(["uuid", "date", "dt"]).copy()
    if not mob.empty:
        for (uid, d), g in mob.groupby(["uuid", "date"]):
            key = (str(uid), pd.Timestamp(d))

            c_sub = g[g["event_std"] == "CELL_CHANGE"]
            cell_cnt[key] = cell_cnt.get(key, 0) + len(c_sub)
            if "cell_lac" in c_sub.columns:
                cells = c_sub["cell_lac"].dropna().astype(str).tolist()
                cell_uniqs.setdefault(key, set()).update(cells)

            w_sub = g[g["event_std"] == "WIFI_SSID"].copy()
            if not w_sub.empty and "wifi_ssid" in w_sub.columns:
                w_sub["wifi_ssid"] = w_sub["wifi_ssid"].astype(str)
                ssids = w_sub["wifi_ssid"].values
                wifi_uniqs.setdefault(key, set()).update(ssids.tolist())
                # 연속 SSID 변화 카운트
                changes = int(np.sum(ssids[1:] != ssids[:-1]))
                wifi_changes[key] = wifi_changes.get(key, 0) + changes

    # --- STEP COUNT (★ 모듈 전용: step_count 컬럼 직접 합산) ---
    if "step_count" in df.columns:
        sc = df.copy()
        sc["step_count"] = pd.to_numeric(sc["step_count"], errors="coerce")
        for (uid, d), g in sc.groupby(["uuid", "date"]):
            key = (str(uid), pd.Timestamp(d))
            s = float(g["step_count"].sum(min_count=1))
            if not np.isnan(s):
                step_sums[key] = step_sums.get(key, 0.0) + s

    all_keys = set(cell_cnt) | set(step_sums) | set(wifi_uniqs)
    if not all_keys:
        return pd.DataFrame(columns=EMPTY_COLS)

    out: List[Dict[str, Any]] = []
    for (uid, d) in sorted(all_keys):
        key = (uid, d)
        out.append({
            "uuid":                uid,
            "date":                d,
            "cell_change_cnt":     int(cell_cnt.get(key, 0)),
            "wifi_change_cnt_est": int(wifi_changes.get(key, 0)),
            "unique_wifi_cnt":     float(len(wifi_uniqs.get(key, set()))),
            "unique_cell_cnt":     float(len(cell_uniqs.get(key, set()))),
            "step_sum":            step_sums.get(key, np.nan),
        })

    return pd.DataFrame(out)


# ──────────────────────────────────────────────────────────────────────────────
# 2.6  메타/QC 피처  (notebook build_daily_meta_qc)
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_meta_qc(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 2.6 → groupby."""
    EMPTY_COLS = [
        "uuid", "date", "heartbeat_cnt", "retry_max", "queue_max",
        "tz_offset_min", "tz_offset_max", "tz_changed", "qc_last_ts_max",
    ]

    # heartbeat 행 필터링
    hb = pd.DataFrame()
    if "type" in df.columns:
        hb = df[df["type"].astype("string").str.lower() == "heartbeat"].copy()
    if hb.empty:
        hb = df[df["event_std"].str.lower() == "heartbeat"].copy()
    if hb.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    out: List[Dict[str, Any]] = []
    for (uid, d), g in hb.groupby(["uuid", "date"]):
        def _col_max(col):
            if col not in g.columns:
                return np.nan
            return float(pd.to_numeric(g[col], errors="coerce").max())

        def _col_min(col):
            if col not in g.columns:
                return np.nan
            return float(pd.to_numeric(g[col], errors="coerce").min())

        tmin = _col_min("tz_offset_minutes")
        tmax = _col_max("tz_offset_minutes")
        tz_changed = False
        if pd.notna(tmin) and pd.notna(tmax):
            tz_changed = bool(abs(tmax - tmin) >= TZ_CHANGE_THR_MINUTES)

        out.append({
            "uuid":           uid,
            "date":           d,
            "heartbeat_cnt":  int(len(g)),
            "retry_max":      _col_max("retry_count"),
            "queue_max":      _col_max("queue_size"),
            "tz_offset_min":  tmin,
            "tz_offset_max":  tmax,
            "tz_changed":     tz_changed,
            "qc_last_ts_max": _col_max("client_last_event_ts"),
        })

    return pd.DataFrame(out) if out else pd.DataFrame(columns=EMPTY_COLS)


# ──────────────────────────────────────────────────────────────────────────────
# 3.0  병합 유틸  (notebook _assert_or_dedup / _outer_merge_on_uuid_date)
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_uuid_date(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not {"uuid", "date"}.issubset(df.columns):
        raise ValueError(f"{name}: missing uuid/date columns")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df.duplicated(["uuid", "date"]).any():
        df = (
            df.sort_values(["uuid", "date"])
              .drop_duplicates(["uuid", "date"], keep="last")
              .reset_index(drop=True)
        )
    return df


def _outer_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy() if right is not None else pd.DataFrame()
    if right is None or right.empty:
        return left.copy()
    return left.merge(right, on=["uuid", "date"], how="outer")


def _to01_or_nan(s: pd.Series) -> pd.Series:
    """bool/object 섞여있는 시리즈 → 0/1/NaN float."""
    if s is None:
        return s
    x = s.copy().where(~pd.isna(s), np.nan)
    if x.dtype == bool:
        return x.astype(float)
    if pd.api.types.is_object_dtype(x):
        x = x.replace({"True": 1, "False": 0, True: 1, False: 0})
    return pd.to_numeric(x, errors="coerce")


# ──────────────────────────────────────────────────────────────────────────────
# 3.1  병합  (notebook build_daily_feature_table_v3)
# ──────────────────────────────────────────────────────────────────────────────

def _merge_features(
    activity_df: pd.DataFrame,
    rhythm_df:   pd.DataFrame,
    gap_df:      pd.DataFrame,
    session_df:  pd.DataFrame,
    mob_df:      pd.DataFrame,
    meta_df:     pd.DataFrame,
) -> pd.DataFrame:
    """notebook 3.1 build_daily_feature_table_v3의 merge 부분."""
    dfs = {
        "activity": activity_df, "rhythm": rhythm_df, "gap": gap_df,
        "session":  session_df,  "mob":    mob_df,     "meta": meta_df,
    }
    cleaned = {k: _dedup_uuid_date(v, k) for k, v in dfs.items()}

    df = cleaned["activity"].copy()
    for name in ("rhythm", "gap", "session", "mob", "meta"):
        df = _outer_merge(df, cleaned[name])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = (
        df.dropna(subset=["uuid", "date"])
          .sort_values(["uuid", "date"])
          .reset_index(drop=True)
    )

    # Unlock
    if "Unlock" not in df.columns:
        df["Unlock"] = 0
    df["Unlock"]     = pd.to_numeric(df["Unlock"], errors="coerce").fillna(0).astype(int)
    df["unlock_cnt"] = df["Unlock"]

    # CORE + PASSIVE 타입 고정
    for c in CORE_SENSORS + PASSIVE_SENSORS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["unlock_cnt"] = pd.to_numeric(df["unlock_cnt"], errors="coerce").fillna(0).astype(int)

    # daily_event_cnt = core 합 (Screen + UserAct 만)
    core_present = [c for c in CORE_SENSORS if c in df.columns]
    df["daily_event_cnt"] = df[core_present].sum(axis=1).astype(int)
    df["has_activity"]    = df["daily_event_cnt"] > 0

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.2  QC flags  (notebook add_qc_flags_v3)
# ──────────────────────────────────────────────────────────────────────────────

def _add_qc_flags_v3(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 3.2 add_qc_flags_v3."""
    df = df.copy()

    # core QC
    df["qc_core_low_cov"] = df["daily_event_cnt"] < MIN_DAILY_EVENTS
    df["qc_core_very_low_activity"] = (
        (df["daily_event_cnt"] > 0) & (df["daily_event_cnt"] < PARTIAL_CORE_MIN_EVENTS)
    )

    # rhythm QC
    if "rhythm_low_coverage" not in df.columns:
        df["rhythm_low_coverage"] = True
    df["rhythm_low_coverage"] = (
        df["rhythm_low_coverage"].fillna(True)
        .infer_objects(copy=False)
        .astype(bool)
    )
    df["qc_rhythm_low_cov"] = df["rhythm_low_coverage"]

    # gap QC
    if "gap_low_coverage" not in df.columns:
        df["gap_low_coverage"] = True
    df["gap_low_coverage"] = (
        df["gap_low_coverage"].fillna(True)
        .infer_objects(copy=False)
        .astype(bool)
    )
    df["qc_gap_low_cov"] = df["gap_low_coverage"]

    # meta QC
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

    # tz_changed fallback
    if "tz_changed" not in df.columns:
        if "tz_offset_min" in df.columns and "tz_offset_max" in df.columns:
            tmin = pd.to_numeric(df["tz_offset_min"], errors="coerce")
            tmax = pd.to_numeric(df["tz_offset_max"], errors="coerce")
            df["tz_changed"] = (tmax - tmin).abs() >= TZ_CHANGE_THR_MINUTES

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.3  soft context signals  (notebook add_soft_context_signals_v3)
# ──────────────────────────────────────────────────────────────────────────────

def _add_soft_context_signals_v3(df: pd.DataFrame) -> pd.DataFrame:
    """notebook 3.3 add_soft_context_signals_v3."""
    df = df.copy()

    # (1) tz 변화 신호
    if "tz_changed" in df.columns:
        df["tz_change_signal"] = _to01_or_nan(df["tz_changed"].astype("boolean"))
    else:
        df["tz_change_signal"] = 0.0

    # (2) partial 신호
    core_very_low = (
        df.get("qc_core_very_low_activity", pd.Series(False, index=df.index))
          .fillna(False)
          .astype(bool)
    )
    retry01 = _to01_or_nan(df.get("qc_meta_retry_warn", pd.Series(np.nan, index=df.index)))
    queue01 = _to01_or_nan(df.get("qc_meta_queue_warn", pd.Series(np.nan, index=df.index)))

    base = core_very_low.astype(int)
    df["partial_signal_raw"] = (
        base
        + pd.to_numeric(retry01, errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(queue01, errors="coerce").fillna(0).astype(int)
    ).astype(float)
    df.loc[~core_very_low, "partial_signal_raw"] = 0.0

    # (3) travel 신호
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
        u1 = pd.to_numeric(df["unique_cell_cnt"], errors="coerce").fillna(0)
        u1_prev = u1.groupby(df["uuid"]).shift(1)
        travel += ((u1_prev.notna()) & (u1 >= (u1_prev + 3))).astype(float)

    df["travel_signal_raw"] = travel
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.4  delta 피처  (notebook add_delta_features_v3)
# ──────────────────────────────────────────────────────────────────────────────

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
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_features(
    silver_events: pd.DataFrame,
    config: dict,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """
    silver_events DataFrame → 유저별 일별 피처 DataFrame.

    Parameters
    ----------
    silver_events : pd.DataFrame
        ETL 출력. 필수: uuid, timestamp(ms epoch), event_name, tz_offset_minutes.
        선택: type, step_count, cell_lac, wifi_ssid,
               retry_count, queue_size, client_last_event_ts.
    config : dict
        config.yaml 결과 (fe.long_session_sec 참조).
    debug : bool
        True이면 단계별 shape 출력.

    Returns
    -------
    pd.DataFrame  (uuid, date 기준 1행)
    """
    long_session_sec = int(config.get("fe", {}).get("long_session_sec", 1800))

    # 1) 전처리
    df = _preprocess_silver_events(silver_events)
    if debug:
        print(f"[FE] preprocessed: {df.shape}")

    # 2) 피처 빌더 (notebook 2.x)
    activity_df = _build_daily_activity(df)
    rhythm_df   = _build_daily_rhythm(df)
    gap_df      = _build_daily_gap(df)
    session_df  = _build_daily_session(df, long_session_sec=long_session_sec)
    mob_df      = _build_daily_mobility(df)
    meta_df     = _build_daily_meta_qc(df)

    if debug:
        for name, sub in [
            ("activity", activity_df), ("rhythm", rhythm_df), ("gap", gap_df),
            ("session", session_df),   ("mob",    mob_df),    ("meta", meta_df),
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
