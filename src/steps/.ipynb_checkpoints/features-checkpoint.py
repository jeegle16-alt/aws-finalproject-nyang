"""
src/steps/features.py
=====================
v3_FE.ipynb 로직을 그대로 포팅.

[변경점 - 로직 외]
- 입력: CSV Path list  →  pandas DataFrame (silver parquet에서 읽어온 raw 이벤트)
- 출력: daily_feature_v3 DataFrame (파일 저장은 batch_runner/train 담당)
- 로직: 100% 동일 (상수/함수/집계 순서 모두 유지)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

pd.set_option("future.no_silent_downcasting", True)

# ---------------------------------------------------------------------------
# 0. Schema constants  (v3_FE 그대로)
# ---------------------------------------------------------------------------

CORE_SENSORS    = ["Screen", "UserAct"]
PASSIVE_SENSORS = ["Notif"]
AUX_SENSORS     = ["Unlock"]

EVENT_MAP: Dict[str, str] = {
    "USER_INTERACTION":       "UserAct",
    "SCREEN_INTERACTIVE":     "Screen",
    "SCREEN_NON_INTERACTIVE": "Screen",
    "NOTIF_INTERRUPTION":     "Notif",
    "KEYGUARD_HIDDEN":        "Unlock",
    "SCREEN":   "Screen",
    "USERACT":  "UserAct",
    "NOTIF":    "Notif",
    "UNLOCK":   "Unlock",
}

RHYTHM_SENSORS  = ["Screen", "UserAct", "Unlock"]
GAP_SENSORS     = ["Screen", "UserAct", "Unlock"]
SESSION_EVENTS  = ["SCREEN_INTERACTIVE", "SCREEN_NON_INTERACTIVE"]
MOBILITY_EVENTS = ["CELL_CHANGE", "WIFI_SSID"]
META_EVENT_TYPE = "HEARTBEAT"
STEP_SENSOR_ALIASES = ["Step_Count", "Step", "Steps", "STEP_COUNT", "STEP", "Acc_Avg"]

# ---------------------------------------------------------------------------
# 1. Thresholds  (v3_FE 그대로)
# ---------------------------------------------------------------------------

MIN_DAILY_EVENTS        = 50
MIN_RHYTHM_EVENTS       = 50
MIN_GAP_EVENTS          = 50
MIN_HEARTBEAT_PER_DAY   = 1
RETRY_WARN_THR          = 3
QUEUE_WARN_THR          = 20
TZ_CHANGE_THR_MINUTES   = 60
PARTIAL_CORE_MIN_EVENTS = 10
GAP_THR_HOURS           = 2.0
GAP_THR_6HOURS          = 6.0

# ---------------------------------------------------------------------------
# 2. Column candidates  (v3_FE 그대로)
# ---------------------------------------------------------------------------

EVENT_COL_CANDIDATES = [
    "event_name", "sensor_id", "event",
    "name", "action", "type_name", "eventCode", "event_id", "event_type",
]
TS_COL_CANDIDATES = [
    "ts_raw", "timestamp", "ts",
    "time", "event_time", "created_at", "occurred_at",
    "client_ts", "client_time",
]
HEARTBEAT_TS_CANDIDATES = [
    "timestamp", "ts_raw", "ts",
    "client_ts", "client_time",
    "time", "event_time", "created_at", "occurred_at",
]

# ---------------------------------------------------------------------------
# 3. Utility functions  (v3_FE 그대로)
# ---------------------------------------------------------------------------

def entropy_from_counts(counts: np.ndarray) -> float:
    s = float(np.nansum(counts))
    if s <= 0:
        return np.nan
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def safe_concat(frames: List[pd.DataFrame], cols: List[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def pick_first_existing_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def standardize_event_ts_cols(
    df: pd.DataFrame,
    event_candidates: List[str],
    ts_candidates: List[str],
    out_event_col: str = "event_std",
    out_ts_col:    str = "ts_std",
) -> pd.DataFrame:
    cols  = df.columns.tolist()
    e_col = pick_first_existing_col(cols, event_candidates)
    t_col = pick_first_existing_col(cols, ts_candidates)

    if e_col is None:
        df[out_event_col] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df[out_event_col] = df[e_col].astype("string")
        df[out_event_col] = df[out_event_col].where(df[out_event_col].notna(), pd.NA)

    df[out_event_col] = (
        df[out_event_col].astype("string").str.strip().str.upper()
    )
    df[out_ts_col] = df[t_col] if t_col else np.nan
    return df


def to_dt_date_hour_from_ms(
    df: pd.DataFrame,
    ts_col: str = "ts_std",
    dt_col: str = "dt",
) -> pd.DataFrame:
    if ts_col not in df.columns:
        df[dt_col] = pd.NaT
        return df

    x = df[ts_col]
    if pd.api.types.is_numeric_dtype(x):
        xn = pd.to_numeric(x, errors="coerce")
        if xn.dropna().empty:
            df[dt_col] = pd.NaT
            return df
        med_len = xn.dropna().astype("int64").astype(str).str.len().median()
        if pd.notna(med_len) and med_len <= 10:
            dt = pd.to_datetime(xn, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(xn, unit="ms", errors="coerce")
    else:
        dt = pd.to_datetime(x, errors="coerce")

    df = df.assign(**{dt_col: dt}).dropna(subset=[dt_col])
    if df.empty:
        return df

    if "tz_offset_minutes" in df.columns:
        tz_ms = pd.to_numeric(df["tz_offset_minutes"], errors="coerce").fillna(540) * 60 * 1000
        dt_local = pd.to_datetime(df[ts_col].astype("int64") + tz_ms.astype("int64"), unit="ms", errors="coerce")
        df["date"] = dt_local.dt.normalize()
        df["hour"] = dt_local.dt.hour
    else:
        df["date"] = df[dt_col].dt.normalize()
        df["hour"] = df[dt_col].dt.hour
    return df
    
def map_event_to_cat(event_std: pd.Series, event_map: Dict[str, str]) -> pd.Series:
    e = event_std.astype("string").str.strip().str.upper()
    return e.map(event_map).astype("string")


def filter_by_events(df: pd.DataFrame, events: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    if "event_std" in df.columns:
        return df[df["event_std"].astype("string").isin(events)].copy()
    if "sensor_id" in df.columns:
        return df[df["sensor_id"].astype("string").isin(events)].copy()
    return pd.DataFrame(columns=df.columns)


def filter_by_cats(df: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    if "event_cat" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df["event_cat"].astype("string").isin(cats)].copy()


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
    out = (
        df.sort_values(["uuid", "date"])
        .drop_duplicates(["uuid", "date"], keep="last")
        .reset_index(drop=True)
    )
    print(f"⚠️ [{name}] duplicated (uuid,date)={dup} -> keep last row")
    return out


def _outer_merge_on_uuid_date(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# 4. preprocess_chunk  (v3_FE 그대로 - uuid 단위 DataFrame 받도록만 변경)
# ---------------------------------------------------------------------------

def preprocess_chunk(
    chunk: pd.DataFrame,
    uid: str,
    event_candidates: List[str],
    ts_candidates: List[str],
    event_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if event_map is None:
        event_map = EVENT_MAP
    if chunk is None or chunk.empty:
        return pd.DataFrame()

    chunk = standardize_event_ts_cols(chunk, event_candidates, ts_candidates)
    chunk = to_dt_date_hour_from_ms(chunk, ts_col="ts_std", dt_col="dt")
    if chunk.empty:
        return chunk

    if "uuid" in chunk.columns:
        chunk["uuid"] = chunk["uuid"].astype("string").fillna(uid)
    else:
        chunk["uuid"] = pd.Series([uid] * len(chunk), dtype="string")

    chunk["event_cat"] = map_event_to_cat(chunk["event_std"], event_map)
    return chunk


# ---------------------------------------------------------------------------
# 5. Feature builders  (v3_FE 로직 100% 동일)
# ---------------------------------------------------------------------------

def build_daily_activity(raw_df: pd.DataFrame) -> pd.DataFrame:
    activity_sensors = CORE_SENSORS + PASSIVE_SENSORS + AUX_SENSORS
    rows: List[pd.DataFrame] = []

    for uid, user_df in raw_df.groupby("uuid"):
        uid = str(uid)
        ch  = preprocess_chunk(user_df.copy(), uid, EVENT_COL_CANDIDATES, TS_COL_CANDIDATES)
        if ch.empty:
            continue
        ch = filter_by_cats(ch, activity_sensors)
        if ch.empty:
            continue

        summary = ch.groupby(["uuid", "date", "event_cat"]).size().reset_index(name="cnt")
        daily = (
            summary.groupby(["uuid", "date", "event_cat"])["cnt"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        for c in activity_sensors:
            if c not in daily.columns:
                daily[c] = 0
        target_cols = ["uuid", "date"] + activity_sensors
        rows.append(daily[[c for c in target_cols if c in daily.columns]])

    return safe_concat(rows, ["uuid", "date"] + activity_sensors)


def build_daily_rhythm(raw_df: pd.DataFrame) -> pd.DataFrame:
    NIGHT_RANGE = (0, 6)
    DAY_RANGE   = (6, 18)
    EVE_RANGE   = (18, 24)
    out_rows: List[Dict[str, Any]] = []

    for uid, user_df in raw_df.groupby("uuid"):
        uid = str(uid)
        ch  = preprocess_chunk(user_df.copy(), uid, EVENT_COL_CANDIDATES, TS_COL_CANDIDATES)
        if ch.empty:
            continue
        ch = filter_by_cats(ch, RHYTHM_SENSORS)
        if ch.empty:
            continue

        hour_counts: Dict[pd.Timestamp, np.ndarray] = {}
        grp = ch.groupby(["date", "hour"]).size().reset_index(name="cnt")
        for d, sub in grp.groupby("date"):
            arr = hour_counts.setdefault(d, np.zeros(24, dtype=np.int64))
            arr[sub["hour"].to_numpy()] += sub["cnt"].to_numpy()

        for d in sorted(hour_counts.keys()):
            counts24 = hour_counts[d]
            total    = int(counts24.sum())

            if total < MIN_RHYTHM_EVENTS:
                out_rows.append({
                    "uuid": uid, "date": d, "rhythm_event_cnt": total,
                    "rhythm_low_coverage": True,
                    **{k: np.nan for k in ["night_ratio", "hour_entropy",
                                           "day_ratio", "evening_ratio", "peak_hour", "peak_ratio"]},
                })
                continue

            max_val   = np.max(counts24)
            peak_h    = int(np.argmax(counts24)) if max_val > 0 else np.nan
            peak_r    = float(max_val) / total if total > 0 else np.nan
            h_ent     = entropy_from_counts(counts24.astype(float))
            night_cnt = int(counts24[NIGHT_RANGE[0]:NIGHT_RANGE[1]].sum())
            day_cnt   = int(counts24[DAY_RANGE[0]:DAY_RANGE[1]].sum())
            eve_cnt   = int(counts24[EVE_RANGE[0]:EVE_RANGE[1]].sum())

            out_rows.append({
                "uuid": uid, "date": d,
                "rhythm_event_cnt": total,
                "night_ratio":      (night_cnt + 1) / (total + 24),
                "hour_entropy":     h_ent if np.isfinite(h_ent) else np.nan,
                "day_ratio":        float(day_cnt / total),
                "evening_ratio":    float(eve_cnt / total),
                "peak_hour":        float(peak_h),
                "peak_ratio":       float(peak_r),
                "rhythm_low_coverage": False,
            })

    if not out_rows:
        return pd.DataFrame(columns=[
            "uuid", "date", "rhythm_event_cnt", "rhythm_low_coverage",
            "night_ratio", "hour_entropy", "day_ratio", "evening_ratio", "peak_hour", "peak_ratio",
        ])
    return pd.DataFrame(out_rows)


def build_daily_gap(raw_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []

    for uid, user_df in raw_df.groupby("uuid"):
        uid = str(uid)
        ch  = preprocess_chunk(user_df.copy(), uid, EVENT_COL_CANDIDATES, TS_COL_CANDIDATES)
        if ch.empty:
            continue
        ch = filter_by_cats(ch, GAP_SENSORS)
        if ch.empty:
            continue
        ch = ch.sort_values("dt")

        gaps_by_date:       Dict = {}
        event_cnt_by_date:  Dict = {}
        first_dt_by_date:   Dict = {}
        last_dt_by_date:    Dict = {}
        last_ts_per_date:   Dict = {}

        for d, sub in ch.groupby("date"):
            d_idx = pd.Timestamp(d)
            event_cnt_by_date[d_idx] = event_cnt_by_date.get(d_idx, 0) + len(sub)
            s_first, s_last = sub["dt"].iloc[0], sub["dt"].iloc[-1]
            first_dt_by_date[d_idx]  = min(first_dt_by_date.get(d_idx, s_first), s_first)
            last_dt_by_date[d_idx]   = max(last_dt_by_date.get(d_idx, s_last),   s_last)

            if d_idx in last_ts_per_date:
                cross = (s_first - last_ts_per_date[d_idx]).total_seconds() / 3600.0
                if cross > 0:
                    gaps_by_date.setdefault(d_idx, []).append(cross)

            if len(sub) > 1:
                diffs = sub["dt"].diff().dt.total_seconds().dropna().to_numpy() / 3600.0
                gaps_by_date.setdefault(d_idx, []).extend(diffs[diffs > 0].tolist())

            last_ts_per_date[d_idx] = s_last

        overnight_gap_by_date: Dict = {}
        sorted_days = sorted(first_dt_by_date.keys())
        for i in range(1, len(sorted_days)):
            prev_d, curr_d = sorted_days[i - 1], sorted_days[i]
            if (curr_d - prev_d).days == 1:
                og = (first_dt_by_date[curr_d] - last_dt_by_date[prev_d]).total_seconds() / 3600.0
                if og >= 0:
                    overnight_gap_by_date[curr_d] = og

        for d in sorted(event_cnt_by_date.keys()):
            total  = event_cnt_by_date[d]
            og_val = overnight_gap_by_date.get(d, np.nan)
            res = {
                "uuid": uid, "date": d, "gap_event_cnt": total,
                "overnight_gap": og_val,
                "first_hour": float(first_dt_by_date[d].hour),
                "last_hour":  float(last_dt_by_date[d].hour),
                "gap_low_coverage": False,
            }
            if total < MIN_GAP_EVENTS:
                res.update({
                    "gap_low_coverage": True, "gap_max": np.nan, "gap_p95": np.nan,
                    "gap_cnt_2h": np.nan, "gap_cnt_6h": np.nan, "gap_long_ratio": np.nan,
                })
            else:
                combined_gaps = gaps_by_date.get(d, []).copy()
                if not np.isnan(og_val):
                    combined_gaps.append(og_val)
                gaps = np.array(combined_gaps, dtype=float)
                if gaps.size == 0:
                    stats = {"gap_max": np.nan, "gap_p95": np.nan,
                             "gap_cnt_2h": 0, "gap_cnt_6h": 0, "gap_long_ratio": 0.0}
                else:
                    l_sum = float(np.sum(gaps[gaps >= GAP_THR_HOURS]))
                    stats = {
                        "gap_max":       float(np.max(gaps)),
                        "gap_p95":       float(safe_quantile(gaps, 0.95)),
                        "gap_cnt_2h":    int(np.sum(gaps >= GAP_THR_HOURS)),
                        "gap_cnt_6h":    int(np.sum(gaps >= GAP_THR_6HOURS)),
                        "gap_long_ratio": min(l_sum / 24.0, 1.0),
                    }
                res.update(stats)
            out_rows.append(res)

    if not out_rows:
        return pd.DataFrame(columns=[
            "uuid", "date", "gap_event_cnt", "gap_low_coverage",
            "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h",
            "gap_long_ratio", "overnight_gap", "first_hour", "last_hour",
        ])
    return pd.DataFrame(out_rows)


def build_daily_session(raw_df: pd.DataFrame) -> pd.DataFrame:
    START            = "SCREEN_INTERACTIVE"
    END              = "SCREEN_NON_INTERACTIVE"
    long_session_sec = 30 * 60
    out_rows: List[Dict[str, Any]] = []

    for uid, user_df in raw_df.groupby("uuid"):
        uid = str(uid)
        ch  = preprocess_chunk(user_df.copy(), uid, EVENT_COL_CANDIDATES, TS_COL_CANDIDATES)
        if ch.empty:
            continue
        ch = filter_by_events(ch, SESSION_EVENTS)
        if ch.empty:
            continue
        ch = ch.sort_values(["date", "dt"])

        seq_by_date: Dict[pd.Timestamp, List[Tuple]] = {}
        for d, sub in ch.groupby("date"):
            d_idx = pd.Timestamp(d)
            seq_by_date.setdefault(d_idx, []).extend(list(zip(sub["dt"], sub["event_std"])))

        for d in sorted(seq_by_date.keys()):
            seq        = sorted(seq_by_date[d], key=lambda x: x[0])
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

    if not out_rows:
        return pd.DataFrame(columns=[
            "uuid", "date", "session_cnt", "session_total_sec",
            "session_mean_sec", "long_session_cnt",
        ])
    return pd.DataFrame(out_rows)


def build_daily_mobility(raw_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []

    for uid, user_df in raw_df.groupby("uuid"):
        uid = str(uid)
        ch  = preprocess_chunk(user_df.copy(), uid, EVENT_COL_CANDIDATES, TS_COL_CANDIDATES)
        if ch.empty:
            continue

        cell_cnt:            Dict = {}
        cell_uniqs:          Dict = {}
        step_sum:            Dict = {}
        unique_wifi_ssids:   Dict = {}
        wifi_change_cnt_est: Dict = {}
        last_ssid_seen:      Dict = {}

        # mobility
        ch_m = filter_by_events(ch, MOBILITY_EVENTS).sort_values("dt")
        if not ch_m.empty:
            for d, sub in ch_m.groupby("date"):
                d_idx = pd.Timestamp(d)
                c_cnt = len(sub[sub["event_std"] == "CELL_CHANGE"])
                cell_cnt[d_idx] = cell_cnt.get(d_idx, 0) + c_cnt

                if "cell_lac" in sub.columns:
                    cells = sub[sub["event_std"] == "CELL_CHANGE"]["cell_lac"].dropna().astype(str)
                    cell_uniqs.setdefault(d_idx, set()).update(cells.tolist())

                if "wifi_ssid" in sub.columns:
                    wifi_sub = sub[sub["event_std"] == "WIFI_SSID"].copy()
                    wifi_sub["wifi_ssid"] = wifi_sub["wifi_ssid"].astype(str)
                    if not wifi_sub.empty:
                        unique_wifi_ssids.setdefault(d_idx, set()).update(wifi_sub["wifi_ssid"].tolist())
                        ssids        = wifi_sub["wifi_ssid"].values
                        current_last = last_ssid_seen.get(d_idx)
                        changes = 0
                        for i in range(len(ssids)):
                            if current_last is not None and ssids[i] != current_last:
                                changes += 1
                            current_last = ssids[i]
                        wifi_change_cnt_est[d_idx] = wifi_change_cnt_est.get(d_idx, 0) + changes
                        last_ssid_seen[d_idx] = current_last

        # step
        is_step = ch["event_std"].isin(STEP_SENSOR_ALIASES)
        if "sensor_id" in ch.columns:
            is_step |= ch["sensor_id"].astype(str).isin(STEP_SENSOR_ALIASES)
        ch_s = ch[is_step].copy()
        if not ch_s.empty:
            val_col = next((c for c in ["step_count", "value"] if c in ch_s.columns), None)
            if val_col:
                ch_s[val_col] = pd.to_numeric(ch_s[val_col], errors="coerce")
                ch_s = ch_s.loc[~ch_s.index.duplicated(keep="first")]
                for d, s in ch_s.groupby("date")[val_col].sum().items():
                    d_idx = pd.Timestamp(d)
                    step_sum[d_idx] = step_sum.get(d_idx, 0.0) + float(s)

        all_dates = sorted(set(cell_cnt) | set(step_sum) | set(unique_wifi_ssids))
        for d in all_dates:
            out_rows.append({
                "uuid": uid, "date": d,
                "cell_change_cnt":     int(cell_cnt.get(d, 0)),
                "wifi_change_cnt_est": int(wifi_change_cnt_est.get(d, 0)),
                "unique_wifi_cnt":     float(len(unique_wifi_ssids.get(d, set()))),
                "unique_cell_cnt":     float(len(cell_uniqs.get(d, set()))),
                "step_sum":            step_sum.get(d, np.nan),
            })

    if not out_rows:
        return pd.DataFrame(columns=[
            "uuid", "date", "cell_change_cnt", "wifi_change_cnt_est",
            "unique_wifi_cnt", "unique_cell_cnt", "step_sum",
        ])
    return pd.DataFrame(out_rows)


def build_daily_meta_qc(raw_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []

    for uid, user_df in raw_df.groupby("uuid"):
        uid = str(uid)

        ch_hb = pd.DataFrame()
        if "type" in user_df.columns:
            m     = user_df["type"].astype("string").str.lower() == str(META_EVENT_TYPE).lower()
            ch_hb = user_df[m].copy()

        if ch_hb.empty:
            ch2   = standardize_event_ts_cols(user_df.copy(), EVENT_COL_CANDIDATES, HEARTBEAT_TS_CANDIDATES)
            m     = ch2["event_std"].astype("string").str.strip().str.lower() == str(META_EVENT_TYPE).lower()
            ch_hb = ch2[m].copy()

        if ch_hb.empty:
            continue

        ts_col = pick_first_existing_col(ch_hb.columns.tolist(), HEARTBEAT_TS_CANDIDATES)
        if ts_col is None:
            continue

        x = pd.to_numeric(ch_hb[ts_col], errors="coerce")
        if x.dropna().empty:
            continue

        med_len = x.dropna().astype("int64").astype(str).str.len().median()
        unit    = "s" if (pd.notna(med_len) and med_len <= 10) else "ms"
        ch_hb["dt_hb"] = pd.to_datetime(x, unit=unit, errors="coerce")
        ch_hb = ch_hb.dropna(subset=["dt_hb"])
        if ch_hb.empty:
            continue
        if "tz_offset_minutes" in ch_hb.columns:
            tz_ms = pd.to_numeric(ch_hb["tz_offset_minutes"], errors="coerce").fillna(540) * 60 * 1000
            dt_local = pd.to_datetime(x.loc[ch_hb.index].astype("int64") + tz_ms.astype("int64"), unit="ms", errors="coerce")
            ch_hb["date"] = dt_local.dt.normalize()
        else:
            ch_hb["date"] = ch_hb["dt_hb"].dt.normalize()

        hb_cnt:         Dict = {}
        retry_max_d:    Dict = {}
        queue_max_d:    Dict = {}
        tz_min_d:       Dict = {}
        tz_max_d:       Dict = {}
        last_event_max: Dict = {}

        for d, sub in ch_hb.groupby("date"):
            d_idx = pd.Timestamp(d)
            hb_cnt[d_idx] = hb_cnt.get(d_idx, 0) + len(sub)

            def update_target(col_name, target_dict, kind: str):
                if col_name not in sub.columns:
                    return
                vals = pd.to_numeric(sub[col_name], errors="coerce").dropna()
                if vals.empty:
                    return
                new_val  = float(vals.max() if kind == "max" else vals.min())
                prev_val = target_dict.get(d_idx, np.nan)
                if not np.isfinite(prev_val):
                    target_dict[d_idx] = new_val
                else:
                    target_dict[d_idx] = (max if kind == "max" else min)(prev_val, new_val)

            update_target("retry_count",         retry_max_d,    "max")
            update_target("queue_size",           queue_max_d,    "max")
            update_target("tz_offset_minutes",    tz_min_d,       "min")
            update_target("tz_offset_minutes",    tz_max_d,       "max")
            update_target("client_last_event_ts", last_event_max, "max")

        for d in sorted(hb_cnt.keys()):
            tmin = tz_min_d.get(d, np.nan)
            tmax = tz_max_d.get(d, np.nan)
            tz_changed = False
            if pd.notna(tmin) and pd.notna(tmax):
                tz_changed = bool(abs(tmax - tmin) >= TZ_CHANGE_THR_MINUTES)
            out_rows.append({
                "uuid": uid, "date": d,
                "heartbeat_cnt":  int(hb_cnt[d]),
                "retry_max":      retry_max_d.get(d, np.nan),
                "queue_max":      queue_max_d.get(d, np.nan),
                "tz_offset_min":  tmin,
                "tz_offset_max":  tmax,
                "tz_changed":     tz_changed,
                "qc_last_ts_max": last_event_max.get(d, np.nan),
            })

    if not out_rows:
        return pd.DataFrame(columns=[
            "uuid", "date", "heartbeat_cnt", "retry_max", "queue_max",
            "tz_offset_min", "tz_offset_max", "tz_changed", "qc_last_ts_max",
        ])
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# 6. QC / context / delta  (v3_FE 그대로)
# ---------------------------------------------------------------------------

def add_qc_flags_v3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qc_core_low_cov"]           = df["daily_event_cnt"] < MIN_DAILY_EVENTS
    df["qc_core_very_low_activity"] = (df["daily_event_cnt"] > 0) & (df["daily_event_cnt"] < PARTIAL_CORE_MIN_EVENTS)

    for col in ["rhythm_low_coverage", "gap_low_coverage"]:
        if col not in df.columns:
            df[col] = True
        df[col] = df[col].fillna(True).astype(bool)
    df["qc_rhythm_low_cov"] = df["rhythm_low_coverage"]
    df["qc_gap_low_cov"]    = df["gap_low_coverage"]

    if "heartbeat_cnt" in df.columns:
        df["qc_meta_low_heartbeat"] = (
            pd.to_numeric(df["heartbeat_cnt"], errors="coerce").fillna(0).astype(int) < MIN_HEARTBEAT_PER_DAY
        )
    else:
        df["qc_meta_low_heartbeat"] = np.nan

    if "retry_max" in df.columns:
        df["qc_meta_retry_warn"] = pd.to_numeric(df["retry_max"], errors="coerce").fillna(-np.inf) >= RETRY_WARN_THR
    else:
        df["qc_meta_retry_warn"] = np.nan

    if "queue_max" in df.columns:
        df["qc_meta_queue_warn"] = pd.to_numeric(df["queue_max"], errors="coerce").fillna(-np.inf) >= QUEUE_WARN_THR
    else:
        df["qc_meta_queue_warn"] = np.nan

    if "tz_changed" not in df.columns and ("tz_offset_min" in df.columns) and ("tz_offset_max" in df.columns):
        tmin = pd.to_numeric(df["tz_offset_min"], errors="coerce")
        tmax = pd.to_numeric(df["tz_offset_max"], errors="coerce")
        df["tz_changed"] = (tmax - tmin).abs() >= TZ_CHANGE_THR_MINUTES
    return df


def add_soft_context_signals_v3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tz_changed" in df.columns:
        df["tz_change_signal"] = _to01_or_nan(df["tz_changed"].astype("boolean"))
    else:
        df["tz_change_signal"] = 0.0

    core_very_low = df.get("qc_core_very_low_activity", False).fillna(False).astype(bool)
    retry01 = _to01_or_nan(df.get("qc_meta_retry_warn", np.nan))
    queue01 = _to01_or_nan(df.get("qc_meta_queue_warn", np.nan))

    base = core_very_low.astype(int)
    df["partial_signal_raw"] = (
        base
        + pd.to_numeric(retry01, errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(queue01, errors="coerce").fillna(0).astype(int)
    ).astype(float)
    df.loc[~core_very_low, "partial_signal_raw"] = 0.0

    travel = pd.Series(0, index=df.index, dtype=float)
    tz01 = pd.to_numeric(df["tz_change_signal"].astype(object), errors="coerce").fillna(0).astype(int)
    travel += (tz01 * 2).astype(float)

    if "cell_change_cnt" in df.columns:
        c      = pd.to_numeric(df["cell_change_cnt"], errors="coerce")
        c_prev = c.groupby(df["uuid"]).shift(1)
        travel += ((c_prev.notna()) & (c >= (c_prev * 2 + 10))).astype(float)

    if "wifi_change_cnt_est" in df.columns:
        w      = pd.to_numeric(df["wifi_change_cnt_est"], errors="coerce")
        w_prev = w.groupby(df["uuid"]).shift(1)
        travel += ((w_prev.notna()) & (w >= (w_prev * 2 + 10))).astype(float)

    if "unique_cell_cnt" in df.columns:
        u1      = pd.to_numeric(df.get("unique_cell_cnt", np.nan), errors="coerce").fillna(0)
        u1_prev = u1.groupby(df["uuid"]).shift(1)
        travel += ((u1_prev.notna()) & (u1 >= (u1_prev + 3))).astype(float)

    df["travel_signal_raw"] = travel
    return df


def add_delta_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["uuid", "date"]).reset_index(drop=True)

    delta_cols = [
        c for c in [
            "daily_event_cnt", "night_ratio", "hour_entropy",
            "gap_max", "gap_p95", "gap_cnt_2h", "gap_cnt_6h",
            "session_total_sec", "session_cnt", "step_sum",
            "cell_change_cnt", "wifi_change_cnt_est",
            "travel_signal_raw", "partial_signal_raw", "tz_change_signal",
        ] if c in df.columns
    ]
    for c in delta_cols:
        x    = pd.to_numeric(df[c].astype(float), errors="coerce")
        prev = x.groupby(df["uuid"]).shift(1)
        df[f"{c}_d1"] = x - prev

    eps = 1e-6
    for c in [c for c in ["daily_event_cnt", "session_total_sec", "step_sum"] if c in df.columns]:
        x    = pd.to_numeric(df[c], errors="coerce")
        prev = x.groupby(df["uuid"]).shift(1)
        df[f"{c}_r1"] = (x - prev) / (prev.abs() + eps)

    return df


# ---------------------------------------------------------------------------
# 7. 메인 진입점
# ---------------------------------------------------------------------------

def run(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    FE Step 메인 함수.

    Parameters
    ----------
    raw_df : pd.DataFrame
        silver parquet에서 읽어온 raw 이벤트 DataFrame.
        필수 컬럼: uuid, event_name, timestamp

    Returns
    -------
    pd.DataFrame
        daily_feature_v3 — (uuid, date) 키의 일별 피처 테이블
    """
    print(f"[FE] input: {raw_df.shape}, users: {raw_df['uuid'].nunique()}")

    activity_df = build_daily_activity(raw_df)
    rhythm_df   = build_daily_rhythm(raw_df)
    gap_df      = build_daily_gap(raw_df)
    session_df  = build_daily_session(raw_df)
    mob_df      = build_daily_mobility(raw_df)
    meta_df     = build_daily_meta_qc(raw_df)

    for df_, name in [(activity_df, "activity"), (rhythm_df, "rhythm"), (gap_df, "gap"),
                      (session_df, "session"), (mob_df, "mobility"), (meta_df, "meta")]:
        _assert_or_dedup_uuid_date(df_, name)

    df = activity_df.copy()
    for right in [rhythm_df, gap_df, session_df, mob_df, meta_df]:
        df = _outer_merge_on_uuid_date(df, right)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["uuid", "date"]).sort_values(["uuid", "date"]).reset_index(drop=True)

    if "Unlock" not in df.columns:
        df["Unlock"] = 0
    df["Unlock"]     = pd.to_numeric(df["Unlock"], errors="coerce").fillna(0).astype(int)
    df["unlock_cnt"] = df["Unlock"]

    for c in CORE_SENSORS + PASSIVE_SENSORS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "unlock_cnt" in df.columns:
        df["unlock_cnt"] = pd.to_numeric(df["unlock_cnt"], errors="coerce").fillna(0).astype(int)

    core_cols_present = [c for c in CORE_SENSORS if c in df.columns]
    df["daily_event_cnt"] = df[core_cols_present].sum(axis=1).astype(int)
    df["has_activity"]    = (df["daily_event_cnt"] > 0)

    df = add_qc_flags_v3(df)
    df = add_soft_context_signals_v3(df)
    df = add_delta_features_v3(df)

    # meta 컬럼 없으면 NaN 보장
    for c in ["tz_changed", "tz_offset_min", "tz_offset_max",
              "heartbeat_cnt", "retry_max", "queue_max", "qc_last_ts_max"]:
        if c not in df.columns:
            df[c] = np.nan

    print(f"[FE] output: {df.shape}")
    return df
