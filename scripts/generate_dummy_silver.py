"""
scripts/generate_dummy_silver.py
=================================
테스트용 silver parquet 한 달치(31일) 생성 + FE → daily-feature S3 저장.

[사용자 그룹 - 원래 50명 UUID 사용]
  - 일반 (35명): 평범한 활동 패턴 → STABLE/CHAOS/SLEEP/LETHARGY
  - NO_DATA (10명): heartbeat만 → NO_DATA
  - TRAVEL (5명): tz 변화 + wifi/cell 변화 多 → TRAVEL

[실행]
  python scripts/generate_dummy_silver.py \
    --s3-silver-uri s3://silver-dummy/silver_events/ \
    --s3-daily-feature-uri s3://nyang-ml-apne2-dev/ml/daily-feature/ \
    --source-parquet notebooks/part-0000.parquet \
    --start-date 2026-01-20 \
    --end-date 2026-02-19
"""

import argparse
import io
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

_root = Path(globals().get("__file__", ".")).resolve().parent
if _root.name == "scripts":
    _root = _root.parent
sys.path.insert(0, str(_root))
from src.steps import features

KST = timezone(timedelta(hours=9))
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# S3 헬퍼
# ---------------------------------------------------------------------------

def _s3():
    return boto3.client("s3")


def _parse_s3_uri(s3_uri: str):
    parts  = s3_uri.rstrip("/").replace("s3://", "").split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _upload_parquet(df: pd.DataFrame, bucket: str, key: str):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"    → s3://{bucket}/{key} ({len(df)}행)")


# ---------------------------------------------------------------------------
# 이벤트 생성 헬퍼
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    "SCREEN_INTERACTIVE", "SCREEN_NON_INTERACTIVE",
    "USER_INTERACTION", "KEYGUARD_HIDDEN",
    "NOTIF_INTERRUPTION", "WIFI_SSID",
    "CELL_CHANGE",
]


def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _base_row(uuid, event_name, ts, tz_offset=540, **kwargs) -> dict:
    return {
        "uuid": uuid, "event_name": event_name, "timestamp": ts,
        "type": kwargs.get("type", "system_event"),
        "wifi_ssid": kwargs.get("wifi_ssid", None),
        "prev_wifi_bssid": kwargs.get("prev_wifi_bssid", None),
        "wifi_change_type": kwargs.get("wifi_change_type", None),
        "step_count": kwargs.get("step_count", 0),
        "cell_lac": kwargs.get("cell_lac", None),
        "queue_size": kwargs.get("queue_size", 0.0),
        "retry_count": kwargs.get("retry_count", 0.0),
        "tz_offset_minutes": tz_offset,
        "client_last_event_ts": ts,
    }


def make_normal_events(uuid: str, date: datetime, n: int = 150, chaos: bool = False) -> list:
    """chaos=True: 오전에 몰아주고 오후~밤에 긴 gap → CHAOS 유발"""
    rows = [_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=0, minute=30)),
                      type="heartbeat")]
    for _ in range(n):
        if chaos:
            # 오전 6~11시에 집중, 오후 이후 거의 없음 → gap_max 12h+
            hour = random.choices(range(24), weights=[
                0,0,0,0,0,0,8,10,10,10,8,6,2,1,1,1,1,1,0,0,0,0,0,0
            ])[0]
        else:
            hour = random.choices(range(24), weights=[
                1,1,1,1,1,1,2,4,6,8,8,7,6,7,8,8,7,6,5,4,3,2,2,1
            ])[0]
        ts  = _ts_ms(date.replace(hour=hour, minute=random.randint(0,59),
                                  second=random.randint(0,59)))
        evt = random.choice(EVENT_TYPES)
        rows.append(_base_row(
            uuid, evt, ts,
            wifi_ssid=f"ssid-{random.randint(1,5):02d}" if evt=="WIFI_SSID" else None,
            prev_wifi_bssid=f"bssid-{random.randint(1,10):010d}" if evt=="WIFI_SSID" else None,
            wifi_change_type="wifi_disconnected" if evt=="WIFI_SSID" else None,
            step_count=random.randint(0, 200),
            queue_size=float(random.randint(0, 5)),
            retry_count=float(random.randint(0, 1)),
            cell_lac=float(random.randint(10000, 99999)) if evt=="CELL_CHANGE" else None,
        ))
    return rows


def make_nodata_events(uuid: str, date: datetime) -> list:
    """heartbeat만 → has_activity=False → NO_DATA"""
    return [_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=0, minute=30)),
                      type="heartbeat")]


def make_travel_events(uuid: str, date: datetime, day_idx: int) -> list:
    """
    10~20일차: 해외(tz=0)
    - WIFI_SSID 이벤트 매우 많이 (30개+) → wifi_change_cnt_est 높음
    - cell_lac 매번 다르게 → cell_change_cnt 높음
    - tz 변화 → tz_change_signal 높음
    → travel_signal_raw >= 2.0 달성
    """
    tz     = 0 if 9 <= day_idx <= 19 else 540
    abroad = (tz == 0)
    rows   = [_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=0, minute=30)),
                        tz_offset=tz, type="heartbeat")]

    # tz 전환일: 같은 "date" 안에 서로 다른 tz_offset heartbeat 2개를 넣어서
    #           tz_offset_min != tz_offset_max 가 확실히 잡히게 함
    if day_idx == 9:
        # 오전: KST
        rows.append(_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=1, minute=0)),
                          tz_offset=540, type="heartbeat"))
        # 점심: UTC(해외)
        rows.append(_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=12, minute=0)),
                          tz_offset=0, type="heartbeat"))

    elif day_idx == 19:
        # 오전: UTC(해외)
        rows.append(_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=1, minute=0)),
                          tz_offset=0, type="heartbeat"))
        # 점심: KST(귀국)
        rows.append(_base_row(uuid, "HEARTBEAT", _ts_ms(date.replace(hour=12, minute=0)),
                          tz_offset=540, type="heartbeat"))
    
    # 해외일 때: WIFI_SSID + CELL 이벤트 집중 추가 (travel_signal 올리기)
    if abroad:
        for i in range(40):
            hour = random.randint(6, 22)
            ts   = _ts_ms(date.replace(hour=hour, minute=random.randint(0,59),
                                   second=random.randint(0,59)))

            # WIFI 이벤트
            rows.append(_base_row(
                uuid, "WIFI_SSID", ts, tz_offset=tz,
                wifi_ssid=f"ssid-abroad-{random.randint(1,50):02d}",
                prev_wifi_bssid=f"bssid-{random.randint(1,200):010d}",
                wifi_change_type="wifi_disconnected",
                step_count=random.randint(100, 500),
                cell_lac=float(random.randint(10000, 99999)),
            ))

            # CELL_CHANGE 이벤트 (FE가 event_name count 기반이면 이게 결정타)
            rows.append(_base_row(
                uuid, "CELL_CHANGE", ts + 500, tz_offset=tz,
                cell_lac=float(random.randint(10000, 99999)),
                step_count=random.randint(100, 500),
            ))

    n = random.randint(120, 160)
    for _ in range(n):
        hour = random.randint(6, 23)
        ts   = _ts_ms(date.replace(hour=hour, minute=random.randint(0,59),
                                   second=random.randint(0,59)))
        evt  = random.choices(EVENT_TYPES, weights=[4,4,4,3,3,2,2])[0]
        rows.append(_base_row(
            uuid, evt, ts, tz_offset=tz,
            wifi_ssid=f"ssid-abroad-{random.randint(1,20):02d}" if (evt=="WIFI_SSID" and abroad)
                      else (f"ssid-{random.randint(1,5):02d}" if evt=="WIFI_SSID" else None),
            prev_wifi_bssid=f"bssid-{random.randint(1,50):010d}" if evt=="WIFI_SSID" else None,
            wifi_change_type="wifi_disconnected" if evt=="WIFI_SSID" else None,
            step_count=random.randint(100, 500) if abroad else random.randint(0, 200),
            cell_lac=float(random.randint(10000, 99999)) if abroad else None,
            queue_size=float(random.randint(0, 5)),
            retry_count=float(random.randint(0, 1)),
        ))
    return rows

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(args):
    # 원본 parquet에서 UUID 로드 (S3 or 로컬)
    if args.source_parquet.startswith("s3://"):
        bucket, key = _parse_s3_uri(args.source_parquet)
        obj = _s3().get_object(Bucket=bucket, Key=key)
        src_df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    else:
        src_df = pd.read_parquet(args.source_parquet)
    all_uuids = src_df["uuid"].unique().tolist()
    print(f"[GEN] UUID {len(all_uuids)}명 로드 완료")

    # 그룹 분배: 35 / 10 / 5
    random.shuffle(all_uuids)
    uuids_normal = all_uuids[:35]
    uuids_nodata = all_uuids[35:45]
    uuids_travel = all_uuids[45:50]

    print(f"[GEN] UUID 그룹 분배 완료")
    print(f"  일반: {len(uuids_normal)}명")
    print(f"  NO_DATA: {len(uuids_nodata)}명")
    print(f"  TRAVEL: {len(uuids_travel)}명")
    print()

    s_bucket, s_prefix = _parse_s3_uri(args.s3_silver_uri)
    f_bucket, f_prefix = _parse_s3_uri(args.s3_daily_feature_uri)

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=KST)
    end_dt   = datetime.strptime(args.end_date,   "%Y-%m-%d").replace(tzinfo=KST)

    current = start_dt
    day_idx = 0
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        print(f"[GEN] {date_str} ({day_idx+1}일차)")

        # 1. silver raw 생성
        rows = []
        for uuid in uuids_normal:
            rows.extend(make_normal_events(uuid, current, random.randint(130, 170)))
        for uuid in uuids_nodata:
            rows.extend(make_nodata_events(uuid, current))
        for uuid in uuids_travel:
            rows.extend(make_travel_events(uuid, current, day_idx))

        silver_df = pd.DataFrame(rows)
        silver_df["timestamp"]            = silver_df["timestamp"].astype("int64")
        silver_df["client_last_event_ts"] = silver_df["client_last_event_ts"].astype("int64")
        silver_df["step_count"]           = silver_df["step_count"].astype("int32")
        silver_df["tz_offset_minutes"]    = silver_df["tz_offset_minutes"].astype("int32")

        # 2. silver S3 저장
        s_key = f"{s_prefix}/dt={date_str}/part-0000.parquet".lstrip("/")
        print(f"  silver 저장 중...")
        _upload_parquet(silver_df, s_bucket, s_key)

        # 3. FE 실행
        print(f"  FE 실행 중...")
        feat_df = features.run(silver_df)

        # 4. daily-feature S3 저장
        f_key = f"{f_prefix}/dt={date_str}/daily-feature.parquet".lstrip("/")
        print(f"  daily-feature 저장 중...")
        _upload_parquet(feat_df, f_bucket, f_key)

        current += timedelta(days=1)
        day_idx += 1

    print(f"\n[GEN] 완료 ✅  {args.start_date} ~ {args.end_date} ({day_idx}일치)")


def parse_args():
    parser = argparse.ArgumentParser(description="테스트용 silver + daily-feature 한 달치 생성")
    parser.add_argument("--source-parquet", type=str,
                        default="s3://silver-dummy/silver_events/dt=2026-02-13/part-0000.parquet",
                        help="UUID 추출용 원본 parquet 경로")
    parser.add_argument("--s3-silver-uri", type=str,
                        default="s3://silver-dummy/silver_events/")
    parser.add_argument("--s3-daily-feature-uri", type=str,
                        default="s3://nyang-ml-apne2-dev/ml/daily-feature/")
    parser.add_argument("--start-date", type=str, default="2026-01-20")
    parser.add_argument("--end-date",   type=str, default="2026-02-19")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())