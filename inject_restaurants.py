"""
inject_restaurants.py

poi_master_step3_enhanced_v2.csv의 식당/카페를
area_profiles_v2.json의 각 cluster의 meal/cafe role_candidates에 주입한다.

실행:
    python inject_restaurants.py

결과:
    output/area_profiles_v2.json 업데이트 (백업: output/area_profiles_v2_backup.json)
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "output" / "poi_master_step3_enhanced_v2.csv"
PROFILE_PATH = BASE_DIR / "output" / "area_profiles_v2.json"
BACKUP_PATH = BASE_DIR / "output" / "area_profiles_v2_backup.json"

# cluster 중심 좌표 (area_profile_builder_v2.py와 동일)
SEOUL_CLUSTERS: dict[str, tuple[float, float]] = {
    "hongdae":      (37.5563, 126.9227),
    "mapo":         (37.5479, 126.9130),
    "mangwon":      (37.5567, 126.9055),
    "sinchon":      (37.5596, 126.9373),
    "gangnam":      (37.4979, 127.0276),
    "samseong_coex":(37.5118, 127.0592),
    "seongsu":      (37.5447, 127.0558),
    "jongno":       (37.5729, 126.9794),
    "insadong":     (37.5741, 126.9861),
    "myeongdong":   (37.5636, 126.9857),
    "itaewon":      (37.5347, 126.9946),
    "yongsan":      (37.5299, 126.9649),
    "dongdaemun":   (37.5666, 127.0097),
    "yeouido":      (37.5217, 126.9244),
    "jamsil":       (37.5133, 127.1028),
    "bukchon":      (37.5826, 126.9836),
    "daehaengno":   (37.5810, 127.0020),
    "seocho":       (37.4837, 127.0324),
    "apgujeong":    (37.5271, 127.0286),
}

# umbrella → member clusters 매핑
UMBRELLA_MEMBERS: dict[str, set[str]] = {
    "hongdae_area":           {"hongdae", "mapo", "mangwon", "sinchon"},
    "gangnam_area":           {"gangnam", "samseong_coex", "apgujeong", "seocho"},
    "jongno_area":            {"jongno", "insadong", "bukchon", "daehaengno"},
    "yongsan_itaewon_area":   {"yongsan", "itaewon"},
    "myeongdong_euljiro_area":{"myeongdong", "dongdaemun"},
}

CLUSTER_RADIUS_KM = 3.8  # 클러스터 할당 반경


# ============================================================
# 유틸
# ============================================================

def safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def clean_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s.lower() not in {"nan", "none", "null", ""} else default


def normalize_name(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^0-9a-z가-힣]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def assign_cluster(lat: float | None, lng: float | None) -> str | None:
    if lat is None or lng is None:
        return None
    best_name, best_dist = None, float("inf")
    for name, (clat, clng) in SEOUL_CLUSTERS.items():
        d = haversine_km(lat, lng, clat, clng)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name if best_dist <= CLUSTER_RADIUS_KM else None


def dedupe_by_name(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for p in pool:
        key = normalize_name(p.get("name", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ============================================================
# CSV 로드 및 권역별 분류
# ============================================================

def load_restaurants_from_csv(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    CSV에서 restaurant/food/cafe 타입 POI를 읽어
    cluster별로 분류해서 반환.
    """
    if not csv_path.exists():
        alt = csv_path.parent / "poi_master_step3.csv"
        if alt.exists():
            csv_path = alt
        else:
            print(f"[inject] CSV 파일 없음: {csv_path}")
            return {}

    cluster_map: dict[str, list[dict[str, Any]]] = {}
    total, skipped = 0, 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptype = clean_str(row.get("poi_type", "")).lower()
            if ptype not in {"restaurant", "food", "cafe"}:
                continue

            total += 1
            name = clean_str(row.get("poi_name") or row.get("name", ""))
            if not name:
                skipped += 1
                continue

            lat = safe_float(row.get("lat"))
            lng = safe_float(row.get("lng"))
            cluster = assign_cluster(lat, lng)

            if not cluster:
                skipped += 1
                continue

            role = "meal" if ptype in {"restaurant", "food"} else "cafe"
            entry = {
                "name": name,
                "poi_type": ptype,
                "lat": lat,
                "lng": lng,
                "cluster": cluster,
                "roles": [role],
                "role_scores": {role: 0.8},
                "representative_score": 0.0,
                "general_representative_ok": True,
                "is_vague_or_broad": False,
                "is_dual_role_anchor": False,
                "google_place_id": clean_str(row.get("google_place_id")),
                "address": clean_str(row.get("address_en") or row.get("address_ko")),
                "source": "poi_master_csv",
            }

            cluster_map.setdefault(cluster, []).append(entry)

    print(f"[inject] CSV 읽기 완료: {total}개 식당/카페, {skipped}개 스킵")
    for cluster, pois in sorted(cluster_map.items()):
        print(f"  {cluster}: {len(pois)}개")

    return cluster_map


# ============================================================
# area_profiles_v2.json 업데이트
# ============================================================

def inject_into_profile(
    profile: dict[str, Any],
    cluster_map: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    cluster_map의 식당/카페를 profile의 각 cluster에 주입.
    umbrella cluster에도 member의 식당을 합산해서 주입.
    """
    stats: dict[str, int] = {}

    # 1단계: base cluster에 직접 주입
    for cluster_name, new_pois in cluster_map.items():
        cdata = profile["clusters"].get(cluster_name)
        if not cdata:
            continue

        role_candidates = cdata.setdefault("role_candidates", {})

        meal_pois = [p for p in new_pois if "meal" in p.get("roles", [])]
        cafe_pois = [p for p in new_pois if "cafe" in p.get("roles", [])]

        # 기존 pool과 합치되 중복 제거
        existing_meals = role_candidates.get("meal", [])
        existing_cafes = role_candidates.get("cafe", [])

        merged_meals = dedupe_by_name(existing_meals + meal_pois)
        merged_cafes = dedupe_by_name(existing_cafes + cafe_pois)

        added_meals = len(merged_meals) - len(existing_meals)
        added_cafes = len(merged_cafes) - len(existing_cafes)

        role_candidates["meal"] = merged_meals
        role_candidates["cafe"] = merged_cafes

        stats[cluster_name] = added_meals + added_cafes
        if added_meals + added_cafes > 0:
            print(f"  [{cluster_name}] meal+{added_meals}, cafe+{added_cafes}")

    # 2단계: umbrella cluster에 member의 식당 합산 주입
    for umbrella_name, members in UMBRELLA_MEMBERS.items():
        cdata = profile["clusters"].get(umbrella_name)
        if not cdata:
            continue

        role_candidates = cdata.setdefault("role_candidates", {})

        all_meals: list[dict[str, Any]] = []
        all_cafes: list[dict[str, Any]] = []

        for member in members:
            member_pois = cluster_map.get(member, [])
            all_meals.extend(p for p in member_pois if "meal" in p.get("roles", []))
            all_cafes.extend(p for p in member_pois if "cafe" in p.get("roles", []))

        existing_meals = role_candidates.get("meal", [])
        existing_cafes = role_candidates.get("cafe", [])

        merged_meals = dedupe_by_name(existing_meals + all_meals)
        merged_cafes = dedupe_by_name(existing_cafes + all_cafes)

        added_meals = len(merged_meals) - len(existing_meals)
        added_cafes = len(merged_cafes) - len(existing_cafes)

        role_candidates["meal"] = merged_meals
        role_candidates["cafe"] = merged_cafes

        stats[umbrella_name] = added_meals + added_cafes
        if added_meals + added_cafes > 0:
            print(f"  [{umbrella_name}] meal+{added_meals}, cafe+{added_cafes} (umbrella합산)")

    return profile, stats


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"[inject] CSV: {CSV_PATH}")
    print(f"[inject] Profile: {PROFILE_PATH}")

    if not PROFILE_PATH.exists():
        print(f"[inject] 오류: area_profiles_v2.json 없음")
        return

    # 백업
    shutil.copy2(PROFILE_PATH, BACKUP_PATH)
    print(f"[inject] 백업 완료: {BACKUP_PATH}")

    # 프로필 로드
    with open(PROFILE_PATH, encoding="utf-8") as f:
        profile = json.load(f)

    # CSV에서 식당/카페 로드
    cluster_map = load_restaurants_from_csv(CSV_PATH)
    if not cluster_map:
        print("[inject] 주입할 식당/카페 없음. 종료.")
        return

    # 주입
    print("\n[inject] 주입 중...")
    profile, stats = inject_into_profile(profile, cluster_map)

    # 저장
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    total_added = sum(stats.values())
    print(f"\n[inject] 완료: 총 {total_added}개 추가")
    print(f"[inject] 저장: {PROFILE_PATH}")

    # 결과 검증
    print("\n[inject] 검증")
    for cluster_name in ["hongdae_area", "gangnam_area", "jongno_area"]:
        cdata = profile["clusters"].get(cluster_name, {})
        meals = cdata.get("role_candidates", {}).get("meal", [])
        cafes = cdata.get("role_candidates", {}).get("cafe", [])
        print(f"  {cluster_name}: meal={len(meals)}개, cafe={len(cafes)}개")
        for p in meals[:3]:
            print(f"    meal: {p.get('name')} ({p.get('poi_type')})")
        for p in cafes[:2]:
            print(f"    cafe: {p.get('name')} ({p.get('poi_type')})")


if __name__ == "__main__":
    main()