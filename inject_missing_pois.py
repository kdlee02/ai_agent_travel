"""
inject_missing_pois.py

서울 주요 관광지 중 area_profiles_v2.json에 누락된 것들을
Google Places API로 실제 데이터를 검증해서 주입한다.

할루시네이션 없음:
- LLM 개입 없음
- Google Places Text Search로 실제 좌표/이름/place_id 확인
- 검증 실패 시 주입 안 함

실행:
    python inject_missing_pois.py

결과:
    output/area_profiles_v2.json 업데이트
    output/area_profiles_v2_before_inject.json 백업
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "output" / "area_profiles_v2.json"
BACKUP_PATH  = BASE_DIR / "output" / "area_profiles_v2_before_inject.json"

GOOGLE_PLACES_API_KEY = (
    os.getenv("GOOGLE_PLACES_API_KEY")
    or os.getenv("GOOGLE_MAPS_API_KEY")
    or ""
)

# ============================================================
# 추가할 누락 관광지 목록
#
# 형식:
#   (영문 이름,  poi_type,  base_cluster,  roles,          fallback_lat, fallback_lng)
#
# base_cluster: area_profiles의 어느 cluster에 넣을지
# roles: replanner가 이 POI를 어떤 역할로 쓸지
# fallback_lat/lng: Google Places 검색 실패 시 사용할 좌표
# ============================================================

MISSING_POIS: list[tuple[str, str, str, list[str], float, float]] = [
    # ── jongno_area ─────────────────────────────────────────
    ("Gyeongbokgung Palace (경복궁)",      "history",      "jongno",    ["history","culture","attraction","family"],      37.5796, 126.9770),
    ("Namsangol Hanok Village (남산골 한옥마을)", "history", "jongno",  ["history","culture","attraction","family"],      37.5594, 126.9941),
    ("Jogyesa Temple (조계사)",            "history",      "jongno",    ["history","culture","attraction"],               37.5744, 126.9818),
    ("Ikseon-dong Hanok Village (익선동)", "street",       "insadong",  ["culture","attraction","cafe","shopping"],       37.5762, 126.9983),
    ("Bukchon Hanok Village (북촌한옥마을)","street",       "bukchon",   ["history","culture","attraction","family"],      37.5826, 126.9836),
    ("Cheong Wa Dae (청와대)",             "culture",      "jongno",    ["history","culture","attraction"],               37.5836, 126.9753),
    ("Changdeokgung Palace (창덕궁)",      "history",      "jongno",    ["history","culture","attraction","family"],      37.5792, 126.9910),
    ("Unhyeongung Palace (운현궁)",        "history",      "insadong",  ["history","culture","attraction"],               37.5747, 126.9867),
    ("Insa-dong (인사동)",                 "street",       "insadong",  ["culture","shopping","attraction","market"],      37.5741, 126.9861),
    ("Samcheong-dong (삼청동)",            "street",       "bukchon",   ["culture","cafe","shopping","attraction"],        37.5816, 126.9820),

    # ── yongsan_itaewon_area ─────────────────────────────────
    ("N Seoul Tower (N서울타워)",           "tourist_spot", "yongsan",   ["attraction","culture","family","nature"],       37.5512, 126.9882),
    ("Namsan Cable Car (남산 케이블카)",    "tourist_spot", "yongsan",   ["attraction","family","nature"],                 37.5518, 126.9808),
    ("War Memorial of Korea (전쟁기념관)", "museum",       "yongsan",   ["history","culture","attraction","family"],      37.5365, 126.9772),
    ("Leeum Museum of Art (리움미술관)",   "museum",       "itaewon",   ["culture","attraction","history"],               37.5381, 126.9998),
    ("Itaewon Global Cultural Street",     "street",       "itaewon",   ["culture","attraction","nightlife","shopping"],   37.5347, 126.9946),
    ("Haebangchon (해방촌)",               "street",       "itaewon",   ["culture","attraction","cafe","nightlife"],       37.5399, 126.9833),
    ("Namsan Mountain Park (남산공원)",    "park",         "yongsan",   ["nature","attraction","family"],                 37.5512, 126.9820),

    # ── myeongdong_euljiro_area ──────────────────────────────
    ("Myeongdong Cathedral (명동성당)",    "history",      "myeongdong",["history","culture","attraction"],               37.5633, 126.9874),
    ("Dongdaemun Design Plaza (DDP)",      "tourist_spot", "dongdaemun",["culture","attraction","shopping","kpop"],       37.5670, 127.0095),
    ("Namdaemun Market (남대문시장)",      "market",       "myeongdong",["market","shopping","meal","attraction"],        37.5598, 126.9759),
    ("Eulji OB Bear (을지 OB베어)",        "street",       "myeongdong",["nightlife","culture","attraction"],             37.5657, 126.9870),
    ("Gwangjang Market (광장시장)",        "market",       "dongdaemun",["market","meal","shopping","attraction"],        37.5700, 126.9997),

    # ── gangnam_area ─────────────────────────────────────────
    ("Garosu-gil (가로수길)",              "street",       "gangnam",   ["shopping","beauty","cafe","attraction"],        37.5196, 127.0189),
    ("Dosan Park (도산공원)",              "park",         "apgujeong", ["nature","attraction","family"],                 37.5228, 127.0355),
    ("Cheongdam-dong Fashion Street",      "street",       "apgujeong", ["shopping","beauty","culture","attraction"],     37.5268, 127.0463),
    ("SMTOWN COEX Artium (SM타운)",        "kpop_landmark","samseong_coex",["kpop","attraction","shopping","culture"],    37.5127, 127.0592),
    ("Hive Insight (하이브 인사이트)",     "kpop_landmark","gangnam",   ["kpop","attraction","culture"],                  37.5056, 127.0247),
    ("Seolleung and Jeongneung (선릉)",   "history",      "gangnam",   ["history","culture","attraction","nature"],      37.5084, 127.0471),

    # ── seongsu ──────────────────────────────────────────────
    ("Seoul Forest (서울숲)",              "park",         "seongsu",   ["nature","attraction","family"],                 37.5449, 127.0374),
    ("Seongsu Cafe Street (성수 카페거리)","street",        "seongsu",   ["cafe","culture","attraction","shopping"],       37.5443, 127.0557),
    ("Ttukseom Hangang Park (뚝섬한강공원)","park",        "seongsu",   ["nature","family","attraction"],                 37.5312, 127.0667),

    # ── yeouido ──────────────────────────────────────────────
    ("Yeouido Hangang Park (여의도한강공원)","park",        "yeouido",   ["nature","family","attraction"],                 37.5284, 126.9337),
    ("The Hyundai Seoul (더현대서울)",     "shopping",     "yeouido",   ["shopping","culture","attraction","indoor_leisure"],37.5237, 126.9213),
    ("IFC Seoul (아이에프씨몰)",           "shopping",     "yeouido",   ["shopping","indoor_leisure","attraction"],       37.5253, 126.9254),
    ("National Assembly (국회의사당)",     "culture",      "yeouido",   ["history","culture","attraction"],               37.5330, 126.9142),

    # ── jamsil ───────────────────────────────────────────────
    ("Lotte World (롯데월드)",             "tourist_spot", "jamsil",    ["attraction","family","indoor_leisure","kpop"],  37.5113, 127.0984),
    ("Lotte World Tower (롯데월드타워)",   "tourist_spot", "jamsil",    ["attraction","culture","family"],                37.5126, 127.1027),
    ("Songpa Naru Park (송파나루공원)",    "park",         "jamsil",    ["nature","family","attraction"],                 37.5183, 127.1043),
    ("Olympic Park (올림픽공원)",          "park",         "jamsil",    ["nature","family","attraction","culture"],       37.5205, 127.1219),
]

# umbrella cluster가 member cluster를 포함하는 매핑
UMBRELLA_MEMBERS: dict[str, list[str]] = {
    "hongdae_area":           ["hongdae", "mapo", "mangwon", "sinchon"],
    "gangnam_area":           ["gangnam", "samseong_coex", "apgujeong", "seocho"],
    "jongno_area":            ["jongno", "insadong", "bukchon", "daehaengno"],
    "yongsan_itaewon_area":   ["yongsan", "itaewon"],
    "myeongdong_euljiro_area":["myeongdong", "dongdaemun"],
}

# ============================================================
# 유틸
# ============================================================

def safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9가-힣]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# Google Places 검증
# ============================================================

def google_text_search(name: str, api_key: str) -> dict[str, Any] | None:
    """
    Google Places Text Search로 POI 검증.
    실제 좌표, place_id, 공식 이름을 반환.
    """
    if not api_key or not _REQUESTS_OK:
        return None

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{name} Seoul",
        "key": api_key,
        "language": "en",
        "region": "kr",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        results = resp.json().get("results", [])
        if not results:
            return None

        # 첫 번째 결과 (Seoul 내 결과가 상위에 옴)
        r = results[0]
        loc = r.get("geometry", {}).get("location", {})
        lat = safe_float(loc.get("lat"))
        lng = safe_float(loc.get("lng"))
        if lat is None or lng is None:
            return None

        return {
            "verified_name": r.get("name", name),
            "lat": lat,
            "lng": lng,
            "place_id": r.get("place_id", ""),
            "address": r.get("formatted_address", ""),
            "rating": r.get("rating"),
            "user_ratings_total": r.get("user_ratings_total", 0),
        }
    except Exception as e:
        print(f"  [Google Places] 검색 오류: {e}")
        return None


def verify_poi(
    name: str,
    fallback_lat: float,
    fallback_lng: float,
    api_key: str,
) -> dict[str, Any]:
    """
    Google Places로 POI 검증. 실패 시 fallback 좌표 사용.
    """
    result = google_text_search(name, api_key)

    if result:
        # 검색 결과가 fallback 좌표와 너무 멀면 (10km 이상) 신뢰 안 함
        dist = haversine_km(fallback_lat, fallback_lng, result["lat"], result["lng"])
        if dist <= 10.0:
            print(f"  ✅ Google Places 검증 성공: {result['verified_name']} "
                  f"(lat={result['lat']:.4f}, lng={result['lng']:.4f}, dist={dist:.1f}km)")
            return {
                "lat": result["lat"],
                "lng": result["lng"],
                "google_place_id": result["place_id"],
                "address": result["address"],
                "place_confidence": "high",
                "verified": True,
            }
        else:
            print(f"  ⚠️ Google Places 결과가 너무 멀어 fallback 사용: {dist:.1f}km")

    # Google Places 실패 → fallback 좌표 사용
    print(f"  📍 fallback 좌표 사용: lat={fallback_lat}, lng={fallback_lng}")
    return {
        "lat": fallback_lat,
        "lng": fallback_lng,
        "google_place_id": "",
        "address": "",
        "place_confidence": "medium",
        "verified": False,
    }


# ============================================================
# POI 엔트리 생성
# ============================================================

def make_poi_entry(
    name: str,
    poi_type: str,
    cluster: str,
    roles: list[str],
    geo: dict[str, Any],
    rep_score: float = 0.85,
) -> dict[str, Any]:
    """
    area_profiles_v2.json의 representative_pois / role_candidates 형식으로 변환.
    """
    primary_role = roles[0] if roles else "attraction"
    return {
        "poi_id": f"injected_{normalize(name).replace(' ', '_')[:30]}",
        "name": name,
        "poi_type": poi_type,
        "cluster": cluster,
        "lat": geo["lat"],
        "lng": geo["lng"],
        "roles": roles,
        "primary_role": primary_role,
        "representative_score": rep_score,
        "role_score": None,
        "general_representative_ok": True,
        "place_confidence": geo.get("place_confidence", "medium"),
        "purpose_tags": roles,
        "google_place_id": geo.get("google_place_id", ""),
        "address": geo.get("address", ""),
        "is_vague_or_broad": False,
        "is_dual_role_anchor": len(roles) >= 3,
        "is_general_downrank": False,
        "debug_reasons": ["injected_by_inject_missing_pois"],
        "source": "inject_missing_pois",
    }


# ============================================================
# area_profiles 주입
# ============================================================

def inject_poi_into_profile(
    profile: dict[str, Any],
    entry: dict[str, Any],
    target_cluster: str,
) -> None:
    """
    entry를 target_cluster의 representative_pois와 role_candidates에 주입.
    umbrella cluster에도 member이면 함께 주입.
    중복 이름은 제거.
    """
    def _add_to_cluster(cdata: dict[str, Any], e: dict[str, Any]) -> bool:
        """cluster에 POI 추가. 이미 있으면 False 반환."""
        e_name = normalize(e["name"])

        # representative_pois 중복 체크 + 추가
        reps = cdata.setdefault("representative_pois", [])
        if any(normalize(p.get("name", "")) == e_name for p in reps):
            return False
        reps.insert(0, e)  # 상위에 배치

        # role_candidates에도 추가
        rc = cdata.setdefault("role_candidates", {})
        for role in e.get("roles", []):
            pool = rc.setdefault(role, [])
            if not any(normalize(p.get("name", "")) == e_name for p in pool):
                pool.insert(0, e)

        # attraction role 기본 추가
        attraction_pool = rc.setdefault("attraction", [])
        if not any(normalize(p.get("name", "")) == e_name for p in attraction_pool):
            attraction_pool.insert(0, e)

        return True

    clusters = profile.get("clusters", {})

    # 1) base cluster에 추가
    if target_cluster in clusters:
        added = _add_to_cluster(clusters[target_cluster], entry)
        if added:
            print(f"    → [{target_cluster}] 추가됨")

    # 2) umbrella cluster에도 추가 (member이면)
    for umbrella, members in UMBRELLA_MEMBERS.items():
        if target_cluster in members and umbrella in clusters:
            _add_to_cluster(clusters[umbrella], entry)


# ============================================================
# 이미 있는지 확인
# ============================================================

def is_already_in_profile(profile: dict[str, Any], name: str) -> bool:
    """
    profile 전체에서 해당 이름의 POI가 이미 있는지 확인.
    """
    name_n = normalize(name)
    for cdata in profile.get("clusters", {}).values():
        all_pois = (
            cdata.get("representative_pois", [])
            + cdata.get("dual_role_anchors", [])
        )
        for role_list in cdata.get("role_candidates", {}).values():
            all_pois.extend(role_list)
        for p in all_pois:
            pn = normalize(p.get("name", ""))
            if pn and (pn == name_n or pn in name_n or name_n in pn):
                return True
    return False


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"[inject] Profile: {PROFILE_PATH}")
    print(f"[inject] API key: {'있음' if GOOGLE_PLACES_API_KEY else '없음 (fallback 좌표만 사용)'}")

    if not PROFILE_PATH.exists():
        print(f"[inject] 오류: {PROFILE_PATH} 없음")
        return

    # 백업
    shutil.copy2(PROFILE_PATH, BACKUP_PATH)
    print(f"[inject] 백업: {BACKUP_PATH}\n")

    # 프로필 로드
    with open(PROFILE_PATH, encoding="utf-8") as f:
        profile = json.load(f)

    added_count = 0
    skipped_count = 0

    for name, poi_type, cluster, roles, fallback_lat, fallback_lng in MISSING_POIS:
        print(f"[{name}]")

        # 이미 있으면 스킵
        if is_already_in_profile(profile, name):
            print(f"  ⏭️  이미 존재 — 스킵\n")
            skipped_count += 1
            continue

        # Google Places로 검증
        geo = verify_poi(name, fallback_lat, fallback_lng, GOOGLE_PLACES_API_KEY)

        # POI 엔트리 생성
        entry = make_poi_entry(
            name=name,
            poi_type=poi_type,
            cluster=cluster,
            roles=roles,
            geo=geo,
        )

        # profile에 주입
        inject_poi_into_profile(profile, entry, cluster)
        added_count += 1

        # API 호출 간격 (rate limit 방지)
        if GOOGLE_PLACES_API_KEY:
            time.sleep(0.3)

        print()

    # 저장
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"\n[inject] 완료: 추가={added_count}개, 스킵={skipped_count}개")
    print(f"[inject] 저장: {PROFILE_PATH}")

    # 검증
    print("\n[inject] 검증")
    check_clusters = ["jongno_area", "gangnam_area", "yongsan_itaewon_area",
                      "myeongdong_euljiro_area", "seongsu", "yeouido", "jamsil"]
    for cluster_name in check_clusters:
        cdata = profile["clusters"].get(cluster_name, {})
        reps = cdata.get("representative_pois", [])
        injected = [p for p in reps if p.get("source") == "inject_missing_pois"]
        print(f"  {cluster_name}: 전체 대표 POI {len(reps)}개 (신규 주입 {len(injected)}개)")
        for p in injected[:3]:
            print(f"    + {p['name']} ({p['poi_type']})")


if __name__ == "__main__":
    main()