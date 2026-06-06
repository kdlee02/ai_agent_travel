"""
replanner_v3_1.py

area_profiles_v2.json 기반 SeoulMate Day-level Replanner v3.1

v3.1 핵심 수정 (v3 대비)
-------------------------
1. representative_pois를 진짜 trusted pool로 처리
   - _compatible_clean_pool()에서 trusted_representatives=True일 때
     candidate_is_bad_for_purpose()를 적용하지 않음
   - 대신 is_vague_or_broad 플래그 + vague/broad 이름 체크 + area 호환성만 검사

2. candidate_is_vague_or_broad_name() 수정
   - BROAD_AREA_EXACT_ALIASES 부분 매치 체크를 representative에 적용하지 않음
   - "경의선 숲길", "연남동 카페거리", "망원시장" 같은 구체적 이름이
     "연남동", "망원동" 부분 매치로 탈락하는 문제 해결
   - 별도 함수 candidate_is_vague_or_broad_name_strict()로 분리

3. candidate_is_bad_for_purpose()의 "역" 패턴 수정
   - 단순 "역" 포함 여부가 아니라 "역" 단독 토큰 or "역" 접미사(지하철역/버스역) 패턴으로 강화
   - 무역센터(COEX), 경의선 같은 이름이 잘못 걸리는 오탐 제거

4. lunch slot을 reorder_tail_by_distance 후에도 두 번째 자리로 고정
   - reorder_tail_by_distance()가 selected[:2]를 fixed로 유지하더라도
     _attach_schedule()에서 2번째 인덱스 후보가 meal-like인지 재확인하고
     meal-like가 없으면 lunch를 다시 두 번째 자리로 이동시킴

실행 예시
---------
python replanner_v3_1.py --output output/replanned_itinerary_v3_1.json
python critic_v1_1.py --input output/replanned_itinerary_v3_1.json --output output/critic_result_v3_1.json --purpose general
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Google Places 직접 호출 (meals pool 비었을 때 fallback)
try:
    from google_places_fetcher import fetch_restaurants_for_area, parse_dietary_restrictions
    _GOOGLE_PLACES_AVAILABLE = True
except ImportError:
    _GOOGLE_PLACES_AVAILABLE = False
    def fetch_restaurants_for_area(*a, **kw): return []
    def parse_dietary_restrictions(d): return []

def _get_google_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    except Exception:
        pass
    return (
        os.getenv("GOOGLE_PLACES_API_KEY")
        or os.getenv("GOOGLE_MAPS_API_KEY")
        or ""
    )

GOOGLE_PLACES_API_KEY = _get_google_api_key()


# ============================================================
# 정책값
# ============================================================

STRUCTURAL_ISSUE_TYPES = {
    "off_theme_cluster_poi",
    "theme_mismatch",
    "cluster_scattered",
    "vague_poi",
    "area_anchor_needs_concrete_poi",
    "not_in_sandbox",
    "not_in_profile",
    "weak_sandbox_match",
    "oh_conflict",
    "oh_match_uncertain",
    "too_sparse_day",
    "no_representative_anchor",
    "bad_general_poi",
    # complex_transfer, route_too_spread, weak_representative_anchor 은 LOW severity
    # replanner 트리거 안 함 (repair 또는 허용)
}

MEAL_ISSUE_TYPES = {
    "lunch_missing",
    "dinner_missing",
    "meal_missing",
}

DEFAULT_DAY_POI_COUNT = 4
MIN_FULL_DAY_POI_COUNT = 4
MAX_DAY_POI_COUNT = 5

DEFAULT_START_TIME = "10:00"
TRAVEL_BUFFER_MINUTES = 25

PURPOSE_TO_PRIORITY_ROLES = {
    "general": ["attraction", "culture", "shopping", "market", "nature", "history"],
    "culture": ["culture", "history", "attraction", "market"],
    "history": ["history", "culture", "attraction"],
    "shopping": ["shopping", "market", "beauty", "attraction"],
    "food": ["meal", "market", "cafe", "attraction"],
    "cafe": ["cafe", "attraction", "shopping"],
    "cafe_hopping": ["cafe", "attraction", "shopping"],
    "kpop": ["kpop", "shopping", "culture", "attraction"],
    "nature": ["nature", "family", "attraction", "cafe"],
    "family": ["family", "nature", "attraction", "indoor_leisure"],
    "nightlife": ["nightlife", "meal", "shopping"],
    "beauty": ["beauty", "shopping", "attraction"],
}

MEAL_ROLES = {"meal", "cafe", "market"}
ANCHOR_ROLES = {
    "attraction", "shopping", "history", "culture", "nature",
    "kpop", "family", "market", "beauty", "indoor_leisure"
}

BAD_GENERAL_ROLES = {"accommodation", "transport", "education"}
NIGHTLIFE_ONLY_ROLES = {"nightlife"}

SPECIAL_ACTIVITY_PATTERNS = {
    "water sports", "수상스포츠",
    "one day", "one-day", "원데이",
    "class", "체험",
    "photo booth", "self photo", "셀프사진", "포토부스",
}

# station 패턴: "역" 단독 토큰이거나 지하철역/버스역처럼 역으로 끝나는 교통시설 단어
# 무역센터, 경의선처럼 "역"이 중간에 포함된 일반 단어는 제외
STATION_PATTERNS = {
    "station",             # 영문 station
    "지하철역",
    "버스역",
    "기차역",
    "전철역",
}
# 한글 "역" 단독 매치용 (토큰 단위로 검사)
STATION_EXACT_TOKENS = {"역"}

BAD_GENERAL_PATTERNS_NO_STATION = {
    "karaoke", "노래방",
    "guest house", "guesthouse", "게스트하우스",
    "university", "univ", "대학교", "대학",
    "airport", "공항",
}

# BROAD_AREA_EXACT_ALIASES: 이름 전체가 이와 일치하면 vague
BROAD_AREA_EXACT_ALIASES = {
    "hongdae", "hongik univ", "hongik university", "hongik university street",
    "gangnam", "gangnam station",
    "seongsu", "seongsu dong", "seongsu-dong",
    "yongsan", "mangwon dong", "mangwon-dong", "yeonnam dong", "yeonnam-dong",
    "jongno", "myeongdong", "itaewon", "sinchon", "mapo", "jamsil",
    "bukchon", "insa dong", "insadong", "ikseon dong", "hannam dong", "euljiro area",
    "gwanghwamun area",
    "홍대", "홍익대", "홍익대학교", "강남", "강남역", "성수", "성수동",
    "용산", "망원동", "연남동", "종로", "명동", "이태원", "신촌", "마포", "잠실",
    "북촌", "인사동", "익선동", "한남동", "을지로", "광화문",
}

CONCRETE_MARKERS = {
    "performance", "market", "park", "museum", "palace", "temple", "library",
    "cafe street", "food street", "shopping street", "shopping center", "mall",
    "square", "village", "trail", "alley", "street performance", "observatory",
    "거리공연", "시장", "공원", "궁", "사원", "절", "도서관", "광장", "마을", "길", "골목",
}

AREA_ALIASES = {
    "hongdae_area": {
        "hongdae", "hongik", "hongik university", "yeonnam", "mangwon", "mangnidan", "mapo",
        "gyeongui", "홍대", "홍익대", "연남", "망원", "망리단", "마포", "경의선",
    },
    "gangnam_area": {
        "gangnam", "coex", "samseong", "starfield", "bongeunsa", "garosu", "apgujeong",
        "sinsa", "dosan", "seocho", "seolleung",
        "강남", "코엑스", "삼성", "스타필드", "봉은사", "가로수", "압구정", "신사", "도산", "서초", "선릉",
    },
    "seongsu": {
        "seongsu", "seoul forest", "성수", "성수동", "서울숲",
    },
    "jongno_area": {
        "jongno", "insadong", "bukchon", "gyeongbokgung", "gwanghwamun", "samcheong",
        "ikseon", "daehangno", "종로", "인사동", "북촌", "경복궁", "광화문", "삼청", "익선", "대학로",
    },
    "yongsan_itaewon_area": {
        "yongsan", "itaewon", "hannam", "hybe", "leeum", "용산", "이태원", "한남", "하이브", "리움",
    },
    "myeongdong_euljiro_area": {
        "myeongdong", "euljiro", "namdaemun", "dongdaemun", "ddp",
        "명동", "을지로", "남대문", "동대문",
    },
    "yeouido": {"yeouido", "ifc", "hyundai seoul", "hangang", "여의도", "더현대", "한강"},
    "jamsil": {"jamsil", "lotte world", "songnidan", "잠실", "롯데월드", "송리단"},
}


# ============================================================
# Dataclass
# ============================================================

@dataclass
class ReplanAction:
    action_type: str
    day: int | None
    status: str
    reason: str
    before: list[str]
    after: list[str]
    target_area: str | None = None
    evidence: list[str] | None = None


@dataclass
class ReplanResult:
    changed: bool
    passed_to_critic: bool
    needs_replan: bool
    actions: list[ReplanAction]
    warnings: list[str]
    debug: dict[str, Any]


# ============================================================
# 기본 유틸
# ============================================================

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def clean_str(value: Any, default: str = "") -> str:
    if is_missing(value):
        return default
    return str(value).strip()


def normalize_name(value: Any) -> str:
    if is_missing(value):
        return ""
    s = str(value).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^0-9a-z가-힣]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_any(text: Any, patterns: set[str] | list[str]) -> bool:
    n = normalize_name(text)
    for p in patterns:
        pp = normalize_name(p)
        if pp and pp in n:
            return True
    return False


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if is_missing(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if is_missing(value):
            return default
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def issue_get(issue: Any, *keys: str, default: Any = None) -> Any:
    d = as_dict(issue)
    for k in keys:
        if k in d:
            return d[k]
    return getattr(issue, keys[0], default) if keys else default


def public_poi_name(poi: dict[str, Any]) -> str:
    return clean_str(poi.get("name") or poi.get("poi_name") or poi.get("title"))


def poi_type(poi: dict[str, Any]) -> str:
    return clean_str(poi.get("type") or poi.get("poi_type"), "tourist_spot")


def extract_day_number(day_obj: dict[str, Any], idx: int) -> int:
    return safe_int(day_obj.get("day"), idx + 1) or idx + 1


def time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def minutes_to_time(m: int) -> str:
    m = max(0, min(23 * 60 + 59, int(m)))
    return f"{m // 60:02d}:{m % 60:02d}"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# 이름/후보 판단
# ============================================================

def dedupe_key(name: str) -> str:
    n = normalize_name(name)

    if "starfield library" in n or "별마당" in n:
        return "starfield_library"
    if "bongeunsa" in n or "봉은사" in n:
        return "bongeunsa_temple"
    if "coex" in n or "코엑스" in n:
        return "coex_complex"
    if "hangaram art museum" in n or "한가람미술관" in n:
        return "hangaram_art_museum"
    if "seoul arts center" in n or "예술의전당" in n:
        return "seoul_arts_center"
    if "gyeongui line forest park" in n or "경의선숲길" in n or "경의선 숲길" in n:
        return "gyeongui_line_forest_park"
    if "yeonnam" in n and ("cafe" in n or "카페" in n):
        return "yeonnam_cafe_street"
    if "mangwon market" in n or "망원시장" in n or "망원 시장" in n:
        return "mangwon_market"
    if "mangnidan" in n or "망리단" in n:
        return "mangnidan_street"
    if "hongdae street performance" in n or "홍대 거리공연" in n:
        return "hongdae_street_performance"
    if "k pop square hongdae" in n or "kpop square hongdae" in n or "케이팝 스퀘어 홍대" in n:
        return "kpop_square_hongdae"
    if "seoul forest" in n or "서울숲" in n:
        return "seoul_forest"

    return n


def candidate_text(p: dict[str, Any]) -> str:
    return " ".join([
        clean_str(p.get("name")),
        clean_str(p.get("poi_type")),
        clean_str(p.get("address")),
        " ".join(str(x) for x in (p.get("purpose_tags") or [])),
    ])


def candidate_has_special_activity(p: dict[str, Any]) -> bool:
    return contains_any(candidate_text(p), SPECIAL_ACTIVITY_PATTERNS)


def candidate_is_vague_or_broad_name(name: str, ptype: str = "") -> bool:
    """
    일반 후보(role candidate, fallback)용 vague/broad 검사.
    BROAD_AREA_EXACT_ALIASES 부분 매치(3토큰 이하 이름)도 검사한다.
    """
    n = normalize_name(name)
    t = normalize_name(ptype)

    if not n:
        return True

    # concrete marker가 있으면 구체적인 이름 → vague 아님
    if any(normalize_name(m) in n for m in CONCRETE_MARKERS):
        return False

    # 전체 이름이 broad alias와 완전 일치
    if n in BROAD_AREA_EXACT_ALIASES:
        return True

    if t in {"area", "district", "neighborhood"}:
        return True

    # 3토큰 이하 짧은 이름에서 broad alias 부분 포함 검사
    tokens = n.split()
    if len(tokens) <= 3:
        for alias in BROAD_AREA_EXACT_ALIASES:
            a = normalize_name(alias)
            if a and a in n:
                return True

    return False


def candidate_is_vague_or_broad_name_strict(name: str, ptype: str = "") -> bool:
    """
    representative_pois용 완화된 vague/broad 검사.
    - 이름이 비어있으면 True
    - is_vague_or_broad 플래그는 호출 전에 이미 검사
    - 전체 이름이 BROAD_AREA_EXACT_ALIASES와 완전 일치할 때만 True
    - 부분 매치는 검사하지 않음 (구체적인 대표 장소 이름 보호)
    - poi_type이 area/district/neighborhood이면 True
    """
    n = normalize_name(name)
    t = normalize_name(ptype)

    if not n:
        return True

    # concrete marker가 있으면 무조건 통과
    if any(normalize_name(m) in n for m in CONCRETE_MARKERS):
        return False

    # 완전 일치만 검사 (부분 매치 제거)
    if n in BROAD_AREA_EXACT_ALIASES:
        return True

    if t in {"area", "district", "neighborhood"}:
        return True

    return False


def _text_contains_station_pattern(text: str) -> bool:
    """
    교통 station 패턴 검사.
    단순 "역" 포함이 아니라:
    - 영문 "station" 단어 포함
    - 한글 "역" 토큰이 단독으로 존재하거나 교통시설 접두/접미어와 함께 등장
    - 지하철역, 버스역, 기차역, 전철역 포함
    오탐 방지: 무역센터, 경의선, 전통역사 같은 단어에는 걸리지 않음
    """
    n = normalize_name(text)
    tokens = n.split()

    # 영문 station
    if "station" in tokens:
        return True

    # 교통시설 복합 단어
    for pat in {"지하철역", "버스역", "기차역", "전철역"}:
        if normalize_name(pat) in n:
            return True

    # 한글 "역" 단독 토큰 (예: "강남역" 처럼 붙어있는 경우는 dedupe_key/BROAD_AREA로 처리)
    if "역" in tokens:
        return True

    # "역" 접미사로 끝나는 단어가 있고 그게 지명+역 패턴인지 확인
    # e.g. "강남역" → 강남 + 역 → BROAD_AREA에 이미 있음
    # e.g. "경의선숲길" → 역 없음 → 통과
    # e.g. "무역센터" → 역이 중간에 있고 단독 토큰 아님 → 통과
    for token in tokens:
        if token.endswith("역") and len(token) >= 2:
            # 무역, 역할, 역사, 역대 같은 단어 제외
            non_station_endings = {"무역", "역할", "역사", "역대", "역량", "역설", "역시", "역점", "역전"}
            if token not in non_station_endings and not any(token == ns for ns in non_station_endings):
                # 2글자 이상이고 비교통 단어가 아니면 station으로 판단
                # 단, "역" 자체를 포함하는 단어라도 명확한 교통 맥락이 아니면 보수적으로 False
                # 이 블록은 "강남역", "홍대입구역" 등을 잡기 위한 것
                # → 이미 BROAD_AREA_EXACT_ALIASES에서 처리됨
                # → 여기서는 단독 "역" 토큰만 처리하고 나머지는 BROAD_AREA에 맡김
                pass

    return False


def profile_poi_is_meal_like(p: dict[str, Any]) -> bool:
    roles = set(p.get("roles") or [])
    ptype = normalize_name(p.get("poi_type", ""))
    return bool(roles & MEAL_ROLES) or ptype in {"restaurant", "food", "cafe", "market"}


def profile_poi_is_anchor_like(
    p: dict[str, Any], purpose: str = "general", trusted_representative: bool = False
) -> bool:
    """
    trusted_representative=True: roles 오염(umbrella cluster union) 우회.
    poi_type 기반으로만 판단한다.
    """
    roles = set(p.get("roles") or [])
    ptype = normalize_name(p.get("poi_type", ""))
    purpose = normalize_name(purpose)

    if purpose in {"food"}:
        return bool(roles & {"meal", "market", "cafe", "attraction", "shopping"})
    if purpose in {"cafe", "cafe_hopping"}:
        return bool(roles & {"cafe", "attraction", "shopping", "market"})
    if purpose == "shopping":
        return bool(roles & {"shopping", "market", "beauty", "attraction"})
    if purpose == "kpop":
        return bool(roles & {"kpop", "shopping", "culture", "attraction"})
    if purpose == "nature":
        return bool(roles & {"nature", "family", "attraction"})
    if purpose == "family":
        return bool(roles & {"family", "nature", "indoor_leisure", "attraction"})
    if purpose == "beauty":
        return bool(roles & {"beauty", "shopping", "attraction"})

    if purpose == "general" and not trusted_representative:
        if roles & {"accommodation", "transport", "education", "nightlife"}:
            return False
        if candidate_has_special_activity(p):
            return False

    return bool(roles & ANCHOR_ROLES) or ptype in {
        "tourist_spot", "street", "park", "museum", "shopping", "market",
        "culture", "history", "library", "kpop_landmark",
    }


def purpose_allows_candidate(p: dict[str, Any], purpose: str) -> bool:
    roles = set(p.get("roles") or [])
    purpose = normalize_name(purpose or "general")

    if purpose == "shopping" and roles & {"shopping", "market", "beauty", "attraction"}:
        return True
    if purpose == "kpop" and roles & {"kpop", "shopping", "culture", "attraction"}:
        return True
    if purpose == "food" and roles & {"meal", "market", "cafe"}:
        return True
    if purpose in {"cafe", "cafe_hopping"} and roles & {"cafe", "attraction", "shopping"}:
        return True
    if purpose == "nature" and roles & {"nature", "family", "attraction"}:
        return True
    if purpose == "family" and roles & {"family", "nature", "indoor_leisure", "attraction"}:
        return True
    if purpose == "beauty" and roles & {"beauty", "shopping"}:
        return True
    if purpose in {"history", "culture"} and roles & {"history", "culture", "attraction"}:
        return True
    if purpose == "nightlife" and roles & {"nightlife", "meal"}:
        return True

    return False


def candidate_is_bad_for_purpose(p: dict[str, Any], purpose: str) -> bool:
    """
    일반 role 후보 / fallback 후보용 필터.
    representative_pois에는 적용하지 않는다.

    v3.1 수정:
    - "역" 단순 포함 → _text_contains_station_pattern() 으로 교체
      (무역센터, 경의선, 역사적 장소 등 오탐 방지)
    """
    roles = set(p.get("roles") or [])
    text = candidate_text(p)
    purpose = normalize_name(purpose or "general")

    if purpose_allows_candidate(p, purpose):
        return False

    if purpose not in {"nightlife"} and roles & NIGHTLIFE_ONLY_ROLES:
        return True
    if purpose not in {"transport", "education", "accommodation"} and roles & BAD_GENERAL_ROLES:
        return True

    if purpose != "nightlife" and contains_any(text, {"karaoke", "노래방"}):
        return True
    if purpose not in {"education"} and contains_any(text, {"university", "대학교", "대학"}):
        return True
    if purpose not in {"accommodation"} and contains_any(text, {"guest house", "guesthouse", "게스트하우스"}):
        return True

    # v3.1: "역" 단순 패턴 → 교통 station 전용 패턴으로 교체
    if purpose not in {"transport"} and _text_contains_station_pattern(text):
        return True

    return False


def candidate_is_bad_for_purpose_representative(p: dict[str, Any], purpose: str) -> bool:
    """
    representative_pois 전용 필터. 이름(text) 기반으로만 판단한다.

    주의: area_profiles_v2.json의 umbrella cluster representative_pois는
    roles 필드에 accommodation/transport/education/nightlife 등이 모두 포함된다.
    (builder가 하위 cluster POI의 roles를 union으로 합산하는 구조)
    따라서 roles 기반 필터는 절대 사용하지 않는다.

    걸러내는 케이스 (이름 기반만):
    - 노래방
    - 게스트하우스
    - 공항
    - 한국종합무역센터
    """
    text = candidate_text(p)
    name = normalize_name(clean_str(p.get("name")))

    if contains_any(text, {"karaoke", "노래방"}):
        return True
    if contains_any(text, {"guest house", "guesthouse", "게스트하우스"}):
        return True
    if contains_any(text, {"airport", "공항"}):
        return True
    if "한국종합무역센터" in name:
        return True

    return False


def candidate_rank_tuple(
    p: dict[str, Any],
    purpose: str,
    role: str | None = None,
    prefer_representative: bool = False,
) -> tuple:
    rep_score = safe_float(p.get("representative_score"), 0.0) or 0.0
    role_score = safe_float(p.get("role_score"), 0.0) or 0.0
    general_ok = 1 if p.get("general_representative_ok", True) else 0
    is_dual = 1 if p.get("is_dual_role_anchor") else 0
    is_rep = 1 if prefer_representative else 0

    special_penalty = -1 if normalize_name(purpose) == "general" and candidate_has_special_activity(p) else 0
    meal_penalty_for_anchor = -1 if role == "anchor" and profile_poi_is_meal_like(p) else 0

    return (
        is_rep,
        special_penalty,
        meal_penalty_for_anchor,
        general_ok,
        rep_score,
        role_score,
        is_dual,
    )


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for p in candidates:
        name = clean_str(p.get("name"))
        key = dedupe_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def distance_km_between_pois(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    try:
        lat1 = safe_float(a.get("lat"))
        lng1 = safe_float(a.get("lng"))
        lat2 = safe_float(b.get("lat"))
        lng2 = safe_float(b.get("lng"))
        if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
            return None
        return haversine_km(lat1, lng1, lat2, lng2)
    except Exception:
        return None


def reorder_tail_by_distance(
    selected: list[dict[str, Any]],
    lunch_idx: int | None = None,
) -> list[dict[str, Any]]:
    """
    첫 anchor(index 0)와 lunch(index 1 또는 lunch_idx)는 고정하고
    나머지를 거리 기반으로 정렬.

    v3.1: lunch_idx를 명시적으로 받아서 고정 구간을 정확히 처리.
    lunch_idx가 None이면 기본 selected[:2]를 고정.
    """
    if len(selected) <= 2:
        return selected

    fixed_count = 2
    if lunch_idx is not None and lunch_idx >= 1:
        fixed_count = lunch_idx + 1

    fixed = selected[:fixed_count]
    tail = selected[fixed_count:]

    if not tail:
        return selected

    ordered = list(fixed)
    current = fixed[-1]

    remaining = list(tail)
    while remaining:
        best_idx = 0
        best_dist = float("inf")
        for i, cand in enumerate(remaining):
            d = distance_km_between_pois(current, cand)
            if d is None:
                d = 999.0
            if d < best_dist:
                best_dist = d
                best_idx = i
        current = remaining.pop(best_idx)
        ordered.append(current)

    return ordered


# ============================================================
# Area inference / profile index
# ============================================================

def infer_area_from_text(text: str, profile: dict[str, Any]) -> str | None:
    n = normalize_name(text)

    scores = []
    for area, aliases in AREA_ALIASES.items():
        score = 0
        for alias in aliases:
            aa = normalize_name(alias)
            if aa and aa in n:
                score += len(aa)
        if score:
            scores.append((score, area))

    if scores:
        return sorted(scores, reverse=True)[0][1]

    for key in profile.get("clusters", {}):
        kk = normalize_name(key)
        if kk and kk in n:
            return key

    return None


def normalize_purpose(user_state: dict[str, Any], day: dict[str, Any] | None = None) -> str:
    texts = []
    for key in ["purpose", "travel_purpose", "theme", "interest", "interests"]:
        val = user_state.get(key)
        if val:
            texts.append(" ".join(val) if isinstance(val, list) else str(val))
    if day:
        texts.append(str(day.get("theme", "")))

    blob = normalize_name(" ".join(texts))

    checks = [
        ("kpop", ["kpop", "k pop", "케이팝", "한류", "idol", "아이돌"]),
        ("shopping", ["shopping", "쇼핑", "mall", "market", "시장"]),
        ("food", ["food", "restaurant", "맛집", "먹거리", "음식"]),
        ("cafe_hopping", ["cafe", "카페"]),
        ("history", ["history", "palace", "museum", "역사", "궁", "박물관"]),
        ("culture", ["culture", "art", "문화", "예술"]),
        ("nature", ["nature", "park", "forest", "자연", "공원", "숲"]),
        ("family", ["family", "kids", "가족", "아이"]),
        ("nightlife", ["nightlife", "bar", "club", "밤", "야경", "술"]),
        ("beauty", ["beauty", "cosmetic", "뷰티", "화장품"]),
    ]

    for purpose, keys in checks:
        if any(normalize_name(k) in blob for k in keys):
            return purpose
    return "general"


class AreaProfileIndex:
    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = profile
        self.clusters = profile.get("clusters", {})
        self.name_to_profile_poi: dict[str, dict[str, Any]] = {}
        self._build_name_index()

    @classmethod
    def from_path(cls, path: Path) -> "AreaProfileIndex":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def _build_name_index(self) -> None:
        for cdata in self.clusters.values():
            pools = []
            pools.extend(cdata.get("representative_pois", []))
            pools.extend(cdata.get("dual_role_anchors", []))
            pools.extend(cdata.get("vague_or_broad_pois", []))
            pools.extend(cdata.get("downranked_or_excluded_pois", []))
            for role_list in cdata.get("role_candidates", {}).values():
                pools.extend(role_list)

            for p in pools:
                name = clean_str(p.get("name"))
                if not name:
                    continue
                self.name_to_profile_poi.setdefault(normalize_name(name), p)

    def find_profile_poi(self, name: str) -> dict[str, Any] | None:
        n = normalize_name(name)
        if not n:
            return None
        if n in self.name_to_profile_poi:
            return self.name_to_profile_poi[n]

        best = None
        best_score = 0.0
        n_tokens = set(n.split())

        for key, p in self.name_to_profile_poi.items():
            score = 0.0
            if key in n or n in key:
                score = min(len(key), len(n)) / max(len(key), len(n))
            else:
                k_tokens = set(key.split())
                if n_tokens and k_tokens:
                    score = len(n_tokens & k_tokens) / max(len(n_tokens | k_tokens), 1)

            if score > best_score:
                best = p
                best_score = score

        if best_score >= 0.55:
            return best
        return None

    def get_cluster(self, key: str | None) -> dict[str, Any]:
        if not key:
            return {}
        return self.clusters.get(key, {})

    def compatible_clusters(self, area_key: str | None) -> set[str]:
        if not area_key:
            return set()
        cdata = self.get_cluster(area_key)
        compatible = set(cdata.get("compatible_clusters") or [])
        compatible |= set(cdata.get("members") or [])
        compatible.add(area_key)
        return compatible

    def is_profile_poi_compatible(self, poi: dict[str, Any], area_key: str | None) -> bool:
        if not area_key:
            return True
        cluster = clean_str(poi.get("cluster"))
        if not cluster:
            return False
        return cluster in self.compatible_clusters(area_key) or cluster == area_key


def infer_day_target_area(
    day: dict[str, Any], user_state: dict[str, Any], profile: dict[str, Any]
) -> str | None:
    texts = [
        clean_str(day.get("theme")),
        clean_str(day.get("title")),
        clean_str(day.get("area")),
        clean_str(day.get("location")),
        clean_str(user_state.get("location")),
        clean_str(user_state.get("area")),
    ]

    for text in texts:
        area = infer_area_from_text(text, profile)
        if area:
            return area

    joined = " ".join(public_poi_name(p) for p in day.get("pois", []))
    return infer_area_from_text(joined, profile)


# ============================================================
# Candidate selector
# ============================================================

class CandidateSelector:
    def __init__(self, index: AreaProfileIndex) -> None:
        self.index = index

    # ----------------------------------------------------------
    # v3.1 핵심 수정: representative 전용 필터 경로 분리
    # ----------------------------------------------------------

    def _clean_representative_pool(
        self,
        area_key: str,
        pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        representative_pois 전용 필터.
        적용 순서:
        1. 이름 없음 → 제거
        2. is_vague_or_broad 플래그 → 제거
        3. candidate_is_vague_or_broad_name_strict() → 완전 일치만 검사, 부분 매치 없음
        4. area 호환성 검사
        5. candidate_is_bad_for_purpose_representative() → 숙소/노래방/공항만 제거

        candidate_is_bad_for_purpose()는 적용하지 않음.
        """
        out = []
        for p in pool:
            name = clean_str(p.get("name"))
            if not name:
                continue
            # vague/broad 플래그
            if p.get("is_vague_or_broad"):
                continue
            # 이름 기반 vague 검사 (완화 버전)
            if candidate_is_vague_or_broad_name_strict(name, clean_str(p.get("poi_type"))):
                continue
            # area 호환성
            if not self.index.is_profile_poi_compatible(p, area_key):
                continue
            # 절대 불가 케이스만 (숙소, 노래방, 공항)
            if candidate_is_bad_for_purpose_representative(p, ""):
                continue
            out.append(p)

        return dedupe_candidates(out)

    def _compatible_clean_pool(
        self,
        area_key: str,
        pool: list[dict[str, Any]],
        purpose: str,
    ) -> list[dict[str, Any]]:
        """
        일반 role 후보 / fallback 후보용 필터.
        representative_pois에는 사용하지 않음.
        """
        out = []
        for p in pool:
            name = clean_str(p.get("name"))
            if not name:
                continue
            if p.get("is_vague_or_broad"):
                continue
            if candidate_is_vague_or_broad_name(name, clean_str(p.get("poi_type"))):
                continue
            if not self.index.is_profile_poi_compatible(p, area_key):
                continue
            if candidate_is_bad_for_purpose(p, purpose):
                continue
            out.append(p)

        return dedupe_candidates(out)

    # 명백히 식사 불가능한 poi_type
    _NON_MEAL_POI_TYPES = {
        "street", "park", "nature", "history", "culture", "museum",
        "tourist_spot", "kpop_landmark", "shopping", "library",
        "shopping_mall", "theme_park", "gallery",
    }
    # 이름에 있으면 poi_type 무시하고 식사 가능으로 인정
    _MEAL_NAME_KEYWORDS = {
        "restaurant", "식당", "칼국수", "냉면", "라멘", "kalguksu",
        "bbq", "grill", "갈비", "삼겹살", "치킨", "chicken",
        "ramen", "noodle", "국밥", "순대", "떡볶이", "pizza", "burger",
        "sushi", "steak", "bistro", "kitchen", "eatery", "diner",
        "cafe", "coffee", "카페", "커피", "bakery", "베이커리",
        "market", "시장",
    }

    def _meal_clean_pool(
        self,
        area_key: str,
        pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        식사 후보 전용 필터.
        - area 호환성 체크 없음
        - poi_type이 명백히 비식사이고 이름에도 식사 키워드 없으면 제외
        - 절대 불가(노래방/공항)만 추가 제외
        """
        out = []
        for p in pool:
            name = clean_str(p.get("name"))
            if not name:
                continue
            if p.get("is_vague_or_broad"):
                continue
            n = normalize_name(name)
            if not n:
                continue
            ptype = normalize_name(p.get("poi_type", ""))
            has_meal_keyword = any(k in n for k in self._MEAL_NAME_KEYWORDS)
            if ptype in self._NON_MEAL_POI_TYPES and not has_meal_keyword:
                continue
            text = candidate_text(p)
            if contains_any(text, {"karaoke", "노래방", "guest house", "guesthouse", "airport", "공항"}):
                continue
            out.append(p)
        return dedupe_candidates(out)

    def get_area_candidates(
        self, area_key: str, purpose: str
    ) -> dict[str, list[dict[str, Any]]]:
        cdata = self.index.get_cluster(area_key)
        if not cdata:
            return {
                "representatives": [],
                "attractions": [],
                "meals": [],
                "cafes": [],
                "purpose": [],
                "raw_representatives": [],
            }

        role_candidates = cdata.get("role_candidates", {})
        priority_roles = PURPOSE_TO_PRIORITY_ROLES.get(
            purpose, PURPOSE_TO_PRIORITY_ROLES["general"]
        )

        # --- representative: 전용 필터 경로 사용 ---
        raw_representatives = cdata.get("representative_pois", [])
        representatives = self._clean_representative_pool(
            area_key=area_key,
            pool=raw_representatives,
        )
        representatives.sort(
            key=lambda p: candidate_rank_tuple(
                p, purpose, role="anchor", prefer_representative=True
            ),
            reverse=True,
        )

        # --- attractions: 일반 필터 경로 ---
        raw_attractions = []
        for role in [
            "attraction", "culture", "history", "shopping",
            "market", "nature", "kpop", "family", "beauty",
        ]:
            raw_attractions.extend(role_candidates.get(role, []))
        raw_attractions.extend(cdata.get("dual_role_anchors", []))

        attractions = self._compatible_clean_pool(
            area_key=area_key,
            pool=raw_attractions,
            purpose=purpose,
        )
        attractions = [p for p in attractions if profile_poi_is_anchor_like(p, purpose)]
        attractions.sort(
            key=lambda p: candidate_rank_tuple(p, purpose, role="anchor"), reverse=True
        )

        # --- meals ---
        # meal/market 후보는 전용 필터 사용 (비식사 poi_type 제거)
        meals = self._meal_clean_pool(
            area_key=area_key,
            pool=role_candidates.get("meal", []) + role_candidates.get("market", []),
        )
        meals.sort(key=lambda p: candidate_rank_tuple(p, purpose, role="meal"), reverse=True)

        # --- cafes ---
        cafes = self._compatible_clean_pool(
            area_key=area_key,
            pool=role_candidates.get("cafe", []),
            purpose="cafe_hopping" if purpose == "general" else purpose,
        )
        cafes.sort(key=lambda p: candidate_rank_tuple(p, purpose, role="cafe"), reverse=True)

        # --- purpose pool ---
        purpose_pool: list[dict[str, Any]] = []
        for role in priority_roles:
            purpose_pool.extend(role_candidates.get(role, []))
        purpose_pool = self._compatible_clean_pool(
            area_key=area_key,
            pool=purpose_pool,
            purpose=purpose,
        )
        purpose_pool.sort(
            key=lambda p: candidate_rank_tuple(p, purpose, role="anchor"), reverse=True
        )

        return {
            "representatives": representatives,
            "attractions": dedupe_candidates(attractions),
            "meals": dedupe_candidates(meals),
            "cafes": dedupe_candidates(cafes),
            "purpose": dedupe_candidates(purpose_pool),
            "raw_representatives": raw_representatives,
        }

    def select_rebuilt_day_pois(
        self,
        area_key: str,
        purpose: str,
        original_day: dict[str, Any],
        issue_types: set[str],
        excluded_keys: set[str] | None = None,
        supplement_restaurants: list[dict[str, Any]] | None = None,
        user_state: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        excluded_keys: 다른 day에서 이미 사용된 POI dedupe_key 집합 (day간 중복 방지)
        supplement_restaurants: Google Places 식당/카페 후보 (재사용용)
        user_state: dietary 등 사용자 정보 (Google Places dietary 필터용)
        """
        evidence: list[str] = []
        pools = self.get_area_candidates(area_key, purpose)
        excluded_keys = excluded_keys or set()
        supplement_restaurants = supplement_restaurants or []
        user_state = user_state or {}

        representatives = pools["representatives"]
        attractions = dedupe_candidates(
            pools["representatives"] + pools["purpose"] + pools["attractions"]
        )
        meals = pools["meals"]
        cafes = pools["cafes"]

        evidence.append(f"area={area_key}, purpose={purpose}")
        evidence.append(
            f"raw_representatives={len(pools['raw_representatives'])}, "
            f"representatives={len(representatives)}, "
            f"attractions={len(attractions)}, meals={len(meals)}, cafes={len(cafes)}"
        )

        if len(representatives) == 0 and len(pools["raw_representatives"]) > 0:
            evidence.append(
                "warning: raw representatives exist but all were filtered. "
                "Check compatibility/bad filters."
            )

        # fallback: attractions 부족 시 dual_role/market/cafe 후보로 보충
        if len(attractions) < 2:
            fallback = []
            for p in meals + cafes + representatives:
                roles = set(p.get("roles") or [])
                if p.get("is_dual_role_anchor") or roles & {"market", "shopping", "cafe", "attraction"}:
                    fallback.append(p)
            attractions = dedupe_candidates(attractions + fallback)
            evidence.append(f"dual_role_fallback_attractions={len(attractions)}")

        if len(attractions) < 2 and len(representatives) < 2:
            return [], evidence + ["not_enough_safe_anchors"]

        original_count = len(original_day.get("pois", []))
        desired_count = max(MIN_FULL_DAY_POI_COUNT, original_count, DEFAULT_DAY_POI_COUNT)
        desired_count = min(MAX_DAY_POI_COUNT, desired_count)

        selected: list[dict[str, Any]] = []
        used: set[str] = set()

        def add_candidate(p: dict[str, Any]) -> bool:
            key = dedupe_key(clean_str(p.get("name")))
            if not key or key in used:
                return False
            if key in excluded_keys:
                return False
            selected.append(p)
            used.add(key)
            return True

        # ------------------------------------------------
        # 1) first_anchor: representative에서 non-meal 우선
        #    trusted_representative=True → roles 오염 우회, poi_type 기반 판단
        # ------------------------------------------------
        first_anchor = None
        for p in representatives:
            if profile_poi_is_anchor_like(p, purpose, trusted_representative=True) and not profile_poi_is_meal_like(p):
                first_anchor = p
                break

        if first_anchor is None:
            for p in representatives:
                if profile_poi_is_anchor_like(p, purpose, trusted_representative=True):
                    first_anchor = p
                    break

        # representative 없을 때만 attraction fallback
        if first_anchor is None:
            for p in attractions:
                if profile_poi_is_anchor_like(p, purpose) and not profile_poi_is_meal_like(p):
                    first_anchor = p
                    break

        if first_anchor is None and attractions:
            first_anchor = attractions[0]

        if first_anchor is None:
            return [], evidence + ["no_first_anchor"]

        add_candidate(first_anchor)
        evidence.append(f"first_anchor={first_anchor.get('name')}")

        # ------------------------------------------------
        # 2) lunch 선택 — 새 우선순위
        #
        #    1순위: Google Places 직접 호출 (실시간, dietary 필터, 인근 거리순)
        #    2순위: supplement_restaurants (generator가 미리 가져온 것, 재호출 없이 재사용)
        #    3순위: profile meals pool (최후 수단, 데이터 품질 낮음)
        #
        #    - 비식사 장소는 절대 선택 안 함
        #    - 브레이크 타임(15:00~17:00): _attach_schedule에서 식당 18:00으로 자동 이동
        # ------------------------------------------------

        def _supplement_to_poi(s: dict[str, Any]) -> dict[str, Any]:
            """Google Places 결과를 profile POI 형식으로 변환."""
            return {
                "name": s.get("poi_name") or s.get("name", ""),
                "poi_type": s.get("poi_type", "restaurant"),
                "lat": s.get("lat"),
                "lng": s.get("lng"),
                "roles": ["meal"] if s.get("poi_type") == "restaurant" else ["cafe"],
                "rating": s.get("rating"),
                "user_ratings_total": s.get("user_ratings_total", 0),
                "cluster": area_key,
                "source": s.get("source", "Google Places"),
                "address": s.get("address_en") or s.get("address", ""),
            }

        def _calc_dist(s: dict[str, Any], anchor: dict[str, Any]) -> float:
            lat1 = safe_float(anchor.get("lat"))
            lng1 = safe_float(anchor.get("lng"))
            lat2 = safe_float(s.get("lat"))
            lng2 = safe_float(s.get("lng"))
            if None in {lat1, lng1, lat2, lng2}:
                return 999.0
            return haversine_km(lat1, lng1, lat2, lng2)

        anchor_for_dist = selected[0] if selected else None
        lunch_added = False
        lunch_idx: int | None = None

        # dietary 제한 파싱 (user_state에서)
        dietary_str = user_state.get("dietary") or ""
        dietary_restrictions = parse_dietary_restrictions(dietary_str)

        # ---- 1순위: Google Places 직접 호출 ----
        if not lunch_added and GOOGLE_PLACES_API_KEY and anchor_for_dist:
            anchor_lat = safe_float(anchor_for_dist.get("lat"))
            anchor_lng = safe_float(anchor_for_dist.get("lng"))
            if anchor_lat and anchor_lng:
                live_results = fetch_restaurants_for_area(
                    area_key=area_key,
                    anchor_lat=anchor_lat,
                    anchor_lng=anchor_lng,
                    api_key=GOOGLE_PLACES_API_KEY,
                    dietary_restrictions=dietary_restrictions,
                    radius=1000,
                )
                # 거리순 정렬
                live_results.sort(key=lambda s: _calc_dist(s, anchor_for_dist))
                evidence.append(f"google_live={len(live_results)}개 (dietary={dietary_restrictions or 'none'})")
                for s in live_results:
                    sp = _supplement_to_poi(s)
                    key = dedupe_key(clean_str(sp.get("name")))
                    if key and key not in used and key not in excluded_keys:
                        used.add(key)
                        selected.append(sp)
                        lunch_added = True
                        lunch_idx = len(selected) - 1
                        dist = _calc_dist(s, anchor_for_dist)
                        evidence.append(f"lunch_candidate={sp.get('name')} (Google Places live, {dist:.1f}km)")
                        break

        # ---- 2순위: supplement_restaurants (generator가 미리 가져온 것) ----
        if not lunch_added and supplement_restaurants and anchor_for_dist:
            sorted_sups = sorted(supplement_restaurants, key=lambda s: _calc_dist(s, anchor_for_dist))
            evidence.append(f"supplement_restaurants={len(sorted_sups)}개 (거리순)")
            for s in sorted_sups:
                sp = _supplement_to_poi(s)
                key = dedupe_key(clean_str(sp.get("name")))
                if key and key not in used and key not in excluded_keys:
                    used.add(key)
                    selected.append(sp)
                    lunch_added = True
                    lunch_idx = len(selected) - 1
                    dist = _calc_dist(s, anchor_for_dist)
                    evidence.append(f"lunch_candidate={sp.get('name')} (supplement, {dist:.1f}km)")
                    break

        # ---- 3순위: profile meals pool (최후 수단) ----
        if not lunch_added:
            lunch_pool = dedupe_candidates(meals + cafes)
            for p in lunch_pool:
                if add_candidate(p):
                    lunch_added = True
                    lunch_idx = len(selected) - 1
                    evidence.append(f"lunch_candidate={p.get('name')} (profile fallback)")
                    break

        if not lunch_added:
            evidence.append("lunch_not_found: all sources exhausted")

        # ------------------------------------------------
        # 3) 나머지: representative 우선, 그 다음 attraction/purpose
        # ------------------------------------------------
        fill_pool = dedupe_candidates(
            representatives + attractions + pools["purpose"]
        )
        for p in fill_pool:
            if len(selected) >= desired_count:
                break
            is_rep = p in representatives
            if purpose == "general" and not is_rep and candidate_has_special_activity(p):
                continue
            add_candidate(p)

        # 4) 그래도 부족하면 특수체험 포함 fallback
        if len(selected) < desired_count:
            fallback_pool = dedupe_candidates(
                attractions + cafes + meals + representatives
            )
            for p in fallback_pool:
                if len(selected) >= desired_count:
                    break
                add_candidate(p)

        if len(selected) < MIN_FULL_DAY_POI_COUNT:
            return [], evidence + [f"selected_too_few={len(selected)}"]

        # ------------------------------------------------
        # 5) 후처리: 연속 meal 방지 → 거리 정렬 → 스케줄 부착
        #    lunch_idx를 reorder_tail_by_distance에 전달해서 lunch slot 고정
        # ------------------------------------------------
        selected = self._avoid_consecutive_meal_like(selected)

        # _avoid_consecutive_meal_like 후 lunch 위치가 바뀔 수 있으므로 재탐색
        lunch_idx = self._find_lunch_idx(selected)

        selected = reorder_tail_by_distance(selected, lunch_idx=lunch_idx)
        selected = self._attach_schedule(selected)

        evidence.append(f"selected={[p.get('name') for p in selected]}")
        return selected, evidence

    def _find_lunch_idx(self, selected: list[dict[str, Any]]) -> int | None:
        """선택된 후보 중 첫 번째 meal-like 후보의 인덱스를 반환. 없으면 None."""
        for i, p in enumerate(selected):
            if profile_poi_is_meal_like(p):
                return i
        return None

    def _is_meal_like(self, p: dict[str, Any]) -> bool:
        return profile_poi_is_meal_like(p)

    def _avoid_consecutive_meal_like(
        self, selected: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(selected) < 3:
            return selected

        out = list(selected)
        for i in range(len(out) - 1):
            if self._is_meal_like(out[i]) and self._is_meal_like(out[i + 1]):
                swap_idx = None
                for j in range(i + 2, len(out)):
                    if not self._is_meal_like(out[j]):
                        swap_idx = j
                        break
                if swap_idx is not None:
                    out[i + 1], out[swap_idx] = out[swap_idx], out[i + 1]
        return out

    def _attach_schedule(self, pois: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        브레이크 타임 정책:
        - 순수 식당이 15:00~17:00에 배치되면 18:00(저녁)으로 자동 이동
        - 카페/시장은 브레이크 타임에도 허용
        """
        out = []
        current = time_to_minutes(DEFAULT_START_TIME)
        BREAK_START = time_to_minutes("15:00")
        BREAK_END = time_to_minutes("17:00")
        DINNER_START = time_to_minutes("18:00")

        for i, p in enumerate(pois):
            q = copy.deepcopy(p)

            ptype = normalize_name(q.get("poi_type") or q.get("type") or "")
            name_n = normalize_name(q.get("name", ""))
            is_pure_restaurant = (
                ptype in {"restaurant", "food"}
                or any(k in name_n for k in ["restaurant", "식당", "칼국수", "냉면", "라멘"])
            ) and ptype not in {"cafe", "market"}

            # 두 번째 후보가 식사형이면 lunch 시간대로 맞춤
            if i == 1 and self._is_meal_like(q):
                current = max(current, time_to_minutes("11:30"))

            # 순수 식당이 브레이크 타임에 걸리면 18:00으로 밀기
            if is_pure_restaurant and BREAK_START <= current < BREAK_END:
                current = DINNER_START

            stay = self._recommended_stay_minutes(q)
            q["name"] = clean_str(q.get("name"))
            q["type"] = clean_str(q.get("poi_type") or q.get("type"), "tourist_spot")
            q["stay_minutes"] = stay
            q["estimated_start_time"] = minutes_to_time(current)
            q["estimated_end_time"] = minutes_to_time(current + stay)
            q["notes"] = self._make_notes(q)
            q["source"] = p.get("source") or "area_profiles_v2_replanner_v3_1"

            current = current + stay + TRAVEL_BUFFER_MINUTES
            out.append(q)

        return out

    def _recommended_stay_minutes(self, p: dict[str, Any]) -> int:
        roles = set(p.get("roles") or [])
        ptype = normalize_name(p.get("poi_type", ""))

        if roles & {"history", "culture"} or ptype in {"history", "culture", "museum"}:
            return 90
        if roles & {"shopping"} or ptype == "shopping":
            return 90
        if roles & {"market", "meal", "cafe"} or ptype in {"restaurant", "food", "cafe", "market"}:
            return 60
        if roles & {"nature", "family"} or ptype in {"park", "nature"}:
            return 75
        return 60

    def _make_notes(self, p: dict[str, Any]) -> str:
        roles = ", ".join(p.get("roles") or [])
        msg = "area profile 기반으로 선택된 권역 적합 후보입니다."
        if roles:
            msg += f" 역할: {roles}."
        return msg


# ============================================================
# Replanner
# ============================================================

class ItineraryReplanner:
    def __init__(self, profile_path: Path | str) -> None:
        self.profile_path = Path(profile_path)
        self.index = AreaProfileIndex.from_path(self.profile_path)
        self.selector = CandidateSelector(self.index)

    def replan(
        self,
        itinerary: dict[str, Any],
        critic_result: Any | None = None,
        user_state: dict[str, Any] | None = None,
        force_days: list[int] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_state = user_state or {}
        rebuilt = copy.deepcopy(itinerary)
        actions: list[ReplanAction] = []
        warnings: list[str] = []
        debug: dict[str, Any] = {"target_days": [], "day_debug": {}}

        # Google Places 식당/카페 (generator가 첨부한 것)
        supplement_restaurants: list[dict[str, Any]] = itinerary.get("supplement_restaurants") or []
        if supplement_restaurants:
            print(f"[replanner] supplement_restaurants {len(supplement_restaurants)}개 로드")

        issues = self._extract_issues(critic_result)
        target_days = self._find_target_days(
            rebuilt, issues, user_state, force_days=force_days
        )
        debug["target_days"] = sorted(target_days)

        changed = False

        # day간 중복 POI 방지: itinerary_used_keys 추적
        itinerary_used_keys: set[str] = set()

        # 재구성 대상이 아닌 day의 POI를 excluded에 미리 포함
        for idx0, day0 in enumerate(rebuilt.get("days", [])):
            day_num0 = extract_day_number(day0, idx0)
            if day_num0 not in target_days:
                for p0 in day0.get("pois", []):
                    k = dedupe_key(public_poi_name(p0))
                    if k:
                        itinerary_used_keys.add(k)

        for idx, day in enumerate(rebuilt.get("days", [])):
            day_num = extract_day_number(day, idx)
            if day_num not in target_days:
                continue

            before_names = [public_poi_name(p) for p in day.get("pois", [])]
            day_issues = [
                i for i in issues
                if safe_int(issue_get(i, "day"), None) == day_num
            ]
            issue_types = {
                clean_str(issue_get(i, "issue_type", "type")).lower()
                for i in day_issues
            }
            heuristic_reasons = self._heuristic_rebuild_reasons(day, user_state)

            target_area = infer_day_target_area(day, user_state, self.index.profile)
            purpose = normalize_purpose(user_state, day)

            day_debug = {
                "before": before_names,
                "issue_types": sorted(issue_types),
                "heuristic_reasons": heuristic_reasons,
                "target_area": target_area,
                "purpose": purpose,
            }

            if not target_area:
                msg = f"Day {day_num}: target area를 추론하지 못해 rebuild를 건너뜀"
                warnings.append(msg)
                actions.append(
                    ReplanAction(
                        action_type="rebuild_day",
                        day=day_num,
                        status="skipped",
                        reason=msg,
                        before=before_names,
                        after=before_names,
                        target_area=None,
                        evidence=[],
                    )
                )
                debug["day_debug"][str(day_num)] = day_debug
                continue

            selected, evidence = self.selector.select_rebuilt_day_pois(
                area_key=target_area,
                purpose=purpose,
                original_day=day,
                issue_types=issue_types,
                excluded_keys=itinerary_used_keys,
                supplement_restaurants=supplement_restaurants,
                user_state=user_state,
            )

            day_debug["candidate_evidence"] = evidence

            if len(selected) < MIN_FULL_DAY_POI_COUNT:
                msg = f"Day {day_num}: 안전한 재구성 후보가 부족해 rebuild 실패"
                warnings.append(msg)
                actions.append(
                    ReplanAction(
                        action_type="rebuild_day",
                        day=day_num,
                        status="failed",
                        reason=msg,
                        before=before_names,
                        after=before_names,
                        target_area=target_area,
                        evidence=evidence,
                    )
                )
                debug["day_debug"][str(day_num)] = day_debug
                continue

            incompatible = [
                p.get("name")
                for p in selected
                if not self.index.is_profile_poi_compatible(p, target_area)
            ]
            if incompatible:
                msg = (
                    f"Day {day_num}: target area와 맞지 않는 후보가 포함되어 "
                    f"rebuild rejected: {incompatible}"
                )
                warnings.append(msg)
                actions.append(
                    ReplanAction(
                        action_type="rebuild_day",
                        day=day_num,
                        status="rejected",
                        reason=msg,
                        before=before_names,
                        after=[p.get("name") for p in selected],
                        target_area=target_area,
                        evidence=evidence,
                    )
                )
                debug["day_debug"][str(day_num)] = day_debug
                continue

            after_names = [p.get("name") for p in selected]
            day["pois"] = [self._profile_poi_to_itinerary_poi(p) for p in selected]
            day["theme"] = self._make_rebuilt_theme(target_area, purpose)
            day["replanned"] = True
            day["replan_target_area"] = target_area
            day["replan_reason"] = sorted(set(list(issue_types) + heuristic_reasons))

            # 이번 day 사용 POI를 전체 집합에 추가 (day간 중복 방지)
            for p in selected:
                k = dedupe_key(clean_str(p.get("name")))
                if k:
                    itinerary_used_keys.add(k)

            changed = True

            actions.append(
                ReplanAction(
                    action_type="rebuild_day",
                    day=day_num,
                    status="accepted",
                    reason=(
                        "structural issue가 남아 area profile representative 기반으로 "
                        "day 전체를 재구성"
                    ),
                    before=before_names,
                    after=after_names,
                    target_area=target_area,
                    evidence=evidence,
                )
            )

            day_debug["after"] = after_names
            debug["day_debug"][str(day_num)] = day_debug

        needs_replan = bool(warnings and not changed)

        result = ReplanResult(
            changed=changed,
            passed_to_critic=changed,
            needs_replan=needs_replan,
            actions=actions,
            warnings=warnings,
            debug=debug,
        )

        rebuilt.setdefault("repair_log", [])
        rebuilt["repair_log"].extend([asdict(a) for a in actions])
        rebuilt["replanner_result"] = {
            "changed": result.changed,
            "passed_to_critic": result.passed_to_critic,
            "needs_replan": result.needs_replan,
            "warnings": result.warnings,
        }

        return rebuilt, self.result_to_dict(result)

    def result_to_dict(self, result: ReplanResult) -> dict[str, Any]:
        return {
            "changed": result.changed,
            "passed_to_critic": result.passed_to_critic,
            "needs_replan": result.needs_replan,
            "actions": [asdict(a) for a in result.actions],
            "warnings": result.warnings,
            "debug": result.debug,
        }

    def _profile_poi_to_itinerary_poi(self, p: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": clean_str(p.get("name")),
            "type": clean_str(p.get("poi_type") or p.get("type"), "tourist_spot"),
            "lat": p.get("lat"),
            "lng": p.get("lng"),
            "stay_minutes": p.get("stay_minutes", 60),
            "estimated_start_time": p.get("estimated_start_time"),
            "estimated_end_time": p.get("estimated_end_time"),
            "notes": p.get("notes") or "area profile 기반으로 선택된 후보입니다.",
            "source": p.get("source", "area_profiles_v2_replanner_v3_1"),
            "profile_cluster": p.get("cluster"),
            "profile_roles": p.get("roles"),
            "representative_score": p.get("representative_score"),
            "google_place_id": p.get("google_place_id"),
            "address": p.get("address"),
        }

    def _make_rebuilt_theme(self, target_area: str, purpose: str) -> str:
        cdata = self.index.get_cluster(target_area)
        label = clean_str(
            cdata.get("label"), target_area.replace("_", " ").title()
        )
        if purpose and purpose != "general":
            return f"{label} · {purpose}"
        return label

    def _extract_issues(self, critic_result: Any | None) -> list[Any]:
        if critic_result is None:
            return []

        d = as_dict(critic_result)
        issues = []
        for key in ["issues", "unresolved_warnings", "warnings"]:
            val = d.get(key)
            if isinstance(val, list):
                issues.extend(val)

        for key in ["issues", "unresolved_warnings", "warnings"]:
            val = getattr(critic_result, key, None)
            if isinstance(val, list):
                issues.extend(val)

        return issues

    def _find_target_days(
        self,
        itinerary: dict[str, Any],
        issues: list[Any],
        user_state: dict[str, Any],
        force_days: list[int] | None = None,
    ) -> set[int]:
        target_days = set(force_days or [])

        for issue in issues:
            day = safe_int(issue_get(issue, "day"), None)
            issue_type = clean_str(issue_get(issue, "issue_type", "type")).lower()
            severity = clean_str(issue_get(issue, "severity"), "").lower()
            status = clean_str(issue_get(issue, "status"), "").lower()

            if day is None:
                continue

            if issue_type in STRUCTURAL_ISSUE_TYPES or issue_type in MEAL_ISSUE_TYPES:
                target_days.add(day)
            elif severity in {"high", "medium"} and status in {"fail", "warning"}:
                target_days.add(day)

        for idx, day in enumerate(itinerary.get("days", [])):
            day_num = extract_day_number(day, idx)
            reasons = self._heuristic_rebuild_reasons(day, user_state)
            if reasons:
                target_days.add(day_num)

        return target_days

    def _heuristic_rebuild_reasons(
        self, day: dict[str, Any], user_state: dict[str, Any]
    ) -> list[str]:
        reasons = []
        target_area = infer_day_target_area(day, user_state, self.index.profile)
        pois = day.get("pois", [])

        if not pois:
            return ["empty_day"]

        if len(pois) < MIN_FULL_DAY_POI_COUNT:
            reasons.append("too_sparse_day")

        for p in pois:
            if candidate_is_vague_or_broad_name(public_poi_name(p), poi_type(p)):
                reasons.append("broad_or_vague_poi")
                break

        if target_area:
            for p in pois:
                prof = self.index.find_profile_poi(public_poi_name(p))
                if prof and not self.index.is_profile_poi_compatible(prof, target_area):
                    reasons.append("off_theme_cluster_poi")
                    break

        return sorted(set(reasons))


# ============================================================
# Public function
# ============================================================

def replan_itinerary(
    itinerary: dict[str, Any],
    critic_result: Any | None = None,
    user_state: dict[str, Any] | None = None,
    profile_path: str | Path | None = None,
    force_days: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_dir = Path(__file__).resolve().parent
    if profile_path is None:
        profile_path = base_dir / "output" / "area_profiles_v2.json"
    else:
        profile_path = Path(profile_path)
        if not profile_path.is_absolute():
            profile_path = base_dir / profile_path

    replanner = ItineraryReplanner(profile_path)
    return replanner.replan(
        itinerary=itinerary,
        critic_result=critic_result,
        user_state=user_state or {},
        force_days=force_days,
    )


# ============================================================
# CLI / standalone sample
# ============================================================

def sample_itinerary() -> dict[str, Any]:
    return {
        "summary": "테스트 2일 일정",
        "days": [
            {
                "day": 1,
                "theme": "홍대 & 연남동",
                "estimated_cost": "$50-100",
                "pois": [
                    {
                        "name": "Hongdae (Hongik University Street) (홍대)",
                        "type": "street",
                        "lat": 37.5563,
                        "lng": 126.9227,
                        "stay_minutes": 90,
                    },
                    {
                        "name": "Yeonnam-dong Cafe Street (연남동 카페거리)",
                        "type": "cafe",
                        "lat": 37.562,
                        "lng": 126.923,
                        "stay_minutes": 60,
                    },
                    {
                        "name": "Gyeongbokgung Palace (경복궁)",
                        "type": "history",
                        "lat": 37.5796,
                        "lng": 126.9770,
                        "stay_minutes": 120,
                    },
                ],
            },
            {
                "day": 2,
                "theme": "강남",
                "estimated_cost": "$80-150",
                "pois": [
                    {
                        "name": "Gangnam (강남)",
                        "type": "street",
                        "lat": 37.4979,
                        "lng": 127.0276,
                        "stay_minutes": 90,
                    },
                ],
            },
        ],
        "sources": [],
    }


def sample_critic_result() -> dict[str, Any]:
    return {
        "passed": False,
        "issues": [
            {
                "day": 1,
                "issue_type": "vague_poi",
                "severity": "medium",
                "poi_name": "Hongdae (Hongik University Street) (홍대)",
            },
            {
                "day": 1,
                "issue_type": "off_theme_cluster_poi",
                "severity": "medium",
                "poi_name": "Gyeongbokgung Palace (경복궁)",
            },
            {
                "day": 2,
                "issue_type": "vague_poi",
                "severity": "medium",
                "poi_name": "Gangnam (강남)",
            },
        ],
    }


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replan structurally problematic itinerary days using area_profiles_v2.json."
    )
    parser.add_argument("--profile", type=str, default="output/area_profiles_v2.json")
    parser.add_argument(
        "--input", type=str, default="",
        help="Input itinerary JSON. If omitted, built-in sample is used.",
    )
    parser.add_argument("--critic", type=str, default="", help="Optional critic result JSON.")
    parser.add_argument("--output", type=str, default="output/replanned_itinerary_v3_1.json")
    parser.add_argument("--purpose", type=str, default="general")
    parser.add_argument("--location", type=str, default="")
    parser.add_argument(
        "--force-days", type=str, default="",
        help="Comma-separated day numbers to force rebuild, e.g. 1,2",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = base_dir / profile_path

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = base_dir / input_path
        itinerary = load_json(input_path)
    else:
        itinerary = sample_itinerary()

    if args.critic:
        critic_path = Path(args.critic)
        if not critic_path.is_absolute():
            critic_path = base_dir / critic_path
        critic_result = load_json(critic_path)
    else:
        critic_result = sample_critic_result()

    force_days = None
    if args.force_days.strip():
        force_days = [int(x.strip()) for x in args.force_days.split(",") if x.strip()]

    user_state = {
        "location": args.location,
        "purpose": args.purpose or "general",
    }

    rebuilt, result = replan_itinerary(
        itinerary=itinerary,
        critic_result=critic_result,
        user_state=user_state,
        profile_path=profile_path,
        force_days=force_days,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rebuilt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[replanner_v3_1] 완료")
    print(f"  changed       : {result['changed']}")
    print(f"  needs_replan  : {result['needs_replan']}")
    print(f"  output        : {output_path}")

    print("\n[actions]")
    for a in result["actions"]:
        print(f"  - Day {a.get('day')} {a.get('status')} area={a.get('target_area')}")
        print(f"    before: {a.get('before')}")
        print(f"    after : {a.get('after')}")
        if a.get("evidence"):
            for e in a["evidence"][:10]:
                print(f"      evidence: {e}")

    if result["warnings"]:
        print("\n[warnings]")
        for w in result["warnings"]:
            print(f"  - {w}")

    print("\n[final itinerary]")
    for day in rebuilt.get("days", []):
        print(f"\nDay {day.get('day')} — {day.get('theme')}")
        for p in day.get("pois", []):
            print(
                f"  - {p.get('estimated_start_time', '')}-{p.get('estimated_end_time', '')} "
                f"{p.get('name')} ({p.get('type')})"
            )


if __name__ == "__main__":
    main()