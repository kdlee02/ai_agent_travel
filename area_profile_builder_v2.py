
"""
area_profile_builder_v2.py

서울 전체 POI DB 기반 area profile 생성기 v2.

왜 v2인가?
---------
v1은 전체 374개 POI를 cluster/role로 나누는 데 성공했지만,
general representative ranking에서 Karaoke, Bowling, Guest House, Station, University 같은
"실존하지만 일반 여행 대표 후보로는 부적절한 POI"가 상위에 올라오는 문제가 있었다.

v2 개선점
---------
1. 서울 전체 POI 대상 유지
   - 홍대/성수/강남 전용 하드코딩이 아니라, 전체 CSV를 읽어 모든 cluster에 동일한 규칙 적용.

2. 대표 후보 품질 필터 강화
   - karaoke, bowling, guest house, station, university 등은 general representative에서 강하게 제외/감점.
   - 단, purpose-specific 후보로는 보존 가능. 예: nightlife이면 karaoke, family/indoor이면 bowling.

3. umbrella area profile 추가
   - gangnam_area = gangnam + apgujeong + seocho + coex/samseong/bongeunsa 계열
   - hongdae_area = hongdae + mapo + mangwon + sinchon
   - jongno_area = jongno + insadong + bukchon + daehaengno
   - yongsan_itaewon_area = yongsan + itaewon
   - Replanner가 "강남", "홍대", "종로" 같은 넓은 요청을 받았을 때,
     너무 좁은 좌표 cluster만 보지 않도록 하기 위함.

4. dual-role anchor 일반화
   - market, cafe street, food street, alley, shopping street 등은 meal/cafe뿐 아니라
     day anchor/attraction 역할도 가능.

5. Google Places fallback 준비
   - 기본 실행에서는 Google API를 호출하지 않음.
   - 대신 output/area_profile_google_gap_queries.csv를 생성해서 어떤 cluster/role이 부족한지와
     어떤 Google query로 보강하면 되는지 제공.
   - --use-google-fallback 옵션을 켜면 Google Places Text Search를 호출해
     output/area_profile_google_candidates.csv에 후보만 저장한다.
   - main area_profiles_v2.json에는 Google 결과를 자동 병합하지 않는다.
     이유: API 결과는 실행 시점에 따라 바뀌므로, 검토/정제 후 merge하는 편이 안전하다.

실행
----
cd "C:\\Users\\leechaewon\\Desktop\\홍익대 자료\\시스템분석\\project\\ai_agent_travel"

python area_profile_builder_v2.py

선택 실행
---------
python area_profile_builder_v2.py --input output/poi_master_step3_enhanced_v2.csv
python area_profile_builder_v2.py --use-google-fallback --google-max-results 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 서울 cluster center
# ============================================================

SEOUL_CLUSTERS: dict[str, tuple[float, float]] = {
    "hongdae": (37.5563, 126.9227),
    "mapo": (37.5479, 126.9130),
    "mangwon": (37.5567, 126.9055),
    "sinchon": (37.5596, 126.9373),
    "gangnam": (37.4979, 127.0276),
    "samseong_coex": (37.5118, 127.0592),
    "seongsu": (37.5447, 127.0558),
    "jongno": (37.5729, 126.9794),
    "insadong": (37.5741, 126.9861),
    "myeongdong": (37.5636, 126.9857),
    "itaewon": (37.5347, 126.9946),
    "yongsan": (37.5299, 126.9649),
    "dongdaemun": (37.5666, 127.0097),
    "yeouido": (37.5217, 126.9244),
    "jamsil": (37.5133, 127.1028),
    "bukchon": (37.5826, 126.9836),
    "daehaengno": (37.5810, 127.0020),
    "seocho": (37.4837, 127.0324),
    "apgujeong": (37.5271, 127.0286),
}

DEFAULT_CLUSTER_RADIUS_KM = 3.8

# umbrella profile은 base cluster들을 묶은 "검색/재구성용 권역"이다.
UMBRELLA_AREAS: dict[str, dict[str, Any]] = {
    "hongdae_area": {
        "members": {"hongdae", "mapo", "mangwon", "sinchon"},
        "aliases": {"hongdae", "hongik", "yeonnam", "mangwon", "mangnidan", "gyeongui", "홍대", "연남", "망원", "망리단", "경의선"},
        "label": "Hongdae / Yeonnam / Mangwon",
    },
    "gangnam_area": {
        "members": {"gangnam", "samseong_coex", "apgujeong", "seocho"},
        "aliases": {"gangnam", "coex", "samseong", "starfield", "bongeunsa", "garosu", "apgujeong", "dosan", "sinsa", "seolleung", "강남", "코엑스", "삼성", "봉은사", "가로수", "압구정", "도산", "신사", "선릉"},
        "label": "Gangnam / COEX / Apgujeong / Seocho",
    },
    "jongno_area": {
        "members": {"jongno", "insadong", "bukchon", "daehaengno"},
        "aliases": {"jongno", "insadong", "bukchon", "gyeongbokgung", "gwanghwamun", "samcheong", "ikseon", "daehangno", "종로", "인사동", "북촌", "경복궁", "광화문", "삼청", "익선", "대학로"},
        "label": "Jongno / Insadong / Bukchon",
    },
    "yongsan_itaewon_area": {
        "members": {"yongsan", "itaewon"},
        "aliases": {"yongsan", "itaewon", "hannam", "hybe", "leeum", "용산", "이태원", "한남", "하이브", "리움"},
        "label": "Yongsan / Itaewon / Hannam",
    },
    "myeongdong_euljiro_area": {
        "members": {"myeongdong", "dongdaemun"},
        "aliases": {"myeongdong", "euljiro", "namdaemun", "dongdaemun", "명동", "을지로", "남대문", "동대문"},
        "label": "Myeongdong / Euljiro / Dongdaemun",
    },
}

COMPATIBLE_CLUSTERS: dict[str, set[str]] = {
    "hongdae": {"hongdae", "mapo", "mangwon", "sinchon"},
    "mapo": {"mapo", "hongdae", "mangwon", "sinchon"},
    "mangwon": {"mangwon", "mapo", "hongdae"},
    "sinchon": {"sinchon", "hongdae", "mapo"},
    "gangnam": {"gangnam", "samseong_coex", "seocho", "apgujeong"},
    "samseong_coex": {"samseong_coex", "gangnam", "apgujeong"},
    "seocho": {"seocho", "gangnam", "samseong_coex"},
    "apgujeong": {"apgujeong", "gangnam", "samseong_coex"},
    "seongsu": {"seongsu"},
    "jongno": {"jongno", "insadong", "bukchon", "daehaengno"},
    "insadong": {"insadong", "jongno", "bukchon"},
    "bukchon": {"bukchon", "jongno", "insadong"},
    "daehaengno": {"daehaengno", "jongno"},
    "myeongdong": {"myeongdong", "jongno", "dongdaemun"},
    "itaewon": {"itaewon", "yongsan"},
    "yongsan": {"yongsan", "itaewon"},
    "dongdaemun": {"dongdaemun", "myeongdong", "daehaengno"},
    "yeouido": {"yeouido"},
    "jamsil": {"jamsil"},
    "other": {"other"},
    "unknown": {"unknown"},
}


# ============================================================
# Role policy
# ============================================================

ROLE_ORDER = [
    "attraction",
    "meal",
    "cafe",
    "market",
    "shopping",
    "history",
    "culture",
    "nature",
    "kpop",
    "family",
    "nightlife",
    "beauty",
    "indoor_leisure",
    "transport",
    "education",
    "accommodation",
]

TYPE_TO_ROLES: dict[str, set[str]] = {
    "restaurant": {"meal"},
    "food": {"meal"},
    "cafe": {"cafe"},
    "market": {"market", "meal", "shopping", "attraction"},
    "shopping": {"shopping", "attraction"},
    "street": {"attraction", "shopping"},
    "tourist_spot": {"attraction"},
    "landmark": {"attraction"},
    "culture": {"culture", "attraction"},
    "history": {"history", "culture", "attraction"},
    "museum": {"history", "culture", "attraction"},
    "park": {"nature", "family", "attraction"},
    "nature": {"nature", "family", "attraction"},
    "kpop_landmark": {"kpop", "shopping", "culture", "attraction"},
    "nightlife": {"nightlife"},
    "beauty": {"beauty", "shopping"},
    "library": {"culture", "attraction"},
    "station": {"transport"},
    "university": {"education"},
    "accommodation": {"accommodation"},
    "guest_house": {"accommodation"},
    "bowling": {"indoor_leisure", "family"},
    "karaoke": {"nightlife"},
}

PURPOSE_TO_ROLES: dict[str, set[str]] = {
    "food": {"meal"},
    "restaurant": {"meal"},
    "cafe": {"cafe"},
    "cafe_hopping": {"cafe", "attraction"},
    "shopping": {"shopping", "attraction"},
    "market": {"market", "meal", "shopping", "attraction"},
    "history": {"history", "culture", "attraction"},
    "culture": {"culture", "attraction"},
    "nature": {"nature", "family", "attraction"},
    "family": {"family", "attraction"},
    "kpop": {"kpop", "shopping", "culture", "attraction"},
    "beauty": {"beauty", "shopping"},
    "nightlife": {"nightlife"},
    "general": {"attraction"},
    "transport": {"transport"},
    "education": {"education"},
}

DUAL_ROLE_ANCHOR_PATTERNS = [
    "market", "시장",
    "street", "거리", "길", "골목", "alley", "road",
    "cafe street", "카페거리", "카페 골목", "카페골목",
    "food street", "음식문화거리", "먹자골목",
    "shopping street", "쇼핑거리",
    "forest park", "숲길", "공원",
]

# general day 대표 후보에서 내릴 것.
GENERAL_REPRESENTATIVE_DOWNRANK_PATTERNS = [
    "karaoke", "노래방",
    "bowling", "볼링",
    "guest house", "guesthouse", "게스트하우스",
    "station", "역",
    "university", "univ", "대학교", "대학",
    "airport", "공항",
]

# 완전 제외가 아니라 general representative에서만 강하게 제외.
# 목적이 nightlife/transport/education/accommodation이면 role 후보로는 유지된다.
GENERAL_REPRESENTATIVE_EXCLUDE_PATTERNS = [
    "karaoke", "노래방",
    "guest house", "guesthouse", "게스트하우스",
    "station", "역",
    "university", "univ", "대학교", "대학",
]

# 관광 대표성을 높이는 패턴.
TOURIST_ANCHOR_BOOST_PATTERNS = [
    "palace", "궁",
    "temple", "사", "절",
    "museum", "미술관", "박물관",
    "park", "공원", "forest", "숲",
    "market", "시장",
    "street", "거리", "길", "골목",
    "village", "마을",
    "mall", "몰", "shopping", "쇼핑",
    "library", "도서관",
    "square", "광장",
    "observatory", "전망대",
    "tower", "타워",
    "hanok", "한옥",
    "hangang", "한강",
    "coex", "코엑스",
    "starfield", "스타필드",
    "bongeunsa", "봉은사",
    "garosu", "가로수",
    "dosan", "도산",
    "seoul forest", "서울숲",
]

BROAD_AREA_EXACT_ALIASES = {
    "hongdae", "hongik univ", "hongik university", "hongik university street",
    "gangnam", "gangnam station",
    "seongsu", "seongsu dong", "seongsu-dong",
    "yongsan", "mangwon dong", "mangwon-dong", "yeonnam dong", "yeonnam-dong",
    "jongno", "myeongdong", "itaewon", "sinchon", "mapo", "jamsil",
    "bukchon", "insa dong", "in sa dong", "insadong", "ikseon dong", "ikseondong",
    "hannam dong", "hannam-dong", "euljiro area", "gwanghwamun area",
    "홍대", "홍익대", "홍익대학교", "강남", "강남역", "성수", "성수동",
    "용산", "망원동", "연남동", "종로", "명동", "이태원", "신촌", "마포", "잠실",
    "북촌", "인사동", "익선동", "한남동", "을지로", "광화문",
}


@dataclass
class ProfilePOI:
    poi_id: str
    name: str
    poi_type: str
    cluster: str
    lat: float | None
    lng: float | None
    roles: list[str]
    primary_role: str
    representative_score: float
    role_scores: dict[str, float]
    general_representative_ok: bool
    place_confidence: str
    purpose_tags: list[str]
    google_place_id: str
    address: str
    opening_hours: Any
    is_vague_or_broad: bool
    is_dual_role_anchor: bool
    is_general_downrank: bool
    debug_reasons: list[str]


# ============================================================
# Utilities
# ============================================================

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
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


def contains_any(text: str, patterns: list[str] | set[str]) -> bool:
    n = normalize_name(text)
    for p in patterns:
        pp = normalize_name(p)
        if pp and pp in n:
            return True
    return False


def safe_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def confidence_to_score(value: Any, default: float = 0.55) -> float:
    v = clean_str(value).lower()
    if v == "high":
        return 1.0
    if v == "medium":
        return 0.75
    if v == "low":
        return 0.45
    return default


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if is_missing(value):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def json_list(value: Any) -> list[str]:
    v = safe_json_loads(value, default=[])
    if isinstance(v, list):
        return [str(x) for x in v if not is_missing(x)]
    if is_missing(value):
        return []
    return [str(value)]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def assign_cluster(lat: float | None, lng: float | None, radius_km: float = DEFAULT_CLUSTER_RADIUS_KM) -> tuple[str, float | None]:
    if lat is None or lng is None:
        return "unknown", None

    best = "other"
    best_dist = float("inf")
    for name, (clat, clng) in SEOUL_CLUSTERS.items():
        d = haversine_km(lat, lng, clat, clng)
        if d < best_dist:
            best_dist = d
            best = name

    if best_dist <= radius_km:
        return best, round(best_dist, 3)
    return "other", round(best_dist, 3)


def row_text_blob(row: dict[str, Any]) -> str:
    fields = [
        row.get("poi_name", ""),
        row.get("poi_type", ""),
        row.get("address_en", ""),
        row.get("address_ko", ""),
        row.get("google_types", ""),
        row.get("google_editorial_summary", ""),
        row.get("purpose_tags", ""),
        row.get("purpose_evidence", ""),
        row.get("label_evidence", ""),
        row.get("foreigner_tip_ko", ""),
        row.get("foreigner_tip_en", ""),
    ]
    return " ".join(str(x) for x in fields if not is_missing(x))


# ============================================================
# Classification
# ============================================================

def is_dual_role_anchor(row: dict[str, Any]) -> bool:
    name = normalize_name(row.get("poi_name", ""))
    poi_type = normalize_name(row.get("poi_type", ""))
    blob = normalize_name(row_text_blob(row))

    if poi_type in {"market", "street"}:
        return True

    if poi_type in {"shopping"} and any(k in blob for k in ["market", "street", "거리", "길", "골목", "alley"]):
        return True

    if poi_type == "cafe" and any(k in blob for k in ["cafe street", "카페거리", "카페 골목", "카페골목", "street", "거리", "길", "골목"]):
        return True

    return any(normalize_name(p) in blob for p in DUAL_ROLE_ANCHOR_PATTERNS)


def is_vague_broad_name(row: dict[str, Any]) -> bool:
    name = normalize_name(row.get("poi_name", ""))
    poi_type = normalize_name(row.get("poi_type", ""))

    if not name:
        return True

    concrete_markers = [
        "performance", "market", "park", "museum", "palace", "temple", "library",
        "cafe street", "food street", "shopping street", "shopping center", "mall",
        "square", "village", "trail", "alley", "street performance", "observatory",
        "거리공연", "시장", "공원", "궁", "사원", "절", "도서관", "광장", "마을", "길", "골목",
    ]
    if any(normalize_name(m) in name for m in concrete_markers):
        return False

    if name in BROAD_AREA_EXACT_ALIASES:
        return True

    if poi_type in {"area", "district", "neighborhood"}:
        return True

    tokens = name.split()
    if len(tokens) <= 3:
        for alias in BROAD_AREA_EXACT_ALIASES:
            if normalize_name(alias) and normalize_name(alias) in name:
                return True

    return False


def is_general_downrank_candidate(row: dict[str, Any]) -> bool:
    blob = row_text_blob(row)
    return contains_any(blob, GENERAL_REPRESENTATIVE_DOWNRANK_PATTERNS)


def is_general_excluded_candidate(row: dict[str, Any]) -> bool:
    blob = row_text_blob(row)
    return contains_any(blob, GENERAL_REPRESENTATIVE_EXCLUDE_PATTERNS)


def infer_roles(row: dict[str, Any]) -> tuple[set[str], list[str]]:
    roles: set[str] = set()
    reasons: list[str] = []

    poi_type = normalize_name(row.get("poi_type", ""))
    if poi_type in TYPE_TO_ROLES:
        roles |= TYPE_TO_ROLES[poi_type]
        reasons.append(f"type:{poi_type}->{sorted(TYPE_TO_ROLES[poi_type])}")

    for tag in json_list(row.get("purpose_tags")):
        tag_norm = normalize_name(tag).replace(" ", "_")
        tag_candidates = {tag_norm, tag_norm.replace("_", " "), str(tag).strip().lower()}
        for t in tag_candidates:
            if t in PURPOSE_TO_ROLES:
                roles |= PURPOSE_TO_ROLES[t]
                reasons.append(f"purpose:{t}->{sorted(PURPOSE_TO_ROLES[t])}")

    blob = normalize_name(row_text_blob(row))

    if any(k in blob for k in ["k pop", "kpop", "케이팝", "한류", "idol", "아이돌"]):
        roles |= {"kpop", "shopping", "culture", "attraction"}
        reasons.append("keyword:kpop")

    if any(k in blob for k in ["palace", "temple", "shrine", "hanok", "궁", "사찰", "절", "한옥", "royal tomb"]):
        roles |= {"history", "culture", "attraction"}
        reasons.append("keyword:history_culture")

    if any(k in blob for k in ["park", "forest", "river", "hangang", "공원", "숲", "한강"]):
        roles |= {"nature", "family", "attraction"}
        reasons.append("keyword:nature")

    if any(k in blob for k in ["market", "food street", "restaurant", "시장", "먹자", "음식"]):
        roles |= {"meal", "market", "attraction"}
        reasons.append("keyword:market_food")

    if any(k in blob for k in ["karaoke", "노래방"]):
        roles |= {"nightlife"}
        reasons.append("keyword:nightlife_karaoke")

    if any(k in blob for k in ["bowling", "볼링"]):
        roles |= {"indoor_leisure", "family"}
        reasons.append("keyword:indoor_leisure")

    if any(k in blob for k in ["guest house", "guesthouse", "게스트하우스", "hotel", "호텔"]):
        roles |= {"accommodation"}
        reasons.append("keyword:accommodation")

    if any(k in blob for k in ["station", "역"]):
        roles |= {"transport"}
        reasons.append("keyword:transport")

    if any(k in blob for k in ["university", "univ", "대학교", "대학"]):
        roles |= {"education"}
        reasons.append("keyword:education")

    if is_dual_role_anchor(row):
        roles.add("attraction")
        reasons.append("dual_role_anchor:also_attraction")

    if not roles:
        roles.add("attraction")
        reasons.append("fallback:attraction")

    return roles, reasons


def role_score(row: dict[str, Any], role: str, cluster_distance_km: float | None) -> tuple[float, list[str]]:
    reasons: list[str] = []
    roles, role_reasons = infer_roles(row)

    score = 0.0

    if role in roles:
        score += 0.42
        reasons.append("role_match:+0.42")
    elif role == "attraction" and is_dual_role_anchor(row):
        score += 0.38
        reasons.append("dual_role_attraction:+0.38")
    else:
        score += 0.08
        reasons.append("weak_role:+0.08")

    place_conf = confidence_to_score(row.get("place_confidence"), 0.55)
    score += 0.20 * place_conf
    reasons.append(f"place_conf:{place_conf:.2f}*0.20")

    if not is_missing(row.get("google_place_id")):
        score += 0.08
        reasons.append("google_place_id:+0.08")

    if not is_missing(row.get("opening_hours")):
        score += 0.06
        reasons.append("opening_hours:+0.06")

    tags = json_list(row.get("purpose_tags"))
    if tags:
        score += min(0.08, 0.02 * len(tags))
        reasons.append(f"purpose_tags:{len(tags)}")

    if cluster_distance_km is not None:
        if cluster_distance_km <= 0.8:
            score += 0.08
            reasons.append("near_center:+0.08")
        elif cluster_distance_km <= 1.8:
            score += 0.05
            reasons.append("mid_center:+0.05")
        elif cluster_distance_km <= DEFAULT_CLUSTER_RADIUS_KM:
            score += 0.02
            reasons.append("inside_cluster:+0.02")
        else:
            score -= 0.10
            reasons.append("far_from_cluster:-0.10")

    if is_vague_broad_name(row):
        score -= 0.35
        reasons.append("vague_broad:-0.35")

    if role == "attraction" and is_general_downrank_candidate(row):
        score -= 0.20
        reasons.append("general_downrank_for_attraction:-0.20")

    if role == "attraction" and contains_any(row_text_blob(row), TOURIST_ANCHOR_BOOST_PATTERNS):
        score += 0.10
        reasons.append("tourist_anchor_boost:+0.10")

    poi_type = normalize_name(row.get("poi_type", ""))

    if role == "meal" and poi_type in {"restaurant", "food", "market"}:
        score += 0.10
        reasons.append("meal_type_bonus:+0.10")

    if role == "cafe" and poi_type == "cafe":
        score += 0.10
        reasons.append("cafe_type_bonus:+0.10")

    if role == "attraction" and poi_type in {"market", "street", "shopping", "park", "culture", "history", "museum", "tourist_spot", "library"}:
        score += 0.08
        reasons.append("attraction_type_bonus:+0.08")

    if role == "attraction" and is_dual_role_anchor(row):
        score += 0.08
        reasons.append("dual_role_anchor_bonus:+0.08")

    # 목적별 후보는 유지하되 해당 purpose 외 대표 후보로 올라가는 것을 방지
    if role not in {"nightlife", "transport", "education", "accommodation", "indoor_leisure"}:
        if is_general_excluded_candidate(row):
            score -= 0.18
            reasons.append("general_excluded_role_penalty:-0.18")

    score = max(0.0, min(1.0, score))
    return round(score, 4), reasons + role_reasons


def representative_score(row: dict[str, Any], cluster_distance_km: float | None) -> tuple[float, list[str], bool]:
    roles, reasons = infer_roles(row)

    rep_roles = {"attraction", "market", "shopping", "culture", "history", "nature", "kpop", "family"}
    role_fit = 0.75 if roles & rep_roles else 0.35

    if is_dual_role_anchor(row):
        role_fit = max(role_fit, 0.82)
        reasons.append("dual_role_anchor_representative")

    if contains_any(row_text_blob(row), TOURIST_ANCHOR_BOOST_PATTERNS):
        role_fit = min(1.0, role_fit + 0.08)
        reasons.append("tourist_anchor_pattern_boost")

    place_fit = confidence_to_score(row.get("place_confidence"), 0.55)
    source_fit = 1.0 if not is_missing(row.get("google_place_id")) else 0.6

    if cluster_distance_km is None:
        distance_fit = 0.55
    elif cluster_distance_km <= 0.8:
        distance_fit = 1.0
    elif cluster_distance_km <= 1.8:
        distance_fit = 0.82
    elif cluster_distance_km <= DEFAULT_CLUSTER_RADIUS_KM:
        distance_fit = 0.65
    else:
        distance_fit = 0.25

    detail_fit = 0.25 if is_vague_broad_name(row) else 1.0

    general_ok = True

    if is_vague_broad_name(row):
        general_ok = False
        reasons.append("general_rep_excluded:vague_broad")

    if is_general_excluded_candidate(row):
        general_ok = False
        reasons.append("general_rep_excluded:general_excluded_pattern")

    if is_general_downrank_candidate(row):
        role_fit *= 0.55
        reasons.append("general_downrank:role_fit*0.55")

    score = (
        0.34 * role_fit
        + 0.22 * place_fit
        + 0.16 * source_fit
        + 0.18 * distance_fit
        + 0.10 * detail_fit
    )

    if not general_ok:
        score = min(score, 0.49)

    debug = [
        f"role_fit={role_fit:.2f}",
        f"place_fit={place_fit:.2f}",
        f"source_fit={source_fit:.2f}",
        f"distance_fit={distance_fit:.2f}",
        f"detail_fit={detail_fit:.2f}",
        f"general_ok={general_ok}",
        *reasons,
    ]

    return round(max(0.0, min(1.0, score)), 4), debug, general_ok


def primary_role_from_scores(role_scores: dict[str, float]) -> str:
    if not role_scores:
        return "attraction"
    return sorted(
        role_scores.items(),
        key=lambda kv: (-kv[1], ROLE_ORDER.index(kv[0]) if kv[0] in ROLE_ORDER else 999)
    )[0][0]


# ============================================================
# Build
# ============================================================

def discover_default_input(base_dir: Path) -> Path:
    candidates = [
        base_dir / "output" / "poi_master_step3_enhanced_v2.csv",
        base_dir / "output" / "poi_master_step3_enhanced.csv",
        base_dir / "output" / "poi_master_step3.csv",
        base_dir / "output" / "poi_master.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("POI CSV를 찾지 못했습니다. --input으로 지정하세요.")


def poi_public(p: ProfilePOI, role: str | None = None) -> dict[str, Any]:
    return {
        "poi_id": p.poi_id,
        "name": p.name,
        "poi_type": p.poi_type,
        "cluster": p.cluster,
        "lat": p.lat,
        "lng": p.lng,
        "roles": p.roles,
        "primary_role": p.primary_role,
        "representative_score": p.representative_score,
        "role_score": p.role_scores.get(role, None) if role else None,
        "general_representative_ok": p.general_representative_ok,
        "place_confidence": p.place_confidence,
        "purpose_tags": p.purpose_tags,
        "google_place_id": p.google_place_id,
        "address": p.address,
        "is_vague_or_broad": p.is_vague_or_broad,
        "is_dual_role_anchor": p.is_dual_role_anchor,
        "is_general_downrank": p.is_general_downrank,
        "debug_reasons": p.debug_reasons[:10],
    }


def make_cluster_profile(cluster: str, cluster_pois: list[ProfilePOI], top_k: int) -> dict[str, Any]:
    role_candidates = {}
    for role in ROLE_ORDER:
        candidates = [
            p for p in cluster_pois
            if role in p.role_scores
            and not p.is_vague_or_broad
        ]
        candidates.sort(
            key=lambda p: (
                p.role_scores.get(role, 0),
                p.representative_score,
                confidence_to_score(p.place_confidence, 0.55),
            ),
            reverse=True,
        )
        role_candidates[role] = [poi_public(p, role=role) for p in candidates[:top_k]]

    representatives = [
        p for p in cluster_pois
        if p.general_representative_ok
        and not p.is_vague_or_broad
        and p.representative_score >= 0.50
    ]
    representatives.sort(
        key=lambda p: (
            p.representative_score,
            p.role_scores.get("attraction", 0),
            max(p.role_scores.values()) if p.role_scores else 0,
            confidence_to_score(p.place_confidence, 0.55),
        ),
        reverse=True,
    )

    dual_role_anchors = [
        p for p in cluster_pois
        if p.is_dual_role_anchor and not p.is_vague_or_broad
    ]
    dual_role_anchors.sort(key=lambda p: (p.representative_score, max(p.role_scores.values())), reverse=True)

    vague_or_broad = [p for p in cluster_pois if p.is_vague_or_broad]

    downranked = [
        p for p in cluster_pois
        if p.is_general_downrank or not p.general_representative_ok
    ]

    return {
        "center": SEOUL_CLUSTERS.get(cluster),
        "compatible_clusters": sorted(COMPATIBLE_CLUSTERS.get(cluster, {cluster})),
        "stats": {
            "total_pois": len(cluster_pois),
            "valid_representatives": len(representatives),
            "dual_role_anchors": len(dual_role_anchors),
            "vague_or_broad": len(vague_or_broad),
            "general_downranked": len(downranked),
        },
        "representative_pois": [poi_public(p) for p in representatives[:top_k]],
        "role_candidates": role_candidates,
        "dual_role_anchors": [poi_public(p) for p in dual_role_anchors[:top_k]],
        "vague_or_broad_pois": [poi_public(p) for p in vague_or_broad[:top_k]],
        "downranked_or_excluded_pois": [poi_public(p) for p in downranked[:top_k]],
    }


def build_profile_pois(input_path: Path, cluster_radius_km: float) -> list[ProfilePOI]:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False).fillna("")
    pois: list[ProfilePOI] = []

    for idx, row_obj in df.iterrows():
        row = row_obj.to_dict()

        name = clean_str(row.get("poi_name") or row.get("name") or f"poi_{idx}")
        poi_type = clean_str(row.get("poi_type"), "tourist_spot")
        lat = safe_float(row.get("lat"))
        lng = safe_float(row.get("lng"))
        cluster, dist = assign_cluster(lat, lng, radius_km=cluster_radius_km)

        roles, role_reasons = infer_roles(row)
        role_scores = {}
        role_debug = []

        for role in ROLE_ORDER:
            s, rs = role_score(row, role, dist)
            if role in roles or s >= 0.50:
                role_scores[role] = s
                role_debug.extend([f"{role}:{x}" for x in rs[:3]])

        if "attraction" not in role_scores:
            s, rs = role_score(row, "attraction", dist)
            if s >= 0.35:
                role_scores["attraction"] = s
                role_debug.extend([f"attraction:{x}" for x in rs[:3]])

        primary = primary_role_from_scores(role_scores)
        rep_score, rep_debug, general_ok = representative_score(row, dist)

        p = ProfilePOI(
            poi_id=clean_str(row.get("poi_id"), f"row_{idx}"),
            name=name,
            poi_type=poi_type,
            cluster=cluster,
            lat=lat,
            lng=lng,
            roles=sorted(role_scores.keys(), key=lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 999),
            primary_role=primary,
            representative_score=rep_score,
            role_scores=role_scores,
            general_representative_ok=general_ok,
            place_confidence=clean_str(row.get("place_confidence"), "unknown"),
            purpose_tags=json_list(row.get("purpose_tags")),
            google_place_id=clean_str(row.get("google_place_id")),
            address=clean_str(row.get("address_en") or row.get("address_ko")),
            opening_hours=safe_json_loads(row.get("opening_hours"), default=None),
            is_vague_or_broad=is_vague_broad_name(row),
            is_dual_role_anchor=is_dual_role_anchor(row),
            is_general_downrank=is_general_downrank_candidate(row),
            debug_reasons=rep_debug + role_debug[:12],
        )
        pois.append(p)

    return pois


def belongs_to_umbrella(p: ProfilePOI, umbrella: dict[str, Any]) -> bool:
    members = set(umbrella.get("members", []))
    aliases = set(umbrella.get("aliases", []))

    if p.cluster in members:
        return True

    blob = normalize_name(" ".join([p.name, p.poi_type, p.address, " ".join(p.purpose_tags)]))
    for alias in aliases:
        a = normalize_name(alias)
        if a and a in blob:
            return True
    return False


def make_google_gap_queries(profile: dict[str, Any], min_reps: int = 4, min_attractions: int = 3, min_meals: int = 1) -> list[dict[str, Any]]:
    rows = []

    for cluster, cdata in profile["clusters"].items():
        reps = cdata.get("representative_pois", [])
        attraction = cdata.get("role_candidates", {}).get("attraction", [])
        meal = cdata.get("role_candidates", {}).get("meal", [])

        label = cluster.replace("_", " ")
        missing = []
        if len(reps) < min_reps:
            missing.append("representative")
        if len(attraction) < min_attractions:
            missing.append("attraction")
        if len(meal) < min_meals:
            missing.append("meal")

        for role in missing:
            if role == "meal":
                q = f"{label} Seoul restaurant market food street"
            else:
                q = f"{label} Seoul tourist attractions"
            rows.append({
                "cluster": cluster,
                "missing_role": role,
                "current_representatives": len(reps),
                "current_attractions": len(attraction),
                "current_meals": len(meal),
                "query": q,
                "note": "Review Google results before merging into POI master.",
            })

    return rows


def build_profiles(
    input_path: Path,
    output_path: Path,
    csv_output_path: Path | None = None,
    google_gap_output_path: Path | None = None,
    top_k: int = 12,
    cluster_radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
) -> dict[str, Any]:
    pois = build_profile_pois(input_path, cluster_radius_km=cluster_radius_km)

    cluster_stats: dict[str, int] = {}
    for p in pois:
        cluster_stats[p.cluster] = cluster_stats.get(p.cluster, 0) + 1

    clusters: dict[str, Any] = {}
    for cluster in sorted(cluster_stats.keys()):
        cluster_pois = [p for p in pois if p.cluster == cluster]
        clusters[cluster] = make_cluster_profile(cluster, cluster_pois, top_k=top_k)

    # umbrella profiles
    for umbrella_name, umbrella in UMBRELLA_AREAS.items():
        umbrella_pois = [p for p in pois if belongs_to_umbrella(p, umbrella)]
        clusters[umbrella_name] = make_cluster_profile(umbrella_name, umbrella_pois, top_k=top_k)
        clusters[umbrella_name]["umbrella"] = True
        clusters[umbrella_name]["label"] = umbrella.get("label", umbrella_name)
        clusters[umbrella_name]["members"] = sorted(umbrella.get("members", []))
        clusters[umbrella_name]["aliases"] = sorted(umbrella.get("aliases", []))

    profile = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_csv": str(input_path),
            "total_pois": len(pois),
            "cluster_radius_km": cluster_radius_km,
            "top_k": top_k,
            "version": "v2",
            "purpose": "Seoul-wide area profiles for itinerary replanning",
            "design_notes": [
                "General representative ranking excludes/downranks karaoke, guest house, station, university, broad area names.",
                "Purpose-specific role candidates are preserved even when general representative is downranked.",
                "Umbrella profiles prevent wide user areas such as Gangnam from becoming too narrow.",
                "Google fallback is prepared as query gaps, not automatically merged into the main profile.",
            ],
        },
        "cluster_stats": cluster_stats,
        "clusters": clusters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    if csv_output_path:
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in pois:
            rows.append({
                "poi_id": p.poi_id,
                "poi_name": p.name,
                "poi_type": p.poi_type,
                "cluster": p.cluster,
                "lat": p.lat,
                "lng": p.lng,
                "roles": "|".join(p.roles),
                "primary_role": p.primary_role,
                "representative_score": p.representative_score,
                "general_representative_ok": p.general_representative_ok,
                "place_confidence": p.place_confidence,
                "purpose_tags": "|".join(p.purpose_tags),
                "google_place_id": p.google_place_id,
                "is_vague_or_broad": p.is_vague_or_broad,
                "is_dual_role_anchor": p.is_dual_role_anchor,
                "is_general_downrank": p.is_general_downrank,
                "debug_reasons": " || ".join(p.debug_reasons[:14]),
            })
        pd.DataFrame(rows).to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    if google_gap_output_path:
        google_gap_output_path.parent.mkdir(parents=True, exist_ok=True)
        gap_rows = make_google_gap_queries(profile)
        pd.DataFrame(gap_rows).to_csv(google_gap_output_path, index=False, encoding="utf-8-sig")

    return profile


# ============================================================
# Optional Google fallback
# ============================================================

def load_env_file(base_dir: Path) -> None:
    env_path = base_dir / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def google_text_search(query: str, api_key: str, max_results: int = 3, sleep_sec: float = 0.2) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "key": api_key,
        "language": "en",
        "region": "kr",
    }
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return [{"query": query, "error": str(e)}]

    time.sleep(max(0.0, sleep_sec))

    results = []
    for r in data.get("results", [])[:max_results]:
        loc = r.get("geometry", {}).get("location", {})
        results.append({
            "query": query,
            "name": r.get("name", ""),
            "place_id": r.get("place_id", ""),
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "rating": r.get("rating"),
            "user_ratings_total": r.get("user_ratings_total"),
            "types": "|".join(r.get("types", [])),
            "formatted_address": r.get("formatted_address", ""),
            "business_status": r.get("business_status", ""),
            "source": "google_places_text_search",
        })
    return results


def run_google_fallback(gap_csv: Path, output_csv: Path, base_dir: Path, max_results: int, sleep_sec: float) -> None:
    load_env_file(base_dir)
    api_key = (
        os.environ.get("GOOGLE_PLACES_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("MAPS_API_KEY")
    )
    if not api_key:
        print("[Google fallback] API key 없음 — GOOGLE_PLACES_API_KEY 또는 GOOGLE_API_KEY를 .env에 넣어야 합니다.")
        return

    if not gap_csv.exists():
        print(f"[Google fallback] gap csv 없음: {gap_csv}")
        return

    gaps = pd.read_csv(gap_csv, dtype=str, keep_default_na=False).fillna("")
    all_rows = []
    for _, row in gaps.iterrows():
        query = row.get("query", "")
        if not query:
            continue
        print(f"[Google fallback] {query}")
        results = google_text_search(query, api_key=api_key, max_results=max_results, sleep_sec=sleep_sec)
        for r in results:
            merged = dict(row)
            merged.update(r)
            all_rows.append(merged)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Google fallback] write: {output_csv} ({len(all_rows)} rows)")


# ============================================================
# Print
# ============================================================

def print_summary(profile: dict[str, Any]) -> None:
    print("\n[area_profile_builder_v2] 생성 완료")
    print(f"  source: {profile['metadata']['source_csv']}")
    print(f"  total_pois: {profile['metadata']['total_pois']}")

    print("\n[base cluster stats]")
    for cluster, count in sorted(profile["cluster_stats"].items(), key=lambda kv: (-kv[1], kv[0])):
        cdata = profile["clusters"][cluster]
        print(
            f"  - {cluster:15s} total={count:3d} "
            f"representatives={cdata['stats']['valid_representatives']:3d} "
            f"dual_role={cdata['stats']['dual_role_anchors']:2d} "
            f"vague={cdata['stats']['vague_or_broad']:2d} "
            f"downranked={cdata['stats']['general_downranked']:2d}"
        )

    print("\n[umbrella stats]")
    for name in UMBRELLA_AREAS:
        cdata = profile["clusters"].get(name, {})
        print(
            f"  - {name:15s} total={cdata.get('stats', {}).get('total_pois', 0):3d} "
            f"representatives={cdata.get('stats', {}).get('valid_representatives', 0):3d}"
        )

    print("\n[top representatives]")
    important = [
        "hongdae_area", "gangnam_area", "seongsu", "jongno_area",
        "yongsan_itaewon_area", "myeongdong_euljiro_area",
        "yeouido", "jamsil"
    ]
    for cluster in important:
        cdata = profile["clusters"].get(cluster)
        if not cdata:
            continue
        reps = cdata.get("representative_pois", [])[:7]
        names = [r["name"] for r in reps]
        print(f"  - {cluster:22s}: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Seoul-wide area_profiles_v2.json from POI master CSV.")
    parser.add_argument("--input", type=str, default="", help="Input POI CSV. Default: auto-discover in output/")
    parser.add_argument("--output", type=str, default="output/area_profiles_v2.json", help="Output JSON path.")
    parser.add_argument("--csv-output", type=str, default="output/area_profile_poi_roles_v2.csv", help="Debug CSV output path.")
    parser.add_argument("--google-gap-output", type=str, default="output/area_profile_google_gap_queries.csv", help="Google gap query CSV path.")
    parser.add_argument("--top-k", type=int, default=12, help="Top candidates per role/cluster.")
    parser.add_argument("--cluster-radius-km", type=float, default=DEFAULT_CLUSTER_RADIUS_KM, help="Cluster assignment radius.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write debug CSV.")
    parser.add_argument("--no-google-gap", action="store_true", help="Do not write Google gap query CSV.")
    parser.add_argument("--use-google-fallback", action="store_true", help="Optional: call Google Places Text Search for gap queries.")
    parser.add_argument("--google-candidates-output", type=str, default="output/area_profile_google_candidates.csv")
    parser.add_argument("--google-max-results", type=int, default=3)
    parser.add_argument("--google-sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    input_path = Path(args.input) if args.input else discover_default_input(base_dir)
    if not input_path.is_absolute():
        input_path = base_dir / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    csv_output = None
    if not args.no_csv:
        csv_output = Path(args.csv_output)
        if not csv_output.is_absolute():
            csv_output = base_dir / csv_output

    google_gap_output = None
    if not args.no_google_gap:
        google_gap_output = Path(args.google_gap_output)
        if not google_gap_output.is_absolute():
            google_gap_output = base_dir / google_gap_output

    profile = build_profiles(
        input_path=input_path,
        output_path=output_path,
        csv_output_path=csv_output,
        google_gap_output_path=google_gap_output,
        top_k=args.top_k,
        cluster_radius_km=args.cluster_radius_km,
    )

    print_summary(profile)
    print(f"\n[write] json: {output_path}")
    if csv_output:
        print(f"[write] csv : {csv_output}")
    if google_gap_output:
        print(f"[write] gap : {google_gap_output}")

    if args.use_google_fallback:
        google_candidates_output = Path(args.google_candidates_output)
        if not google_candidates_output.is_absolute():
            google_candidates_output = base_dir / google_candidates_output

        if not google_gap_output:
            print("[Google fallback] --no-google-gap 상태에서는 fallback을 실행할 수 없습니다.")
        else:
            run_google_fallback(
                gap_csv=google_gap_output,
                output_csv=google_candidates_output,
                base_dir=base_dir,
                max_results=args.google_max_results,
                sleep_sec=args.google_sleep_sec,
            )


if __name__ == "__main__":
    main()
