"""
area_profile_builder.py

서울 여행 POI DB 전체를 읽어 area_profiles.json을 생성하는 모듈.

목표
----
- 홍대/성수/강남만을 위한 하드코딩이 아니라, poi_master_step3_enhanced_v2.csv 전체 POI를
  좌표·타입·목적태그·이름 패턴·신뢰도 기준으로 권역별/역할별 분류한다.
- Critic/Repair/Replanner를 분리하기 위한 기반 데이터(area profile)를 만든다.
- market, cafe street, alley, street 같은 POI는 meal 또는 cafe로만 보지 않고
  attraction/day-anchor 역할도 동시에 부여한다.

출력
----
1) output/area_profiles.json
   - cluster별 대표 후보, 역할별 후보, 통계
2) output/area_profile_poi_roles.csv
   - 각 POI의 cluster/roles/score/debug reason 확인용

실행 예시
---------
cd "C:\\Users\\leechaewon\\Desktop\\홍익대 자료\\시스템분석\\project\\ai_agent_travel"
python area_profile_builder.py
python area_profile_builder.py --input output/poi_master_step3_enhanced_v2.csv --output output/area_profiles.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 서울 권역 정의
# ============================================================
# 주의:
# - 이것은 "추천 후보 하드코딩"이 아니라 좌표를 권역으로 묶기 위한 공간 기준이다.
# - 실제 후보 선택은 CSV의 전체 POI를 대상으로 role/score ranking으로 수행한다.
SEOUL_CLUSTERS: dict[str, tuple[float, float]] = {
    "hongdae": (37.5563, 126.9227),
    "mapo": (37.5479, 126.9130),
    "mangwon": (37.5567, 126.9055),
    "sinchon": (37.5596, 126.9373),
    "gangnam": (37.4979, 127.0276),
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

# 서울 안에서 cluster center와 거리가 너무 멀면 other로 둔다.
# 기존 critic_repair의 1.5km보다 넓게 둔다. Replanner용 후보 pool은 좁으면 너무 쉽게 비어버린다.
DEFAULT_CLUSTER_RADIUS_KM = 3.5

# target cluster와 호환 가능한 인접권역.
# Replanner가 day를 다시 짤 때 "홍대/연남/망원/마포"처럼 자연스러운 인접권역을 허용하기 위한 정책.
COMPATIBLE_CLUSTERS: dict[str, set[str]] = {
    "hongdae": {"hongdae", "mapo", "mangwon", "sinchon"},
    "mapo": {"mapo", "hongdae", "mangwon", "sinchon"},
    "mangwon": {"mangwon", "mapo", "hongdae"},
    "sinchon": {"sinchon", "hongdae", "mapo"},
    "gangnam": {"gangnam", "seocho", "apgujeong"},
    "seocho": {"seocho", "gangnam"},
    "apgujeong": {"apgujeong", "gangnam"},
    "seongsu": {"seongsu"},
    "jongno": {"jongno", "insadong", "bukchon", "daehaengno"},
    "insadong": {"insadong", "jongno", "bukchon"},
    "bukchon": {"bukchon", "jongno", "insadong"},
    "daehaengno": {"daehaengno", "jongno"},
    "myeongdong": {"myeongdong", "jongno"},
    "itaewon": {"itaewon", "yongsan"},
    "yongsan": {"yongsan", "itaewon"},
    "dongdaemun": {"dongdaemun", "daehaengno"},
    "yeouido": {"yeouido"},
    "jamsil": {"jamsil"},
}


# ============================================================
# 목적/역할 분류 정책
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
]

# POI type 기반 기본 역할
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
}

# purpose tag 기반 역할
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
}

# 이름/텍스트 패턴 기반 dual-role 정책.
# 홍대만의 규칙이 아니라, 서울 전체에서 "거리/시장/골목/카페거리"가 day anchor가 될 수 있다는 일반 규칙이다.
DUAL_ROLE_ANCHOR_PATTERNS = [
    "market", "시장",
    "street", "거리", "길", "골목", "alley", "road",
    "cafe street", "카페거리", "카페 골목", "카페골목",
    "food street", "음식문화거리", "먹자골목",
    "shopping street", "쇼핑거리",
    "forest park", "숲길", "공원",
]

VAGUE_OR_BROAD_PATTERNS = [
    "area", "district", "neighborhood",
    "동", "일대", "권역",
]

# 너무 넓은 단독 권역명은 POI로 보지 않는다.
# 단, "Hongdae Street Performance", "Seongsu Handmade Shoes Street"처럼 수식어가 있으면
# is_vague_broad_name()이 False가 되도록 아래 함수에서 길이/구체명 패턴을 함께 본다.
BROAD_AREA_EXACT_ALIASES = {
    "hongdae", "hongik univ", "hongik university", "hongik university street",
    "gangnam", "gangnam station",
    "seongsu", "seongsu dong", "seongsu-dong",
    "yongsan", "mangwon dong", "mangwon-dong", "yeonnam dong", "yeonnam-dong",
    "jongno", "myeongdong", "itaewon", "sinchon", "mapo", "jamsil",
    "홍대", "홍익대", "홍익대학교", "강남", "강남역", "성수", "성수동",
    "용산", "망원동", "연남동", "종로", "명동", "이태원", "신촌", "마포", "잠실",
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
    place_confidence: str
    purpose_tags: list[str]
    google_place_id: str
    address: str
    opening_hours: Any
    is_vague_or_broad: bool
    is_dual_role_anchor: bool
    debug_reasons: list[str]


# ============================================================
# 기본 유틸
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
# 역할/대표성 분류
# ============================================================

def is_dual_role_anchor(row: dict[str, Any]) -> bool:
    """시장/거리/골목/카페거리처럼 식사·쇼핑·산책·관광 역할을 동시에 하는 POI인지."""
    name = normalize_name(row.get("poi_name", ""))
    poi_type = normalize_name(row.get("poi_type", ""))
    blob = normalize_name(row_text_blob(row))

    if poi_type in {"market", "street"}:
        return True

    if poi_type in {"shopping"} and any(k in blob for k in ["market", "street", "거리", "길", "골목", "alley"]):
        return True

    # 카페 하나가 아니라 카페거리/카페골목이면 day anchor 역할 가능
    if poi_type == "cafe" and any(k in blob for k in ["cafe street", "카페거리", "카페 골목", "카페골목", "street", "거리", "길", "골목"]):
        return True

    return any(normalize_name(p) in blob for p in DUAL_ROLE_ANCHOR_PATTERNS)


def is_vague_broad_name(row: dict[str, Any]) -> bool:
    """구체 POI가 아니라 넓은 권역명/역명/동네명인지 판단."""
    name_raw = clean_str(row.get("poi_name", ""))
    name = normalize_name(name_raw)
    poi_type = normalize_name(row.get("poi_type", ""))

    if not name:
        return True

    # 정확히 권역명만 있는 경우
    if name in BROAD_AREA_EXACT_ALIASES:
        return True

    # "Hongdae (Hongik University Street)"는 괄호 제거 후 hongdae가 되므로 broad로 처리
    if name in {"hongdae", "gangnam", "yongsan", "seongsu", "yeonnam dong", "mangwon dong"}:
        return True

    # 단독 station/area/district 계열
    if poi_type in {"area", "district", "neighborhood"}:
        return True

    # 다만 Street Performance, Cafe Street, Market, Park처럼 구체 수식어가 있으면 broad로 보지 않음
    concrete_markers = [
        "performance", "market", "park", "museum", "palace", "temple", "library",
        "cafe street", "food street", "shopping center", "mall", "square", "village",
        "street performance", "거리공연", "시장", "공원", "궁", "사원", "도서관", "광장", "마을",
    ]
    if any(m in name for m in [normalize_name(x) for x in concrete_markers]):
        return False

    # 이름이 짧고 broad area keyword가 들어가면 vague 가능성
    tokens = name.split()
    if len(tokens) <= 3 and any(alias in name for alias in BROAD_AREA_EXACT_ALIASES):
        return True

    return False


def infer_roles(row: dict[str, Any]) -> tuple[set[str], list[str]]:
    roles: set[str] = set()
    reasons: list[str] = []

    poi_type = normalize_name(row.get("poi_type", ""))
    if poi_type in TYPE_TO_ROLES:
        roles |= TYPE_TO_ROLES[poi_type]
        reasons.append(f"type:{poi_type}->{sorted(TYPE_TO_ROLES[poi_type])}")

    for tag in json_list(row.get("purpose_tags")):
        tag_norm = normalize_name(tag).replace(" ", "_")
        # normalize_name으로 cafe_hopping이 cafe hopping이 될 수 있어 둘 다 처리
        tag_candidates = {tag_norm, tag_norm.replace("_", " "), str(tag).strip().lower()}
        for t in tag_candidates:
            if t in PURPOSE_TO_ROLES:
                roles |= PURPOSE_TO_ROLES[t]
                reasons.append(f"purpose:{t}->{sorted(PURPOSE_TO_ROLES[t])}")

    blob = normalize_name(row_text_blob(row))

    if any(k in blob for k in ["k pop", "kpop", "케이팝", "한류", "idol", "아이돌"]):
        roles |= {"kpop", "shopping", "culture", "attraction"}
        reasons.append("keyword:kpop")

    if any(k in blob for k in ["palace", "temple", "shrine", "hanok", "궁", "사찰", "절", "한옥"]):
        roles |= {"history", "culture", "attraction"}
        reasons.append("keyword:history_culture")

    if any(k in blob for k in ["park", "forest", "river", "hangang", "공원", "숲", "한강"]):
        roles |= {"nature", "family", "attraction"}
        reasons.append("keyword:nature")

    if any(k in blob for k in ["market", "food street", "restaurant", "시장", "먹자", "음식"]):
        roles |= {"meal", "market", "attraction"}
        reasons.append("keyword:market_food")

    if is_dual_role_anchor(row):
        roles.add("attraction")
        reasons.append("dual_role_anchor:also_attraction")

    if not roles:
        roles.add("attraction")
        reasons.append("fallback:attraction")

    return roles, reasons


def role_score(row: dict[str, Any], role: str, cluster_distance_km: float | None) -> tuple[float, list[str]]:
    """특정 role 후보로서의 점수."""
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

    # 장소 실존 근거
    place_conf = confidence_to_score(row.get("place_confidence"), 0.55)
    score += 0.20 * place_conf
    reasons.append(f"place_conf:{place_conf:.2f}*0.20")

    # google place id
    if not is_missing(row.get("google_place_id")):
        score += 0.08
        reasons.append("google_place_id:+0.08")

    # opening_hours
    if not is_missing(row.get("opening_hours")):
        score += 0.06
        reasons.append("opening_hours:+0.06")

    # 목적 태그 수
    tags = json_list(row.get("purpose_tags"))
    if tags:
        score += min(0.08, 0.02 * len(tags))
        reasons.append(f"purpose_tags:{len(tags)}")

    # 권역 중심에 너무 멀지 않은 후보 선호
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

    # vague/broad 후보 감점
    if is_vague_broad_name(row):
        score -= 0.35
        reasons.append("vague_broad:-0.35")

    # role-specific 보정
    poi_type = normalize_name(row.get("poi_type", ""))

    if role == "meal" and poi_type in {"restaurant", "food", "market"}:
        score += 0.10
        reasons.append("meal_type_bonus:+0.10")

    if role == "cafe" and poi_type == "cafe":
        score += 0.10
        reasons.append("cafe_type_bonus:+0.10")

    if role == "attraction" and poi_type in {"market", "street", "shopping", "park", "culture", "history", "museum", "tourist_spot"}:
        score += 0.08
        reasons.append("attraction_type_bonus:+0.08")

    if role == "attraction" and is_dual_role_anchor(row):
        score += 0.08
        reasons.append("dual_role_anchor_bonus:+0.08")

    score = max(0.0, min(1.0, score))
    return round(score, 4), reasons + role_reasons


def representative_score(row: dict[str, Any], cluster_distance_km: float | None) -> tuple[float, list[str]]:
    """권역 대표 POI로서의 종합 점수."""
    roles, reasons = infer_roles(row)

    # attraction, market, shopping, culture, history, nature, kpop 등은 대표 후보로 강함
    rep_roles = {"attraction", "market", "shopping", "culture", "history", "nature", "kpop", "family"}
    role_fit = 0.75 if roles & rep_roles else 0.45

    if is_dual_role_anchor(row):
        role_fit = max(role_fit, 0.82)
        reasons.append("dual_role_anchor_representative")

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

    detail_fit = 0.35 if is_vague_broad_name(row) else 1.0

    score = (
        0.34 * role_fit
        + 0.22 * place_fit
        + 0.16 * source_fit
        + 0.18 * distance_fit
        + 0.10 * detail_fit
    )

    debug = [
        f"role_fit={role_fit:.2f}",
        f"place_fit={place_fit:.2f}",
        f"source_fit={source_fit:.2f}",
        f"distance_fit={distance_fit:.2f}",
        f"detail_fit={detail_fit:.2f}",
        *reasons,
    ]

    return round(max(0.0, min(1.0, score)), 4), debug


def primary_role_from_scores(role_scores: dict[str, float]) -> str:
    if not role_scores:
        return "attraction"
    # ROLE_ORDER로 tie-break
    return sorted(
        role_scores.items(),
        key=lambda kv: (-kv[1], ROLE_ORDER.index(kv[0]) if kv[0] in ROLE_ORDER else 999)
    )[0][0]


# ============================================================
# Profile 생성
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
    raise FileNotFoundError(
        "POI CSV를 찾지 못했습니다. --input output/poi_master_step3_enhanced_v2.csv 처럼 직접 지정하세요."
    )


def build_profiles(
    input_path: Path,
    output_path: Path,
    csv_output_path: Path | None = None,
    top_k: int = 12,
    cluster_radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
) -> dict[str, Any]:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False).fillna("")

    pois: list[ProfilePOI] = []
    cluster_stats: dict[str, int] = {}

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

        # 최소 attraction 후보는 항상 평가 가능하도록 둔다.
        if "attraction" not in role_scores:
            s, rs = role_score(row, "attraction", dist)
            if s >= 0.35:
                role_scores["attraction"] = s
                role_debug.extend([f"attraction:{x}" for x in rs[:3]])

        primary = primary_role_from_scores(role_scores)
        rep_score, rep_debug = representative_score(row, dist)

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
            place_confidence=clean_str(row.get("place_confidence"), "unknown"),
            purpose_tags=json_list(row.get("purpose_tags")),
            google_place_id=clean_str(row.get("google_place_id")),
            address=clean_str(row.get("address_en") or row.get("address_ko")),
            opening_hours=safe_json_loads(row.get("opening_hours"), default=None),
            is_vague_or_broad=is_vague_broad_name(row),
            is_dual_role_anchor=is_dual_role_anchor(row),
            debug_reasons=rep_debug + role_debug[:10],
        )
        pois.append(p)
        cluster_stats[cluster] = cluster_stats.get(cluster, 0) + 1

    clusters: dict[str, Any] = {}

    for cluster in sorted(cluster_stats.keys()):
        cluster_pois = [p for p in pois if p.cluster == cluster]

        def poi_public(p: ProfilePOI, role: str | None = None) -> dict[str, Any]:
            d = {
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
                "place_confidence": p.place_confidence,
                "purpose_tags": p.purpose_tags,
                "google_place_id": p.google_place_id,
                "address": p.address,
                "is_vague_or_broad": p.is_vague_or_broad,
                "is_dual_role_anchor": p.is_dual_role_anchor,
                "debug_reasons": p.debug_reasons[:8],
            }
            return d

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
            if not p.is_vague_or_broad
            and p.representative_score >= 0.50
        ]
        representatives.sort(
            key=lambda p: (
                p.representative_score,
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

        clusters[cluster] = {
            "center": SEOUL_CLUSTERS.get(cluster),
            "compatible_clusters": sorted(COMPATIBLE_CLUSTERS.get(cluster, {cluster})),
            "stats": {
                "total_pois": len(cluster_pois),
                "valid_representatives": len(representatives),
                "dual_role_anchors": len(dual_role_anchors),
                "vague_or_broad": len(vague_or_broad),
            },
            "representative_pois": [poi_public(p) for p in representatives[:top_k]],
            "role_candidates": role_candidates,
            "dual_role_anchors": [poi_public(p) for p in dual_role_anchors[:top_k]],
            "vague_or_broad_pois": [poi_public(p) for p in vague_or_broad[:top_k]],
        }

    profile = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_csv": str(input_path),
            "total_pois": len(pois),
            "cluster_radius_km": cluster_radius_km,
            "top_k": top_k,
            "purpose": "Area profiles for Seoul-wide itinerary replanning",
            "design_notes": [
                "Uses all POIs from the input CSV, not only Hongdae/Gangnam/Seongsu.",
                "Market/street/cafe-street POIs can be dual-role anchors.",
                "Hardcoded POI recommendations are intentionally avoided; cluster centers are used only for spatial grouping.",
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
                "place_confidence": p.place_confidence,
                "purpose_tags": "|".join(p.purpose_tags),
                "google_place_id": p.google_place_id,
                "is_vague_or_broad": p.is_vague_or_broad,
                "is_dual_role_anchor": p.is_dual_role_anchor,
                "debug_reasons": " || ".join(p.debug_reasons[:12]),
            })
        pd.DataFrame(rows).to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    return profile


def print_summary(profile: dict[str, Any]) -> None:
    print("\n[area_profile_builder] 생성 완료")
    print(f"  source: {profile['metadata']['source_csv']}")
    print(f"  total_pois: {profile['metadata']['total_pois']}")
    print("\n[cluster stats]")
    for cluster, count in sorted(profile["cluster_stats"].items(), key=lambda kv: (-kv[1], kv[0])):
        cdata = profile["clusters"][cluster]
        print(
            f"  - {cluster:12s} total={count:3d} "
            f"representatives={cdata['stats']['valid_representatives']:3d} "
            f"dual_role={cdata['stats']['dual_role_anchors']:2d} "
            f"vague={cdata['stats']['vague_or_broad']:2d}"
        )

    print("\n[top representatives]")
    for cluster, cdata in sorted(profile["clusters"].items()):
        reps = cdata.get("representative_pois", [])[:5]
        names = [r["name"] for r in reps]
        print(f"  - {cluster:12s}: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Seoul-wide area_profiles.json from POI master CSV.")
    parser.add_argument("--input", type=str, default="", help="Input POI CSV. Default: auto-discover in output/")
    parser.add_argument("--output", type=str, default="output/area_profiles.json", help="Output JSON path.")
    parser.add_argument("--csv-output", type=str, default="output/area_profile_poi_roles.csv", help="Debug CSV output path.")
    parser.add_argument("--top-k", type=int, default=12, help="Top candidates per role/cluster.")
    parser.add_argument("--cluster-radius-km", type=float, default=DEFAULT_CLUSTER_RADIUS_KM, help="Cluster assignment radius.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write debug CSV.")
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

    profile = build_profiles(
        input_path=input_path,
        output_path=output_path,
        csv_output_path=csv_output,
        top_k=args.top_k,
        cluster_radius_km=args.cluster_radius_km,
    )
    print_summary(profile)
    print(f"\n[write] json: {output_path}")
    if csv_output:
        print(f"[write] csv : {csv_output}")


if __name__ == "__main__":
    main()
