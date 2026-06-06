"""Itinerary planning nodes for the LangGraph.

`retrieve_node` runs FAISS retrieval over course_data.json and stashes
the top courses in state. `plan_node` calls a DSPy signature that turns
those courses + the user's confirmed fields into a structured day-by-day
itinerary.

개선사항:
1. Google Places API로 카페/식당/K-POP 팝업 실시간 보완 → hallucination 방지
2. RAG k값 증가 (2→5) → 더 많은 POI 후보 확보
3. 생성된 POI hallucination 검증 후 제거
4. 식사 슬롯 자동 보완
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import dspy
import requests
from langchain_core.messages import AIMessage

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv가 없어도 앱은 동작하게 둔다.
    load_dotenv = None

from llm import lm_context
from rag import build_query, retrieve_courses
from state import TravelState


# ---------------------------------------------------------------------------
# Google Places 설정
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(_BASE_DIR / ".env", override=True)

GOOGLE_PLACES_API_KEY = (
    os.getenv("GOOGLE_PLACES_API_KEY")
    or os.getenv("GOOGLE_MAPS_API_KEY")
    or ""
)

# 서울 주요 권역 중심 좌표
SEOUL_AREA_CENTERS = {
    "hongdae":    (37.5563, 126.9227),
    "gangnam":    (37.4979, 127.0276),
    "jongno":     (37.5729, 126.9794),
    "myeongdong": (37.5636, 126.9857),
    "seongsu":    (37.5447, 127.0558),
    "itaewon":    (37.5347, 126.9946),
    "sinchon":    (37.5596, 126.9373),
    "dongdaemun": (37.5666, 127.0097),
    "yeouido":    (37.5217, 126.9244),
    "mapo":       (37.5479, 126.9130),
    "jamsil":     (37.5133, 127.1028),
    "insadong":   (37.5741, 126.9861),
    "mangwon":    (37.5530, 126.9028),
    "hapjeong":   (37.5499, 126.9143),
    "sinsa":      (37.5196, 127.0228),
}

DEFAULT_CENTER = (37.5665, 126.9780)  # 서울 시청

# ============================================================
# 권역 → Day 배정 (1권역 = 1일 원칙)
# ============================================================
# 나중에 문화공연/이벤트 연동 시 _assign_areas_to_days()만 교체하면 됨.
# 예: 특정 날짜에 이태원 공연이 있으면 해당 날짜에 이태원 배정


def _assign_areas_to_days(
    areas: list[str],
    # 향후 파라미터 확장 가능:
    # travel_dates: list[str] | None = None,  # ["2025-06-01", ...]
    # events: list[dict] | None = None,        # 날짜별 공연/이벤트
) -> list[list[str]]:
    """
    감지된 권역을 day별로 배정. 1권역 = 1일 원칙.

    현재: 단순 순서대로 1:1 배정
    ["hongdae", "mangwon", "itaewon"]
    → [["hongdae"], ["mangwon"], ["itaewon"]]

    향후 확장 예시 (문화공연 연동):
    travel_dates = ["2025-06-01", "2025-06-02", "2025-06-03"]
    events = [{"date": "2025-06-02", "area": "itaewon", ...}]
    → 2025-06-02에 이태원 공연 있으면 Day 2에 이태원 배정
    → [["hongdae"], ["itaewon"], ["mangwon"]]
    """
    if not areas:
        return []
    return [[area] for area in areas]


# 이전 함수명과 호환성 유지 (코드 다른 곳에서 참조 시)
_group_areas_by_proximity = _assign_areas_to_days


def _build_day_area_prompt(
    area_groups: list[list[str]],
    duration_days: int,
) -> str:
    """
    권역 그룹을 LLM 프롬프트용 day별 배치 지시문으로 변환.

    - area_groups 수 <= duration_days: 나머지 day는 인접 권역 자유 선택
    - area_groups 수 > duration_days: 앞쪽 그룹만 사용
    """
    if not area_groups:
        return ""

    # umbrella 권역명으로 표시 (더 친숙한 이름)
    AREA_DISPLAY = {
        "hongdae": "Hongdae / Yeonnam",
        "mapo": "Hongdae / Mangwon",
        "mangwon": "Mangwon / Mapo",
        "sinchon": "Sinchon / Hongdae",
        "gangnam": "Gangnam / Sinsa",
        "samseong_coex": "COEX / Samseong",
        "seongsu": "Seongsu / Seoul Forest",
        "jongno": "Jongno / Gwanghwamun",
        "insadong": "Insadong / Bukchon",
        "myeongdong": "Myeongdong / Jung-gu",
        "itaewon": "Itaewon / Hannam",
        "yongsan": "Yongsan / Itaewon",
        "dongdaemun": "Dongdaemun / DDP",
        "yeouido": "Yeouido / Han River",
        "jamsil": "Jamsil / Lotte World",
        "bukchon": "Bukchon / Samcheong",
        "daehaengno": "Daehangno / Naksan",
        "seocho": "Seocho / Express Bus Terminal",
        "apgujeong": "Apgujeong / Rodeo",
    }

    lines = [
        "DAY-BY-DAY AREA ASSIGNMENTS (derived from user preferences — FOLLOW STRICTLY):",
    ]
    groups_to_use = area_groups[:duration_days]
    for i, group in enumerate(groups_to_use, 1):
        display_names = [AREA_DISPLAY.get(a, a.replace("_", " ").title()) for a in group]
        if len(display_names) == 1:
            lines.append(f"  Day {i}: {display_names[0]} area")
        else:
            lines.append(f"  Day {i}: {' + '.join(display_names)} area (these are adjacent — good for one day)")

    if len(groups_to_use) < duration_days:
        remaining = duration_days - len(groups_to_use)
        lines.append(f"  Day {len(groups_to_use)+1}~{duration_days}: Explore other Seoul areas freely ({remaining} day(s))")

    lines.append("Each day MUST stay within its assigned area(s). Do NOT mix areas from different days.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Area Anchor / Representative POI 설정
# ---------------------------------------------------------------------------

# 넓은 권역명이 최종 POI로 남는 문제를 막기 위한 대표 POI prior.
# 이 목록은 "무조건 추천"이 아니라, enhanced DB / Google Places 후보 ranking에서
# 대표성을 부여하는 우선순위 힌트로 사용한다.
AREA_REPRESENTATIVE_POI_PRIORS = {
    "gangnam": {
        "general": [
            "Starfield Library", "별마당도서관",
            "COEX", "코엑스", "Starfield COEX Mall", "스타필드 코엑스",
            "Garosu-gil", "가로수길",
            "Bongeunsa Temple", "봉은사",
            "Dosan Park", "도산공원",
            "Gangnam Station Underground Shopping Center", "강남역 지하쇼핑센터",
        ],
        "shopping": [
            "Starfield COEX Mall", "스타필드 코엑스",
            "COEX", "코엑스",
            "Gangnam Station Underground Shopping Center", "강남역 지하쇼핑센터",
            "Garosu-gil", "가로수길",
            "Apgujeong Rodeo", "압구정로데오",
        ],
        "culture": ["Starfield Library", "별마당도서관", "Bongeunsa Temple", "봉은사", "COEX", "코엑스"],
        "history": ["Bongeunsa Temple", "봉은사", "Starfield Library", "별마당도서관"],
        "kpop": ["K-Star Road", "K스타로드", "Apgujeong Rodeo", "압구정로데오", "COEX", "코엑스"],
        "nature": ["Dosan Park", "도산공원", "Bongeunsa Temple", "봉은사", "Seonjeongneung", "선정릉"],
        "family": ["COEX Aquarium", "코엑스 아쿠아리움", "Starfield Library", "별마당도서관", "COEX", "코엑스", "Dosan Park", "도산공원"],
        "food": ["Garosu-gil", "가로수길", "Apgujeong Rodeo", "압구정로데오", "Gangnam Station", "강남역", "Sinsa-dong", "신사동"],
        "nightlife": ["Gangnam Station", "강남역", "Apgujeong Rodeo", "압구정로데오", "Sinsa-dong", "신사동"],
        "beauty": ["Garosu-gil", "가로수길", "Apgujeong Rodeo", "압구정로데오", "K-Star Road", "K스타로드", "COEX", "코엑스"],
    },
    "hongdae": {
        "general": [
            "Hongdae Street", "홍대 거리", "Hongik University Street",
            "Yeonnam-dong Cafe Street", "연남동 카페 골목",
            "Gyeongui Line Forest Park", "경의선숲길",
            "Mangwon Market", "망원시장",
            "KT&G Sangsangmadang", "상상마당",
        ],
        "shopping": ["Hongdae Street", "홍대 거리", "KT&G Sangsangmadang", "상상마당", "Mangwon Market", "망원시장"],
        "food": ["Mangwon Market", "망원시장", "Yeonnam-dong Cafe Street", "연남동 카페 골목", "Hongdae Street", "홍대 거리"],
        "cafe_hopping": ["Yeonnam-dong Cafe Street", "연남동 카페 골목", "Gyeongui Line Forest Park", "경의선숲길", "Hongdae Street", "홍대 거리"],
        "nightlife": ["Hongdae Street", "홍대 거리"],
        "kpop": ["Hongdae Street", "홍대 거리", "KT&G Sangsangmadang", "상상마당"],
    },
    "seongsu": {
        "general": ["Seoul Forest", "서울숲", "Seongsu Yeonmujang-gil", "성수연무장길", "Amore Seongsu", "아모레 성수", "Seongsu Handmade Shoes Street", "성수동 수제화 거리"],
        "shopping": ["Seongsu Yeonmujang-gil", "성수연무장길", "Amore Seongsu", "아모레 성수", "Seongsu Handmade Shoes Street", "성수동 수제화 거리"],
        "cafe_hopping": ["Seongsu Yeonmujang-gil", "성수연무장길", "Seoul Forest", "서울숲"],
        "beauty": ["Amore Seongsu", "아모레 성수", "Seongsu Yeonmujang-gil", "성수연무장길"],
        "nature": ["Seoul Forest", "서울숲"],
    },
    "myeongdong": {
        "general": ["Myeongdong Street", "명동거리", "Myeongdong Cathedral", "명동성당", "Namsan Seoul Tower", "남산서울타워", "Namdaemun Market", "남대문시장"],
        "shopping": ["Myeongdong Street", "명동거리", "Namdaemun Market", "남대문시장"],
        "food": ["Myeongdong Street", "명동거리", "Namdaemun Market", "남대문시장"],
        "history": ["Myeongdong Cathedral", "명동성당", "Namdaemun Market", "남대문시장"],
    },
    "jongno": {
        "general": ["Gyeongbokgung Palace", "경복궁", "Bukchon Hanok Village", "북촌한옥마을", "Insadong", "인사동", "Gwangjang Market", "광장시장", "Cheonggyecheon Stream", "청계천"],
        "history": ["Gyeongbokgung Palace", "경복궁", "Bukchon Hanok Village", "북촌한옥마을", "Changdeokgung Palace", "창덕궁", "Jongmyo Shrine", "종묘"],
        "culture": ["Insadong", "인사동", "Bukchon Hanok Village", "북촌한옥마을", "Gwangjang Market", "광장시장"],
        "food": ["Gwangjang Market", "광장시장", "Insadong", "인사동"],
    },
}

AREA_ALIASES = {
    "gangnam": ["gangnam", "강남", "coex", "코엑스", "sinsa", "신사", "apgujeong", "압구정"],
    "hongdae": ["hongdae", "홍대", "yeonnam", "연남", "hongik", "상수"],
    "seongsu": ["seongsu", "성수", "서울숲", "seoul forest"],
    "myeongdong": ["myeongdong", "명동", "namsan", "남산"],
    "jongno": ["jongno", "종로", "gyeongbokgung", "경복궁", "bukchon", "북촌", "insadong", "인사동"],
    "itaewon": ["itaewon", "이태원", "hannam", "한남"],
    "jamsil": ["jamsil", "잠실", "lotte world", "롯데월드"],
}

PURPOSE_SYNONYMS = {
    "k-pop": "kpop", "k pop": "kpop", "kpop": "kpop", "케이팝": "kpop", "아이돌": "kpop",
    "shopping": "shopping", "쇼핑": "shopping",
    "food": "food", "맛집": "food", "음식": "food", "식도락": "food",
    "cafe": "cafe_hopping", "coffee": "cafe_hopping", "카페": "cafe_hopping", "카페투어": "cafe_hopping",
    "history": "history", "역사": "history",
    "culture": "culture", "문화": "culture",
    "nature": "nature", "park": "nature", "자연": "nature", "공원": "nature",
    "family": "family", "가족": "family",
    "nightlife": "nightlife", "night": "nightlife", "야경": "nightlife", "밤": "nightlife",
    "beauty": "beauty", "뷰티": "beauty", "화장품": "beauty",
}

EXPLICIT_ONLY_KEYWORDS = {
    "karaoke", "노래방", "club", "클럽", "bar", "바", "pub", "펍",
    "nightlife", "술집", "유흥", "lounge", "라운지", "pc방", "만화카페", "찜질방",
}

ENHANCED_POI_FILES = [
    Path(__file__).resolve().parent / "output" / "poi_master_step3_enhanced_v2.csv",
    Path(__file__).resolve().parent / "output" / "poi_master_step3.csv",
]

_ENHANCED_POI_CACHE: list[dict[str, Any]] | None = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).lower().strip()
    s = re.sub(r"[\(\)\[\]{}.,;:|/\\\-_'\"`~!@#$%^&*+=]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        return None
    return None


def _safe_json_loads(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _load_enhanced_pois() -> list[dict[str, Any]]:
    """enhanced POI master DB를 로드한다. Generator는 이 DB를 1순위 후보 소스로 사용한다."""
    global _ENHANCED_POI_CACHE
    if _ENHANCED_POI_CACHE is not None:
        return _ENHANCED_POI_CACHE

    for path in ENHANCED_POI_FILES:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            _ENHANCED_POI_CACHE = rows
            print(f"[planner] enhanced POI DB 로드: {path} ({len(rows)}개)")
            return rows
        except Exception as e:
            print(f"[planner] enhanced POI DB 로드 실패: {path} / {e}")

    _ENHANCED_POI_CACHE = []
    return []


def _detect_area_keys(text: str) -> list[str]:
    norm = _normalize_text(text)
    found = []
    for area, aliases in AREA_ALIASES.items():
        for alias in aliases:
            if _normalize_text(alias) in norm:
                found.append(area)
                break
    return found


def _is_area_anchor_name(name: Any, poi_type: Any = "") -> bool:
    norm = _normalize_text(name)
    ptype = _normalize_text(poi_type)
    if not norm:
        return False
    area_hit = any(_normalize_text(a) in norm for aliases in AREA_ALIASES.values() for a in aliases)
    short_or_generic = len(norm.split()) <= 4
    generic_type = ptype in {"", "street", "area", "district", "neighborhood", "tourist spot", "tourist_spot"}
    return bool(area_hit and (short_or_generic or generic_type))


def _normalize_purpose_token(value: Any) -> str:
    raw = _normalize_text(value).replace(" ", "_")
    raw_space = _normalize_text(value)
    if raw in PURPOSE_SYNONYMS:
        return PURPOSE_SYNONYMS[raw]
    if raw_space in PURPOSE_SYNONYMS:
        return PURPOSE_SYNONYMS[raw_space]
    for k, v in PURPOSE_SYNONYMS.items():
        if _normalize_text(k) in raw_space:
            return v
    return ""


def _infer_purposes(purpose: str) -> tuple[list[str], bool]:
    tokens = re.split(r"[,/|&]+|\s+and\s+", purpose or "")
    purposes = []
    for t in tokens:
        p = _normalize_purpose_token(t)
        if p and p not in purposes:
            purposes.append(p)
    explicit = bool(purposes)
    return purposes or ["general"], explicit


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _area_center(area: str) -> tuple[float, float]:
    return SEOUL_AREA_CENTERS.get(area, DEFAULT_CENTER)


def _representative_terms(area: str, purposes: list[str], explicit: bool) -> list[str]:
    priors = AREA_REPRESENTATIVE_POI_PRIORS.get(area, {})
    if not priors:
        return []
    keys = purposes if explicit else ["general"]
    ordered = []
    for key in keys:
        for term in priors.get(key, []):
            if term not in ordered:
                ordered.append(term)
    if explicit:
        for term in priors.get("general", []):
            if term not in ordered:
                ordered.append(term)
    return ordered


def _row_blob(row: dict[str, Any]) -> str:
    vals = [
        row.get("poi_name", ""), row.get("poi_type", ""), row.get("google_types", ""),
        row.get("google_editorial_summary", ""), row.get("purpose_tags", ""),
        row.get("purpose_evidence", ""), row.get("label_evidence", ""),
    ]
    return " ".join(str(v).lower() for v in vals if v is not None)


def _is_explicit_only_candidate(row: dict[str, Any]) -> bool:
    blob = _row_blob(row)
    poi_type = _normalize_text(row.get("poi_type", ""))
    if poi_type in {"nightlife", "bar", "club", "karaoke"}:
        return True
    return any(k in blob for k in EXPLICIT_ONLY_KEYWORDS)


def _representative_fit(row: dict[str, Any], area: str, purposes: list[str], explicit: bool) -> tuple[float, str]:
    """권역 대표 POI prior와 후보 row의 이름/근거가 얼마나 정확히 맞는지 평가한다.

    기존 문제:
    - "Yeonnam-dong Cafe Street" prior가 generic token(cafe/street/dong) 때문에
      Mangwon Market 같은 다른 후보와 잘못 매칭될 수 있었음.

    개선:
    - full term 포함을 최우선으로 인정
    - 부분 매칭은 의미 있는 distinctive token 기준으로만 허용
    - cafe/street/market/dong/gil 같은 generic token만 겹치는 경우는 매칭 불인정
    """
    terms = _representative_terms(area, purposes, explicit)
    if not terms:
        return 0.55, "no_area_prior"

    blob = _normalize_text(_row_blob(row))
    generic_tokens = {
        "street", "cafe", "market", "mall", "station", "road", "gil", "dong",
        "거리", "카페", "시장", "몰", "역", "길", "동", "센터", "center", "shopping",
    }

    def _tokens(s: str) -> list[str]:
        return [t for t in _normalize_text(s).split() if t]

    for rank, term in enumerate(terms):
        term_norm = _normalize_text(term)
        if not term_norm:
            continue

        # 1) 전체 표현이 그대로 포함되면 강한 매칭
        if term_norm in blob:
            return max(0.62, 1.0 - 0.035 * rank), f"area_prior_match:{term}:rank={rank + 1}:full"

        term_tokens = _tokens(term_norm)
        if not term_tokens:
            continue

        distinctive = [t for t in term_tokens if t not in generic_tokens and len(t) >= 3]
        if not distinctive:
            continue

        matched_distinctive = [t for t in distinctive if t in blob]
        distinctive_ratio = len(matched_distinctive) / max(len(distinctive), 1)

        # 2) 의미 있는 토큰이 대부분 겹칠 때만 부분 매칭 인정
        #    예: "starfield coex mall" ↔ "Starfield COEX Mall"
        #    예: "yeonnam cafe street"가 "Mangwon Market"에 매칭되는 것을 방지
        if distinctive_ratio >= 0.67 and len(matched_distinctive) >= 1:
            return max(0.58, 0.92 - 0.035 * rank), (
                f"area_prior_match:{term}:rank={rank + 1}:distinctive={matched_distinctive}"
            )

    return (0.20 if not explicit else 0.35), "not_in_area_prior"


def _purpose_fit(row: dict[str, Any], purposes: list[str], explicit: bool) -> float:
    if not explicit:
        return 0.75
    tags = _safe_json_loads(row.get("purpose_tags"), [])
    relevance = _safe_json_loads(row.get("purpose_relevance"), {})
    tags_norm = {_normalize_purpose_token(t) for t in tags if _normalize_purpose_token(t)}
    scores = []
    for p in purposes:
        if isinstance(relevance, dict) and p in relevance:
            try:
                scores.append(max(0.0, min(1.0, float(relevance[p]))))
                continue
            except Exception:
                pass
        if p in tags_norm:
            scores.append(0.85)
        else:
            scores.append(0.35)
    return max(scores) if scores else 0.75


def _confidence_score(value: Any, default: float = 0.55) -> float:
    v = str(value or "").strip().lower()
    if v == "high":
        return 1.0
    if v == "medium":
        return 0.75
    if v == "low":
        return 0.45
    return default


def _score_area_rep_candidate(row: dict[str, Any], area: str, purposes: list[str], explicit: bool) -> tuple[float, dict[str, Any]]:
    lat = _safe_float(row.get("lat"))
    lng = _safe_float(row.get("lng"))
    center_lat, center_lng = _area_center(area)
    if lat is None or lng is None:
        distance_km = None
        distance_fit = 0.45
    else:
        distance_km = _haversine_km(center_lat, center_lng, lat, lng)
        if distance_km <= 0.8:
            distance_fit = 1.0
        elif distance_km <= 1.8:
            distance_fit = 0.85
        elif distance_km <= 3.5:
            distance_fit = 0.60
        else:
            distance_fit = 0.25

    rep_fit, rep_reason = _representative_fit(row, area, purposes, explicit)
    p_fit = _purpose_fit(row, purposes, explicit)
    source_fit = _confidence_score(row.get("place_confidence"), 0.55)

    if _is_area_anchor_name(row.get("poi_name"), row.get("poi_type")):
        concrete_fit = 0.2
    else:
        concrete_fit = 1.0

    if _is_explicit_only_candidate(row) and (not explicit or "nightlife" not in purposes):
        return 0.0, {"reject": "explicit_only_candidate_without_matching_purpose"}

    if not explicit:
        total = (
            0.42 * rep_fit
            + 0.18 * distance_fit
            + 0.18 * source_fit
            + 0.12 * concrete_fit
            + 0.10 * p_fit
        )
    else:
        total = (
            0.30 * rep_fit
            + 0.25 * p_fit
            + 0.17 * distance_fit
            + 0.16 * source_fit
            + 0.12 * concrete_fit
        )

    details = {
        "area": area,
        "distance_km": None if distance_km is None else round(distance_km, 3),
        "representative_fit": round(rep_fit, 3),
        "representative_reason": rep_reason,
        "purpose_fit": round(p_fit, 3),
        "source_fit": round(source_fit, 3),
        "distance_fit": round(distance_fit, 3),
        "concrete_fit": round(concrete_fit, 3),
    }
    return round(total, 4), details


def _poi_from_candidate(row: dict[str, Any], *, source: str | None = None) -> dict[str, Any]:
    stay = row.get("estimated_stay_time") or row.get("stay_minutes") or row.get("estimated_stay_minutes") or 60
    try:
        stay = int(float(stay))
    except Exception:
        stay = 60
    return {
        "poi_name": row.get("poi_name", "") or row.get("name", ""),
        "poi_type": row.get("poi_type", "") or row.get("type", "tourist_spot"),
        "address_en": row.get("address_en") or row.get("address") or "",
        "address_ko": row.get("address_ko") or row.get("address") or "",
        "lat": _safe_float(row.get("lat")),
        "lng": _safe_float(row.get("lng")),
        "rating": row.get("rating") or row.get("google_rating") or row.get("review_rating"),
        "estimated_stay_time": stay,
        "source": source or row.get("source") or "Enhanced POI DB",
        "google_place_id": row.get("google_place_id", ""),
        "representative_score": row.get("_representative_score"),
        "representative_reason": row.get("_representative_reason"),
    }


def _place_dedupe_key(place: dict[str, Any]) -> str:
    """이름 변형이 많은 같은 장소를 최대한 하나로 묶기 위한 key."""
    name = _normalize_text(place.get("poi_name") or place.get("name") or "")
    place_id = str(place.get("google_place_id") or "").strip()
    if place_id:
        return f"gid:{place_id}"

    # 흔한 표기 변형 정리
    aliases = [
        ("starfield coex mall", "coex"),
        ("starfield coex", "coex"),
        ("coex starfield", "coex"),
        ("코엑스 스타필드몰", "coex"),
        ("코엑스 스타필드", "coex"),
        ("스타필드 코엑스몰", "coex"),
        ("스타필드 코엑스", "coex"),
        ("seongsu dong cafe street", "seongsu cafe street"),
        ("성수동 카페거리", "seongsu cafe street"),
        ("yeonnam dong cafe street", "yeonnam cafe street"),
        ("연남동 카페 골목", "yeonnam cafe street"),
        ("mangwon market", "mangwon market"),
        ("망원시장", "mangwon market"),
    ]
    for src, dst in aliases:
        if src in name:
            return f"name:{dst}"

    return f"name:{name}"


def _dedupe_places(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for p in places:
        key = _place_dedupe_key(p)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _fetch_google_place_by_text(query: str, api_key: str) -> dict[str, Any] | None:
    if not api_key:
        return None
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name,geometry,formatted_address,types,rating,user_ratings_total",
        "language": "en",
        "key": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        candidates = resp.json().get("candidates", [])
        if not candidates:
            return None
        c = candidates[0]
        loc = (c.get("geometry") or {}).get("location") or {}
        return {
            "poi_name": c.get("name") or query,
            "poi_type": "tourist_spot",
            "address_en": c.get("formatted_address", ""),
            "address_ko": c.get("formatted_address", ""),
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "rating": c.get("rating"),
            "estimated_stay_time": 60,
            "source": "Google Places (Area Representative)",
            "google_place_id": c.get("place_id", ""),
            "_representative_score": 0.76,
            "_representative_reason": f"google_places_fallback:{query}",
        }
    except Exception as e:
        print(f"[Google Places Area Representative] 오류: {e}")
        return None


def build_area_representative_supplement(
    location: str,
    purpose: str,
    api_key: str = "",
    max_per_area: int = 6,
) -> list[dict[str, Any]]:
    """location/purpose에 등장하는 권역명을 대표 구체 POI 후보로 보강한다.

    원칙:
    1. enhanced POI DB에서 먼저 찾는다.
    2. 권역 대표 prior에 맞는 POI를 우선한다.
    3. 부족하면 Google Places fallback을 사용한다.
    """
    text = f"{location or ''} {purpose or ''}"
    areas = _detect_area_keys(text)
    if not areas:
        return []

    purposes, explicit = _infer_purposes(purpose)
    rows = _load_enhanced_pois()
    supplements: list[dict[str, Any]] = []

    for area in areas:
        scored = []
        for row in rows:
            score, details = _score_area_rep_candidate(row, area, purposes, explicit)
            if score < (0.68 if explicit else 0.72):
                continue
            row2 = dict(row)
            row2["_representative_score"] = score
            row2["_representative_reason"] = details.get("representative_reason")
            scored.append((score, row2))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [_poi_from_candidate(row, source="Enhanced POI DB (Area Representative)") for _, row in scored[:max_per_area]]

        # DB에 대표 후보가 부족하면 Google Places fallback
        if len(picked) < min(3, max_per_area) and api_key:
            terms = _representative_terms(area, purposes, explicit)
            for term in terms:
                if len(picked) >= max_per_area:
                    break
                if any(_normalize_text(term) in _normalize_text(p.get("poi_name", "")) for p in picked):
                    continue
                gp = _fetch_google_place_by_text(f"{term} Seoul", api_key)
                if gp:
                    picked.append(gp)
                time.sleep(0.15)

        for p in picked:
            p["area_anchor"] = area
            p["purpose_hint"] = ", ".join(purposes)
        supplements.extend(picked)

    supplements = _dedupe_places(supplements)
    if supplements:
        print(f"[planner] Area representative POI {len(supplements)}개 보강: {areas}")
    return supplements



def _itinerary_poi_key(poi: dict[str, Any]) -> str:
    name = _normalize_text(poi.get("name") or poi.get("poi_name") or "")
    lat = _safe_float(poi.get("lat"))
    lng = _safe_float(poi.get("lng"))
    if lat is not None and lng is not None:
        return f"{name}|{round(lat, 4)}|{round(lng, 4)}"
    return name


def _is_same_poi_name(a: str, b: str) -> bool:
    ak = _normalize_text(a)
    bk = _normalize_text(b)
    if not ak or not bk:
        return False
    if ak == bk:
        return True
    return ak in bk or bk in ak


def _dedupe_itinerary_pois(itinerary: dict[str, Any]) -> dict[str, Any]:
    """생성 결과에서 같은 POI가 반복되는 문제를 최종 정리한다.

    원칙:
    - 같은 day 안의 같은 POI 반복은 제거
    - 전체 일정에서 같은 POI가 반복되면 뒤쪽 반복을 제거하되,
      해당 day가 너무 비어 버리는 경우에는 보수적으로 유지
    """
    global_seen: set[str] = set()

    for day in itinerary.get("days", []) or []:
        pois = day.get("pois", []) or []
        day_seen: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        removed: list[str] = []

        for poi in pois:
            key = _itinerary_poi_key(poi)
            name = poi.get("name", "")

            # 같은 day 내부 중복은 제거
            if key in day_seen:
                removed.append(name)
                continue

            # 전체 일정 중복은 day가 4개 이상 남을 때 제거
            if key in global_seen and len(pois) - len(removed) > 4:
                removed.append(name)
                continue

            day_seen.add(key)
            global_seen.add(key)
            cleaned.append(poi)

        if removed:
            print(f"[planner] Day {day.get('day')} duplicated POI removed: {removed}")

        day["pois"] = cleaned

    return itinerary


def _select_unused_reps(
    reps: list[dict[str, Any]],
    used_names: set[str],
    count: int,
) -> list[dict[str, Any]]:
    selected = []
    for r in reps:
        name = r.get("poi_name", "")
        key = _normalize_text(name)
        if not key or key in used_names:
            continue
        selected.append(r)
        used_names.add(key)
        if len(selected) >= count:
            break
    return selected


def _make_itinerary_poi(candidate: dict[str, Any], notes_prefix: str = "") -> dict[str, Any]:
    name = candidate.get("poi_name", "") or candidate.get("name", "")
    notes = notes_prefix or "Area anchor was expanded into a concrete representative POI."
    reason = candidate.get("representative_reason")
    score = candidate.get("representative_score")

    # evidence는 실제 선택된 candidate의 정보와만 연결한다.
    if reason:
        notes += f" Evidence for {name}: {reason}."
    if score:
        notes += f" Representative score: {score}."

    try:
        stay = int(float(candidate.get("estimated_stay_time") or 60))
    except Exception:
        stay = 60

    return {
        "name": name,
        "type": candidate.get("poi_type", "tourist_spot"),
        "address": candidate.get("address_en") or candidate.get("address_ko") or "",
        "lat": candidate.get("lat"),
        "lng": candidate.get("lng"),
        "stay_minutes": stay,
        "notes": notes,
    }


def _area_reps_by_area(area_representatives: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_area: dict[str, list[dict[str, Any]]] = {}
    for p in area_representatives or []:
        area = p.get("area_anchor")
        if not area:
            continue
        by_area.setdefault(area, []).append(p)
    return by_area


def _expand_or_replace_area_anchors(
    itinerary: dict[str, Any],
    area_representatives: list[dict[str, Any]],
    *,
    user_selected_mode: bool = False,
) -> dict[str, Any]:
    """Planner 출력에 넓은 권역명이 남아 있으면 대표 concrete POI로 바꾼다.

    - day에 POI가 3개 이상이면 1:1 대체
    - day에 POI가 1~2개뿐이고 사용자가 직접 선택한 모드가 아니면 2~4개 mini-course로 확장
    - 동일 representative 후보를 같은 itinerary 안에서 반복 사용하지 않음
    - 후보가 없으면 그대로 두고 Critic이 warning을 잡게 한다
    """
    by_area = _area_reps_by_area(area_representatives)
    if not by_area:
        return itinerary

    used_names: set[str] = set()
    # 기존 itinerary에 이미 들어 있는 구체 POI는 representative 재사용에서 제외
    for day in itinerary.get("days", []) or []:
        for poi in day.get("pois", []) or []:
            if not _is_area_anchor_name(poi.get("name", ""), poi.get("type", "")):
                used_names.add(_normalize_text(poi.get("name", "")))

    for day in itinerary.get("days", []) or []:
        pois = day.get("pois", []) or []
        new_pois: list[dict[str, Any]] = []

        for poi in pois:
            name = poi.get("name", "")
            ptype = poi.get("type", "")

            if not _is_area_anchor_name(name, ptype):
                new_pois.append(poi)
                continue

            area_keys = _detect_area_keys(f"{name} {day.get('theme', '')}")
            area = area_keys[0] if area_keys else None
            reps = by_area.get(area, []) if area else []

            if not reps:
                new_pois.append(poi)
                continue

            if len(pois) <= 2 and not user_selected_mode:
                # 하루가 거의 권역 anchor 하나로만 구성된 경우 mini-course로 확장
                count = min(4, max(2, len(reps)))
                selected = _select_unused_reps(reps, used_names, count)

                # 전부 이미 사용되어 selected가 비면, 강제로 첫 후보 1개만 사용
                if not selected:
                    selected = reps[:1]

                expanded = [
                    _make_itinerary_poi(
                        r,
                        notes_prefix=f"{name} area expanded into a representative POI for this day."
                    )
                    for r in selected
                ]
                print(f"[planner] Area anchor expansion: {name} → {[p['name'] for p in expanded]}")
                new_pois.extend(expanded)
            else:
                selected = _select_unused_reps(reps, used_names, 1)
                if not selected:
                    selected = reps[:1]

                replacement = _make_itinerary_poi(
                    selected[0],
                    notes_prefix=f"{name} area replaced with a concrete representative POI."
                )
                print(f"[planner] Area anchor replacement: {name} → {replacement['name']}")
                new_pois.append(replacement)

        day["pois"] = new_pois

    return _dedupe_itinerary_pois(itinerary)



# ---------------------------------------------------------------------------
# Google Places API 함수들
# ---------------------------------------------------------------------------

def _get_area_center(location: str) -> tuple[float, float]:
    """location 문자열에서 권역 중심 좌표 추출."""
    loc_lower = location.lower()
    for area, coords in SEOUL_AREA_CENTERS.items():
        if area in loc_lower:
            return coords
    return DEFAULT_CENTER


# dietary 제한 키워드 매핑
DIETARY_EXCLUDE_KEYWORDS: dict[str, list[str]] = {
    "seafood":    ["seafood", "fish", "sushi", "sashimi", "crab", "shrimp", "lobster",
                   "해산물", "생선", "초밥", "회", "게", "새우", "랍스터", "해물"],
    "pork":       ["pork", "pig", "bacon", "ham", "삼겹살", "돼지", "베이컨", "햄", "족발"],
    "beef":       ["beef", "steak", "소고기", "스테이크", "육회"],
    "meat":       ["meat", "chicken", "beef", "pork", "육류", "고기", "닭", "소", "돼지"],
    "vegetarian": [],
    "vegan":      [],
    "halal":      ["pork", "돼지", "bacon", "ham", "삼겹살"],
    "nut":        ["nut", "peanut", "알몬드", "견과류", "땅콩"],
    "gluten":     ["ramen", "noodle", "bread", "pasta", "라멘", "국수", "빵", "파스타"],
}


def _parse_dietary_restrictions(dietary: str) -> list[str]:
    if not dietary or dietary.lower() in {"none", "no", "없음", "없어요"}:
        return []
    d = dietary.lower()
    return [key for key in DIETARY_EXCLUDE_KEYWORDS if key in d]


def _place_violates_dietary(name: str, types: list[str], restrictions: list[str]) -> bool:
    if not restrictions:
        return False
    combined = f"{name.lower()} {' '.join(types).lower()}"
    for r in restrictions:
        for word in DIETARY_EXCLUDE_KEYWORDS.get(r, []):
            if word in combined:
                return True
    return False


def fetch_nearby_places(
    lat: float,
    lng: float,
    place_type: str,
    api_key: str,
    radius: int = 1500,
    min_rating: float = 4.3,
    min_reviews: int = 100,
    max_results: int = 5,
    dietary_restrictions: list[str] | None = None,
) -> list[dict]:
    """
    Google Places Nearby Search로 주변 장소 검색.
    place_type: "cafe" | "restaurant" | "bar"

    품질 필터:
    - min_rating: 4.3+ (기본값 강화)
    - min_reviews: 100+ (리뷰 적은 곳 제외)
    - dietary_restrictions: 식이 제한 필터
    - 카페: 리뷰 수(외국인 많이 가는 곳) 기준 정렬
    - 식당: rating 기준 정렬
    """
    if not api_key:
        return []
    dietary_restrictions = dietary_restrictions or []

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": place_type,
        "key": api_key,
        "language": "en",
        "rankby": "prominence",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        results = resp.json().get("results", [])

        # rating + 리뷰 수 필터
        filtered = [
            r for r in results
            if r.get("rating", 0) >= min_rating
            and r.get("user_ratings_total", 0) >= min_reviews
            and r.get("business_status", "OPERATIONAL") == "OPERATIONAL"
        ]

        # dietary 필터
        if dietary_restrictions:
            filtered = [
                r for r in filtered
                if not _place_violates_dietary(
                    r.get("name", ""),
                    r.get("types", []),
                    dietary_restrictions,
                )
            ]

        # 카페는 리뷰 수 기준, 식당은 rating 기준 정렬
        if place_type == "cafe":
            filtered.sort(key=lambda r: r.get("user_ratings_total", 0), reverse=True)
        else:
            filtered.sort(key=lambda r: r.get("rating", 0), reverse=True)

        return [
            {
                "poi_name": r["name"],
                "poi_type": place_type,
                "address_en": r.get("vicinity", ""),
                "address_ko": r.get("vicinity", ""),
                "lat": r["geometry"]["location"]["lat"],
                "lng": r["geometry"]["location"]["lng"],
                "rating": r.get("rating"),
                "user_ratings_total": r.get("user_ratings_total", 0),
                "opening_hours": r.get("opening_hours"),
                "place_id": r.get("place_id", ""),
                "estimated_stay_time": 60 if place_type == "restaurant" else 45,
                "source": "Google Places",
                "dietary_safe": True,
            }
            for r in filtered[:max_results]
        ]
    except Exception as e:
        print(f"[Google Places Nearby] 오류: {e}")
        return []


def fetch_kpop_places(
    lat: float,
    lng: float,
    api_key: str,
    purpose: str,
    radius: int = 3000,
    max_results: int = 5,
) -> list[dict]:
    """
    Google Places Text Search로 K-POP 관련 장소 검색.
    purpose에서 아티스트 이름 추출해서 검색.
    """
    if not api_key:
        return []

    # 아티스트/K-POP 키워드 추출
    kpop_keywords = []
    purpose_lower = purpose.lower()
    artists = ["bts", "blackpink", "aespa", "newjeans", "ive", "stray kids",
               "twice", "exo", "seventeen", "txt", "enhypen"]
    for artist in artists:
        if artist in purpose_lower:
            kpop_keywords.append(f"{artist} popup store Seoul")
            kpop_keywords.append(f"{artist} cafe Seoul")

    if not kpop_keywords:
        kpop_keywords = ["kpop popup store Seoul", "kpop merch store Hongdae"]

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    results = []
    seen_names = set()

    for keyword in kpop_keywords[:2]:  # API 비용 절약
        params = {
            "query": keyword,
            "location": f"{lat},{lng}",
            "radius": radius,
            "key": api_key,
            "language": "en",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            items = resp.json().get("results", [])
            for r in items[:3]:
                name = r.get("name", "")
                if name in seen_names:
                    continue
                seen_names.add(name)
                if r.get("business_status") == "OPERATIONAL":
                    results.append({
                        "poi_name": name,
                        "poi_type": "kpop_landmark",
                        "address_en": r.get("formatted_address", ""),
                        "address_ko": r.get("formatted_address", ""),
                        "lat": r["geometry"]["location"]["lat"],
                        "lng": r["geometry"]["location"]["lng"],
                        "rating": r.get("rating"),
                        "estimated_stay_time": 60,
                        "source": "Google Places (K-POP)",
                    })
            time.sleep(0.3)
        except Exception as e:
            print(f"[Google Places K-POP] 오류: {e}")

    return results[:max_results]


def build_google_supplement(
    location: str,
    purpose: str,
    api_key: str,
    dietary: str = "none",
    center_lat: float | None = None,
    center_lng: float | None = None,
) -> list[dict]:
    """
    사용자 목적에 맞게 Google Places 보완 데이터 수집.
    카페, 식당, K-POP 장소 등을 실시간으로 가져옴.

    center_lat/center_lng: 명시적 좌표 (권역별 검색 시 사용)
                           None이면 location 문자열에서 자동 추론.

    품질 기준:
    - 식당: rating 4.3+, 리뷰 100개 이상, dietary 필터 적용
    - 카페: rating 4.2+, 리뷰 100개 이상, 리뷰 수(외국인 기준) 정렬
    """
    if not api_key:
        return []

    if center_lat is not None and center_lng is not None:
        pass  # 명시적 좌표 사용
    else:
        center_lat, center_lng = _get_area_center(location)
    supplement = []
    purpose_lower = purpose.lower()

    # dietary 제한 파싱
    dietary_restrictions = _parse_dietary_restrictions(dietary)
    if dietary_restrictions:
        print(f"[Google Places] dietary 제한 적용: {dietary_restrictions}")

    # dietary 제한 있으면 기준 완화 (채식/할랄 등 특수 식당이 적은 문제 대응)
    has_dietary = bool(dietary_restrictions)
    restaurant_min_rating = 4.0 if has_dietary else 4.3
    restaurant_min_reviews = 50 if has_dietary else 100

    # 카페 보완 (항상 추가 - 외국인 리뷰 많은 순)
    cafes = fetch_nearby_places(
        center_lat, center_lng, "cafe", api_key,
        radius=1500, min_rating=4.2, min_reviews=100, max_results=5,
        dietary_restrictions=[],
    )
    supplement.extend(cafes)
    print(f"[Google Places] 카페 {len(cafes)}개 추가")

    # 식당 보완
    restaurants = fetch_nearby_places(
        center_lat, center_lng, "restaurant", api_key,
        radius=1500, min_rating=restaurant_min_rating,
        min_reviews=restaurant_min_reviews, max_results=8,
        dietary_restrictions=dietary_restrictions,
    )
    # dietary 있는데 결과 부족하면 반경 확장 재시도
    if has_dietary and len(restaurants) < 3:
        restaurants_retry = fetch_nearby_places(
            center_lat, center_lng, "restaurant", api_key,
            radius=3000, min_rating=3.8, min_reviews=30, max_results=8,
            dietary_restrictions=dietary_restrictions,
        )
        # 기존 결과에 없는 것만 추가
        existing = {r["poi_name"] for r in restaurants}
        restaurants += [r for r in restaurants_retry if r["poi_name"] not in existing]
        print(f"[Google Places] dietary 재시도 — 식당 {len(restaurants)}개")

    supplement.extend(restaurants)
    print(f"[Google Places] 식당 {len(restaurants)}개 추가 (dietary={dietary_restrictions or 'none'})")

    # K-POP 장소 보완
    if any(k in purpose_lower for k in ["kpop", "k-pop", "bts", "blackpink",
                                         "idol", "kpop", "아이돌"]):
        kpop_places = fetch_kpop_places(
            center_lat, center_lng, api_key, purpose,
            radius=5000, max_results=5
        )
        supplement.extend(kpop_places)
        print(f"[Google Places] K-POP 장소 {len(kpop_places)}개 추가")

    # 쇼핑 보완
    if any(k in purpose_lower for k in ["shopping", "쇼핑"]):
        shops = fetch_nearby_places(
            center_lat, center_lng, "shopping_mall", api_key,
            radius=2000, min_rating=4.0, max_results=3
        )
        supplement.extend(shops)
        print(f"[Google Places] 쇼핑 {len(shops)}개 추가")

    return supplement


def _format_google_supplement(places: list[dict]) -> str:
    """보완 POI 데이터를 프롬프트용 텍스트로 변환."""
    if not places:
        return ""
    lines = ["\n\n=== VERIFIED SUPPLEMENTAL POI DATA ==="]
    lines.append(
        "Use these verified real places for area anchors, cafes, restaurants, shopping, and K-POP slots. "
        "For broad area names such as Gangnam/Hongdae/Seongsu, choose concrete representative POIs from this list.\n"
    )
    for p in places:
        rating = p.get("rating")
        rating_str = f"rating={rating}" if rating else ""
        rep = p.get("representative_score")
        rep_str = f"representative_score={rep}" if rep else ""
        lines.append(
            f"  - {p.get('poi_name', '')} [{p.get('poi_type', '')}] "
            f"addr={p.get('address_en') or p.get('address_ko', '')} "
            f"lat={p.get('lat')} lng={p.get('lng')} "
            f"stay={p.get('estimated_stay_time', 60)}min "
            f"{rating_str} {rep_str} source={p.get('source', '')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------

class ItineraryPlanner(dspy.Signature):
    """Generate a personalized Seoul travel itinerary for foreign tourists.

    You are given the user's trip details and a shortlist of candidate
    courses (each with a sequence of POIs), PLUS real-time Google Places
    data for cafes, restaurants, and K-POP spots.

    Build a realistic day-by-day plan following ALL rules below.

      STRUCTURE RULES:
      - One day entry per requested trip duration day.
      - Each day MUST have 5–8 POIs. Never fewer than 5.
      - Each day MUST include at least one restaurant or cafe POI for lunch
        and ideally one for dinner or afternoon coffee.
      - Arrange POIs in chronological visit order starting around 09:00–10:00.
      - Total planned activity + travel time per day: 7–10 hours.

      CONTENT RULES:
      - Carefully read the user's "purpose" and prioritize matching POI types.
        * "cafe" or "coffee" → include 2+ cafe POIs per day (use Google Places cafes)
        * "shopping" → include shopping streets, markets, malls
        * "K-POP" or "kpop" or artist names → include kpop_landmark POIs
          AND use Google Places K-POP spots (popup stores, themed cafes)
        * "culture" or "history" → include museum, history, culture POIs
        * "nature" or "park" → include park POIs
        * "nightlife" or "club" → include nightlife POIs in the evening
      - Honor dietary restrictions strictly when selecting restaurants/cafes.
      - Stay within the user's budget across all days combined.
      - Write "notes" that explain: (1) why this POI matches the user's purpose,
        (2) any cultural tips for foreign visitors (e.g. "1 item per person rule",
        "remove shoes", "cash only"), (3) practical info (reservation needed, etc.).

      GEOGRAPHY RULES:
      - Each day must stay within 1–2 adjacent Seoul neighborhoods.
        Good same-day neighborhood pairs: Hongdae+Hapjeong, Hongdae+Mangwon,
        Gangnam+Sinsa, Jongno+Insadong, Seongsu+Wangsimni, Itaewon+Hanam.
      - Do NOT mix distant areas in one day (e.g. Hongdae + Gangnam = bad).
      - Order POIs geographically to minimize backtracking and travel time.
      - Different days should cover different neighborhoods for variety.

      AREA ANCHOR RULES:
      - Do NOT use broad area names as final POIs: Gangnam, Hongdae, Seongsu,
        Myeongdong, Jongno, Itaewon, Jamsil, etc.
      - If the user asks for an area such as Gangnam, expand it into concrete
        representative POIs from VERIFIED SUPPLEMENTAL POI DATA.
      - For general Gangnam, prefer Starfield Library, COEX/Starfield COEX Mall,
        Garosu-gil, Bongeunsa Temple, Dosan Park, or Gangnam Station Underground
        Shopping Center over generic/ambiguous places.
      - If a day has only one broad area anchor, create a mini-course of 2–4
        representative POIs in that area. If the day already has enough POIs,
        replace the broad area anchor with one concrete representative POI.
      - Never choose karaoke/bar/club/nightlife places unless the user explicitly
        requested nightlife.

      DATA INTEGRITY RULES:
      - For sightseeing/parks/museums: use POIs from candidate_courses ONLY.
      - For cafes/restaurants: use ONLY places from VERIFIED SUPPLEMENTAL POI DATA section.
        NEVER invent restaurant or cafe names not listed in the supplemental data.
        If no suitable restaurant exists in the data, omit the meal slot entirely.
      - For kpop spots: use Google Places K-POP data provided.
      - Do NOT invent or hallucinate POI names, addresses, or coordinates.
      - Do NOT use restaurants/cafes from your training knowledge — only use verified data.
      - Copy lat, lng, address EXACTLY from the data provided.
      - Only list a course in "sources" if you used at least one of its POIs.
        Use the exact course_id and source_url from the input.

    Return ONLY valid JSON matching this schema, with no markdown fences:
    {
      "summary": "<2-3 sentence overview mentioning neighborhoods and highlights>",
      "days": [
        {
          "day": 1,
          "theme": "<short evocative theme, e.g. 'Hongdae Street Culture & Cafes'>",
          "pois": [
            {
              "name": "<POI name exactly as in the data>",
              "type": "<poi_type>",
              "address": "<address from the data>",
              "lat": <number>,
              "lng": <number>,
              "stay_minutes": <integer between 30 and 240>,
              "notes": "<purpose fit + cultural tips + practical info>"
            }
          ],
          "estimated_cost": "<realistic cost range for the day, e.g. '$40–70 USD'>"
        }
      ],
      "sources": [
        {
          "course_id": "<exact course_id from candidate_courses>",
          "course_title": "<exact course_title from candidate_courses>",
          "source": "<Visit Seoul or Visit Korea>",
          "source_url": "<exact source_url from candidate_courses>"
        }
      ]
    }
    """
    duration: str = dspy.InputField(desc="Trip length, e.g. '3 days'.")
    location: str = dspy.InputField(desc="Destination or accommodation area.")
    budget: str = dspy.InputField(desc="Total trip budget.")
    dietary: str = dspy.InputField(desc="Dietary restrictions or preferences.")
    purpose: str = dspy.InputField(desc="Purpose of the trip, e.g. 'cafes, shopping, K-POP, BTS'.")
    candidate_courses: str = dspy.InputField(
        desc=(
            "Shortlist of candidate courses with POIs (from Visit Seoul/Korea DB), "
            "PLUS real-time Google Places data for cafes/restaurants/kpop spots."
        )
    )
    itinerary_json: str = dspy.OutputField(
        desc="Itinerary as a JSON object matching the schema in the docstring."
    )


class FixJSON(dspy.Signature):
    """Repair a JSON document that failed to parse.

    Output ONLY the corrected JSON object. No prose, no markdown fences,
    no explanation. Preserve all fields and values from the broken input;
    only fix the syntax (escape quotes, remove trailing commas, replace
    smart quotes with straight quotes, etc.).
    """
    broken_json: str = dspy.InputField(desc="The malformed JSON text.")
    error_message: str = dspy.InputField(desc="The parser error reported.")
    fixed_json: str = dspy.OutputField(desc="Strictly valid JSON only.")


_planner: dspy.Predict | None = None
_fixer: dspy.Predict | None = None


def get_planner() -> dspy.Predict:
    global _planner
    if _planner is None:
        _planner = dspy.Predict(ItineraryPlanner)
    return _planner


def get_fixer() -> dspy.Predict:
    global _fixer
    if _fixer is None:
        _fixer = dspy.Predict(FixJSON)
    return _fixer


# ---------------------------------------------------------------------------
# Course compaction for the prompt
# ---------------------------------------------------------------------------

def _format_courses_for_prompt(
    courses: list[dict[str, Any]],
    google_supplement: list[dict] | None = None,
) -> str:
    blocks = []
    for i, c in enumerate(courses, start=1):
        title = c.get("course_title", "")
        course_id = c.get("course_id", "")
        source = c.get("source", "")
        source_url = c.get("source_url", "")
        themes = ", ".join(c.get("theme_category", []) or [])
        poi_lines = []
        for p in c.get("sequence", []) or []:
            poi_lines.append(
                f"    - {p.get('poi_name', '')} "
                f"[{p.get('poi_type', '')}] "
                f"addr={p.get('address_en') or p.get('address_ko', '')} "
                f"lat={p.get('lat')} lng={p.get('lng')} "
                f"stay={p.get('estimated_stay_time')}min"
            )
        blocks.append(
            f"Course {i}: {title}\n"
            f"  course_id : {course_id}\n"
            f"  source    : {source}\n"
            f"  source_url: {source_url}\n"
            f"  Themes    : {themes}\n"
            f"  POIs:\n" + "\n".join(poi_lines)
        )

    result = "\n\n".join(blocks)

    # Google Places 보완 데이터 추가
    if google_supplement:
        result += _format_google_supplement(google_supplement)

    return result


# ---------------------------------------------------------------------------
# Hallucination 검증
# ---------------------------------------------------------------------------

def _build_valid_poi_names(
    courses: list[dict[str, Any]],
    google_supplement: list[dict] | None = None,
) -> set[str]:
    """candidate_courses + Google Places의 모든 POI 이름 수집."""
    valid = set()
    for c in courses:
        for p in c.get("sequence", []) or []:
            name = str(p.get("poi_name", "")).lower().strip()
            if name:
                valid.add(name)
    if google_supplement:
        for p in google_supplement:
            name = str(p.get("poi_name", "")).lower().strip()
            if name:
                valid.add(name)
    return valid


def _validate_and_fix_pois(
    itinerary: dict[str, Any],
    valid_names: set[str],
    google_supplement: list[dict] | None = None,
) -> dict[str, Any]:
    """
    hallucinated POI 제거 후 식사 슬롯 부족하면 Google Places 식당/카페로 보완.

    검증 정책:
    - 관광지/쇼핑/역사 등: candidate_courses 이름 기반 부분 매칭 허용
    - 식당/카페: Google Places 목록에 있는 것만 허용 (hallucination 완전 차단)
    - broad area anchor: critic/replanner가 처리하므로 통과
    """
    google_restaurants = [
        p for p in (google_supplement or [])
        if p.get("poi_type") in ["restaurant", "cafe", "market"]
    ]

    # Google Places 식당/카페 이름 집합 (정확 매칭용)
    google_meal_names: set[str] = {
        _normalize_text(p.get("poi_name", ""))
        for p in google_restaurants
        if p.get("poi_name")
    }

    # 관광지용 valid names (부분 매칭 허용)
    normalized_valid_names = {_normalize_text(v) for v in valid_names if v}

    MEAL_TYPES = {"restaurant", "cafe", "food", "bar", "market"}

    for day in itinerary.get("days", []):
        original_pois = day.get("pois", [])
        validated = []
        hallucinated_names = []

        for poi in original_pois:
            poi_name_raw = str(poi.get("name", "")).strip()
            poi_name = _normalize_text(poi_name_raw)
            poi_type_raw = str(poi.get("type", "")).lower().strip()

            # broad area anchor → critic/replanner가 처리, 통과
            if _is_area_anchor_name(poi.get("name", ""), poi.get("type", "")):
                validated.append(poi)
                continue

            # ---- 식당/카페 전용 검증 ----
            if poi_type_raw in MEAL_TYPES:
                # Google Places 목록에 있는 것만 허용 (정확 매칭 우선)
                matched_google = None
                for g_poi in google_restaurants:
                    g_name = _normalize_text(g_poi.get("poi_name", ""))
                    if (poi_name and g_name and
                        (poi_name == g_name or
                         (len(poi_name) >= 5 and len(g_name) >= 5 and
                          (poi_name in g_name or g_name in poi_name)))):
                        matched_google = g_poi
                        break

                if matched_google:
                    # source를 Google Places로 자동 채움
                    poi = dict(poi)
                    poi["source"] = "Google Places"
                    poi["rating"] = matched_google.get("rating") or poi.get("rating")
                    poi["user_ratings_total"] = matched_google.get("user_ratings_total", 0)
                    if not poi.get("lat") and matched_google.get("lat"):
                        poi["lat"] = matched_google["lat"]
                        poi["lng"] = matched_google["lng"]
                    validated.append(poi)
                else:
                    hallucinated_names.append(poi_name_raw + " [MEAL-HALLUCINATED]")
                continue

            # ---- 관광지/기타 POI 검증 (부분 매칭 허용) ----
            is_valid = (
                poi_name in normalized_valid_names
                or any(
                    poi_name and v and len(poi_name) >= 4 and len(v) >= 4 and
                    (poi_name in v or v in poi_name)
                    for v in normalized_valid_names
                )
            )
            if is_valid:
                validated.append(poi)
            else:
                hallucinated_names.append(poi_name_raw)

        if hallucinated_names:
            print(f"[Validator] Day {day.get('day')} hallucinated POI 제거: {hallucinated_names}")

        # 식사 슬롯 체크 - restaurant/cafe/market이 없으면 Google Places에서 자동 추가
        has_meal = any(
            str(p.get("type", "")).lower() in MEAL_TYPES
            for p in validated
        )
        if not has_meal and google_restaurants:
            existing_names = {_normalize_text(p.get("name", "")) for p in validated}
            best = None
            for cand in google_restaurants:
                if _normalize_text(cand.get("poi_name", "")) not in existing_names:
                    best = cand
                    break

            if best:
                meal_poi = {
                    "name": best["poi_name"],
                    "type": best["poi_type"],
                    "address": best.get("address_en", ""),
                    "lat": best["lat"],
                    "lng": best["lng"],
                    "stay_minutes": best.get("estimated_stay_time", 60),
                    "notes": (
                        f"Verified restaurant from Google Places. "
                        f"Rating: {best.get('rating', 'N/A')} "
                        f"({best.get('user_ratings_total', 0):,} reviews)."
                    ),
                }
                insert_idx = min(2, len(validated))
                validated.insert(insert_idx, meal_poi)
                print(f"[Validator] Day {day.get('day')} 식사 슬롯 자동 추가: {best['poi_name']} "
                      f"(rating={best.get('rating')}, reviews={best.get('user_ratings_total', 0)})")

        day["pois"] = validated

    return _dedupe_itinerary_pois(itinerary)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _isolate_json_object(text: str) -> str:
    text = _FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start: end + 1]
    return text


def _simple_repair(text: str) -> str:
    text = (
        text.replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
    )
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _parse_itinerary_json(raw: str, *, use_llm_fallback: bool = True) -> dict[str, Any]:
    isolated = _isolate_json_object(raw)
    try:
        return json.loads(isolated)
    except json.JSONDecodeError:
        pass

    repaired = _simple_repair(isolated)
    second_err: json.JSONDecodeError | None = None
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        second_err = e

    if use_llm_fallback and second_err is not None:
        try:
            with lm_context():
                fixed = get_fixer()(
                    broken_json=isolated[:8000],
                    error_message=str(second_err),
                ).fixed_json
            return json.loads(_isolate_json_object(fixed))
        except Exception:
            pass

    _dump_debug(raw)
    if second_err is not None:
        raise second_err
    raise json.JSONDecodeError("Failed to parse itinerary JSON", raw, 0)


def _dump_debug(raw: str) -> None:
    try:
        dbg_path = Path(__file__).resolve().parent / "planner_last_failed.txt"
        dbg_path.write_text(raw, encoding="utf-8")
        print(f"[planner] wrote failing output to {dbg_path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sources hygiene
# ---------------------------------------------------------------------------

def _normalize_sources(
    itinerary: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {c.get("course_id"): c for c in retrieved if c.get("course_id")}
    by_url = {c.get("source_url"): c for c in retrieved if c.get("source_url")}
    raw_sources = itinerary.get("sources") or []
    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for s in raw_sources:
        if not isinstance(s, dict):
            continue
        match = by_id.get(s.get("course_id")) or by_url.get(s.get("source_url"))
        if not match:
            continue
        cid = match.get("course_id")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        cleaned.append({
            "course_id": cid,
            "course_title": match.get("course_title", ""),
            "source": match.get("source", ""),
            "source_url": match.get("source_url", ""),
        })
    if not cleaned and retrieved:
        cleaned = [
            {
                "course_id": c.get("course_id"),
                "course_title": c.get("course_title", ""),
                "source": c.get("source", ""),
                "source_url": c.get("source_url", ""),
            }
            for c in retrieved
            if c.get("source_url")
        ]
    itinerary["sources"] = cleaned
    return itinerary


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def make_retrieve_node(api_key: str):
    def retrieve_node(state: TravelState) -> TravelState:
        query = build_query(
            purpose=state.get("purpose"),
            dietary=state.get("dietary"),
            location=state.get("location"),
            duration=state.get("duration"),
        )
        try:
            # k=5로 늘려서 더 많은 POI 후보 확보
            courses = retrieve_courses(
                api_key=api_key,
                query=query,
                k=5,
                location=state.get("location"),
                purpose=state.get("purpose"),
            )
        except Exception as e:
            return {
                **state,
                "current_step": "confirm",
                "messages": [AIMessage(content=f"⚠️ Failed to retrieve courses: {e}")],
            }
        return {
            **state,
            "retrieved_courses": courses,
            "current_step": "planning",
        }
    return retrieve_node


def plan_node(state: TravelState) -> TravelState:
    courses = state.get("retrieved_courses") or []
    if not courses:
        return {
            **state,
            "current_step": "done",
            "messages": [AIMessage(content="⚠️ No candidate courses found. Try different details.")],
        }

    location = state.get("location") or ""
    purpose = state.get("purpose") or ""

    # 1) Area anchor 대표 POI 보강: enhanced DB 우선, 부족하면 Google Places fallback
    area_representatives = build_area_representative_supplement(
        location=location,
        purpose=purpose,
        api_key=GOOGLE_PLACES_API_KEY,
    )

    # 2) Google Places로 카페/식당/K-POP 실시간 보완 (dietary 필터 + 품질 필터 적용)
    # 권역별 좌표로 각각 검색해서 각 권역에 맞는 식당/카페 확보
    google_supplement = []
    dietary = state.get("dietary") or "none"
    if GOOGLE_PLACES_API_KEY:
        print("[planner] Google Places 보완 데이터 수집 중...")
        detected_areas_for_google = _detect_area_keys(f"{location} {purpose}")
        if detected_areas_for_google:
            # 권역별로 각각 검색
            seen_names: set[str] = set()
            for area in detected_areas_for_google[:4]:  # 최대 4개 권역
                area_lat, area_lng = SEOUL_AREA_CENTERS.get(area, DEFAULT_CENTER)
                area_supplement = build_google_supplement(
                    location=area,
                    purpose=purpose,
                    api_key=GOOGLE_PLACES_API_KEY,
                    dietary=dietary,
                    center_lat=area_lat,
                    center_lng=area_lng,
                )
                # 중복 제거
                for p in area_supplement:
                    name = p.get("poi_name", "")
                    if name and name not in seen_names:
                        seen_names.add(name)
                        p["area_hint"] = area  # 어느 권역 식당인지 표시
                        google_supplement.append(p)
        else:
            # 권역 감지 안 되면 기존 방식
            google_supplement = build_google_supplement(
                location=location,
                purpose=purpose,
                api_key=GOOGLE_PLACES_API_KEY,
                dietary=dietary,
            )
        print(f"[planner] Google Places 총 {len(google_supplement)}개 보완 데이터 확보")
    else:
        print("[planner] GOOGLE_PLACES_API_KEY 없음 — Google Places 보완 생략")

    supplemental_pois = _dedupe_places(area_representatives + google_supplement)

    # 유효한 POI 이름 집합 (hallucination 검증용)
    valid_names = _build_valid_poi_names(courses, supplemental_pois)

    # 3) 감지된 권역을 거리 기반으로 day별 그룹핑
    detected_areas = _detect_area_keys(f"{location} {purpose}")
    try:
        duration_days = int(str(state.get("duration") or "3").split()[0])
    except Exception:
        duration_days = 3

    area_groups = _group_areas_by_proximity(detected_areas)
    day_area_prompt = _build_day_area_prompt(area_groups, duration_days)

    if area_groups:
        print(f"[planner] 권역 그룹핑: {area_groups} → {duration_days}일")

    # candidate_courses에 day별 권역 지시문 추가
    courses_prompt = _format_courses_for_prompt(courses, supplemental_pois)
    if day_area_prompt:
        courses_prompt = day_area_prompt + "\n\n" + courses_prompt

    try:
        with lm_context():
            result = get_planner()(
                duration=state.get("duration") or "",
                location=location,
                budget=state.get("budget") or "",
                dietary=state.get("dietary") or "none",
                purpose=purpose,
                candidate_courses=courses_prompt,
            )
        itinerary = _parse_itinerary_json(result.itinerary_json)

        # Area anchor를 먼저 concrete POI로 확장/대체
        user_selected_mode = bool(state.get("selected_pois") or state.get("user_selected_pois"))
        itinerary = _expand_or_replace_area_anchors(
            itinerary,
            area_representatives,
            user_selected_mode=user_selected_mode,
        )

        # Hallucination 검증 + 식사 슬롯 보완
        itinerary = _validate_and_fix_pois(itinerary, valid_names, supplemental_pois)

        # 검증 후에도 넓은 권역명이 남아 있으면 한 번 더 처리
        itinerary = _expand_or_replace_area_anchors(
            itinerary,
            area_representatives,
            user_selected_mode=user_selected_mode,
        )

        # 최종 중복 제거 및 sources 정리
        itinerary = _dedupe_itinerary_pois(itinerary)
        itinerary = _normalize_sources(itinerary, courses)

        # Google Places 식당/카페를 replanner가 참조할 수 있도록 itinerary에 첨부
        supplement_restaurants = [
            p for p in google_supplement
            if p.get("poi_type") in {"restaurant", "cafe"}
        ]
        if supplement_restaurants:
            itinerary["supplement_restaurants"] = supplement_restaurants
            print(f"[planner] supplement_restaurants {len(supplement_restaurants)}개 itinerary에 첨부")

    except json.JSONDecodeError as e:
        return {
            **state,
            "current_step": "done",
            "messages": [AIMessage(content=f"⚠️ Planner returned invalid JSON: {e}")],
        }
    except Exception as e:
        return {
            **state,
            "current_step": "done",
            "messages": [AIMessage(content=f"⚠️ Planning failed: {e}")],
        }

    # -------------------------------------------------------
    # Pipeline: critic → replanner → critic → repair → 최종
    # supplement_restaurants가 itinerary에 첨부된 상태로 전달
    # -------------------------------------------------------
    try:
        from pipeline import run_pipeline
        user_state_for_pipeline = {
            "purpose": state.get("purpose") or "general",
            "location": state.get("location") or "",
            "duration": state.get("duration") or "",
            "dietary": state.get("dietary") or "none",
        }
        print("[planner] pipeline 실행 중...")
        itinerary, pipeline_log = run_pipeline(
            itinerary=itinerary,
            user_state=user_state_for_pipeline,
            verbose=True,
        )
        score = pipeline_log.get("final_score", "?")
        passed = pipeline_log.get("final_passed", False)
        changes = pipeline_log.get("total_changes", 0)
        print(f"[planner] pipeline 완료 — score={score}, passed={passed}, changes={changes}")
    except Exception as e:
        print(f"[planner] pipeline 오류 (원본 itinerary 사용): {e}")

    summary = itinerary.get("summary", "")
    day_count = len(itinerary.get("days", []))
    ack = (
        f"✅ Your {day_count}-day itinerary is ready!\n\n"
        f"{summary}\n\n"
        "See the full plan below."
    )

    return {
        **state,
        "itinerary": itinerary,
        "current_step": "done",
        "messages": [AIMessage(content=ack)],
    }