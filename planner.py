"""Itinerary planning nodes for the LangGraph.

`retrieve_node` runs FAISS retrieval over course_data.json and stashes
the top courses in state. `plan_node` calls a DSPy signature that turns
those courses + the user's confirmed fields into a structured day-by-day
itinerary.

Main improvements:
1. Search RAG by requested areas such as Hongdae and Seongsu.
2. Call Google Places for EACH requested area, not only once.
3. Add real cafes/restaurants/K-POP/shopping places from Google Places.
4. Force itinerary to cover all requested neighborhoods.
5. Remove hallucinated POIs that are not in candidate courses or Google Places.
6. Auto-fill missing meals and under-filled days.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import dspy
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from llm import lm_context
from rag import build_query, retrieve_courses
from state import TravelState

load_dotenv()


# ---------------------------------------------------------------------------
# Google Places configuration
# ---------------------------------------------------------------------------

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

SEOUL_AREA_CENTERS: dict[str, tuple[float, float]] = {
    "hongdae": (37.5563, 126.9227),
    "hapjeong": (37.5499, 126.9143),
    "mangwon": (37.5530, 126.9028),
    "yeonnam": (37.5663, 126.9236),
    "seongsu": (37.5447, 127.0558),
    "wangsimni": (37.5612, 127.0371),
    "gangnam": (37.4979, 127.0276),
    "sinsa": (37.5196, 127.0228),
    "garosu-gil": (37.5207, 127.0227),
    "jongno": (37.5729, 126.9794),
    "insadong": (37.5741, 126.9861),
    "myeongdong": (37.5636, 126.9857),
    "itaewon": (37.5347, 126.9946),
    "sinchon": (37.5596, 126.9373),
    "dongdaemun": (37.5666, 127.0097),
    "yeouido": (37.5217, 126.9244),
    "mapo": (37.5479, 126.9130),
    "jamsil": (37.5133, 127.1028),
    "dmc": (37.5770, 126.8902),
}

AREA_ALIASES: dict[str, list[str]] = {
    "hongdae": ["hongdae", "hongik", "hongik univ", "홍대", "hongik university"],
    "hapjeong": ["hapjeong", "합정"],
    "mangwon": ["mangwon", "망원"],
    "yeonnam": ["yeonnam", "연남"],
    "seongsu": ["seongsu", "성수", "seongsu-dong", "성수동"],
    "wangsimni": ["wangsimni", "왕십리"],
    "gangnam": ["gangnam", "강남"],
    "sinsa": ["sinsa", "신사"],
    "garosu-gil": ["garosu", "garosu-gil", "가로수길"],
    "jongno": ["jongno", "종로"],
    "insadong": ["insadong", "인사동"],
    "myeongdong": ["myeongdong", "명동"],
    "itaewon": ["itaewon", "이태원"],
    "sinchon": ["sinchon", "신촌"],
    "dongdaemun": ["dongdaemun", "동대문"],
    "yeouido": ["yeouido", "여의도"],
    "mapo": ["mapo", "마포"],
    "jamsil": ["jamsil", "잠실"],
    "dmc": ["digital media city", "dmc", "상암", "디지털미디어시티"],
}

DEFAULT_CENTER = (37.5665, 126.9780)


# ---------------------------------------------------------------------------
# Area utilities
# ---------------------------------------------------------------------------

def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _extract_requested_areas(location: str | None, purpose: str | None = None) -> list[str]:
    """Extract requested Seoul neighborhoods from user text."""
    text = f"{location or ''} {purpose or ''}".lower()
    found: list[str] = []

    for area, aliases in AREA_ALIASES.items():
        if any(alias.lower() in text for alias in aliases):
            if area not in found:
                found.append(area)

    # If Hongdae is requested, Mangwon/Hapjeong are adjacent support areas,
    # but do not add them as requested areas unless explicitly mentioned.
    return found


def _area_label(area: str) -> str:
    labels = {
        "hongdae": "Hongdae",
        "hapjeong": "Hapjeong",
        "mangwon": "Mangwon",
        "yeonnam": "Yeonnam",
        "seongsu": "Seongsu",
        "wangsimni": "Wangsimni",
        "gangnam": "Gangnam",
        "sinsa": "Sinsa",
        "garosu-gil": "Garosu-gil",
        "jongno": "Jongno",
        "insadong": "Insadong",
        "myeongdong": "Myeongdong",
        "itaewon": "Itaewon",
        "sinchon": "Sinchon",
        "dongdaemun": "Dongdaemun",
        "yeouido": "Yeouido",
        "mapo": "Mapo",
        "jamsil": "Jamsil",
        "dmc": "Digital Media City",
    }
    return labels.get(area, area.title())


def _get_area_center(area_or_location: str) -> tuple[float, float]:
    text = _normalize_text(area_or_location)
    for area, aliases in AREA_ALIASES.items():
        if area in text or any(alias in text for alias in aliases):
            return SEOUL_AREA_CENTERS.get(area, DEFAULT_CENTER)
    return DEFAULT_CENTER


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lam / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _infer_area_from_text_or_coords(
    name: Any = "",
    address: Any = "",
    lat: Any = None,
    lng: Any = None,
) -> str | None:
    text = f"{name or ''} {address or ''}".lower()

    for area, aliases in AREA_ALIASES.items():
        if area in text or any(alias in text for alias in aliases):
            return area

    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except Exception:
        return None

    nearest_area = None
    nearest_dist = 9999.0

    for area, (center_lat, center_lng) in SEOUL_AREA_CENTERS.items():
        dist = _haversine_km(lat_f, lng_f, center_lat, center_lng)
        if dist < nearest_dist:
            nearest_area = area
            nearest_dist = dist

    # Seoul neighborhoods are dense; use a loose threshold.
    if nearest_dist <= 3.2:
        return nearest_area
    return None


def _area_matches_requested(area: str | None, requested: str) -> bool:
    if not area:
        return False

    if area == requested:
        return True

    adjacent = {
        "hongdae": {"hongdae", "hapjeong", "mangwon", "yeonnam", "mapo"},
        "seongsu": {"seongsu", "wangsimni"},
        "gangnam": {"gangnam", "sinsa", "garosu-gil"},
        "jongno": {"jongno", "insadong", "myeongdong"},
    }

    return area in adjacent.get(requested, {requested})


# ---------------------------------------------------------------------------
# Google Places API
# ---------------------------------------------------------------------------

def _google_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
        status = data.get("status")
        if status not in {"OK", "ZERO_RESULTS"}:
            print(f"[Google Places] status={status}, error={data.get('error_message')}")
        return data
    except Exception as e:
        print(f"[Google Places] request error: {e}")
        return {}


def fetch_nearby_places(
    *,
    area: str,
    place_type: str,
    api_key: str,
    radius: int = 1700,
    min_rating: float = 4.0,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Google Places Nearby Search for one area."""
    if not api_key:
        return []

    lat, lng = SEOUL_AREA_CENTERS.get(area, DEFAULT_CENTER)
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": place_type,
        "key": api_key,
        "language": "en",
    }

    data = _google_get(url, params)
    results = data.get("results", []) or []
    filtered = [r for r in results if float(r.get("rating") or 0) >= min_rating]

    places: list[dict[str, Any]] = []
    for r in filtered[:max_results]:
        loc = (r.get("geometry") or {}).get("location") or {}
        if "lat" not in loc or "lng" not in loc:
            continue

        stay = 60
        if place_type == "cafe":
            stay = 45
        elif place_type == "restaurant":
            stay = 60
        elif place_type == "shopping_mall":
            stay = 75

        places.append({
            "poi_name": r.get("name", ""),
            "poi_type": place_type,
            "address_en": r.get("vicinity") or r.get("formatted_address", ""),
            "address_ko": r.get("vicinity") or r.get("formatted_address", ""),
            "lat": loc["lat"],
            "lng": loc["lng"],
            "rating": r.get("rating"),
            "estimated_stay_time": stay,
            "source": f"Google Places ({_area_label(area)})",
            "area": area,
            "place_id": r.get("place_id", ""),
        })

    return places


def fetch_text_places(
    *,
    area: str,
    query: str,
    api_key: str,
    radius: int = 2500,
    min_rating: float = 0.0,
    max_results: int = 5,
    poi_type: str = "tourist_spot",
) -> list[dict[str, Any]]:
    """Google Places Text Search for one area."""
    if not api_key:
        return []

    lat, lng = SEOUL_AREA_CENTERS.get(area, DEFAULT_CENTER)
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": api_key,
        "language": "en",
    }

    data = _google_get(url, params)
    results = data.get("results", []) or []

    places: list[dict[str, Any]] = []
    seen: set[str] = set()

    for r in results:
        name = r.get("name", "")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())

        if r.get("business_status") and r.get("business_status") != "OPERATIONAL":
            continue

        rating = float(r.get("rating") or 0)
        if rating < min_rating:
            continue

        loc = (r.get("geometry") or {}).get("location") or {}
        if "lat" not in loc or "lng" not in loc:
            continue

        places.append({
            "poi_name": name,
            "poi_type": poi_type,
            "address_en": r.get("formatted_address", ""),
            "address_ko": r.get("formatted_address", ""),
            "lat": loc["lat"],
            "lng": loc["lng"],
            "rating": r.get("rating"),
            "estimated_stay_time": 60,
            "source": f"Google Places Text ({_area_label(area)})",
            "area": area,
            "place_id": r.get("place_id", ""),
        })

        if len(places) >= max_results:
            break

    return places


def fetch_kpop_places_for_area(
    *,
    area: str,
    api_key: str,
    purpose: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    if not api_key:
        return []

    purpose_lower = purpose.lower()

    artists = [
        "bts", "blackpink", "aespa", "newjeans", "ive", "stray kids",
        "twice", "exo", "seventeen", "txt", "enhypen", "idol", "kpop", "k-pop",
    ]

    detected = [a for a in artists if a in purpose_lower]
    area_name = _area_label(area)

    queries: list[str] = []

    if detected:
        for artist in detected[:2]:
            artist_clean = artist.replace("k-pop", "kpop")
            queries.append(f"{artist_clean} store {area_name} Seoul")
            queries.append(f"{artist_clean} cafe {area_name} Seoul")

    queries.extend([
        f"kpop store {area_name} Seoul",
        f"kpop merchandise {area_name} Seoul",
        f"kpop popup store {area_name} Seoul",
    ])

    all_places: list[dict[str, Any]] = []
    seen: set[str] = set()

    for q in queries[:4]:
        places = fetch_text_places(
            area=area,
            query=q,
            api_key=api_key,
            radius=3500,
            min_rating=0.0,
            max_results=3,
            poi_type="kpop_landmark",
        )
        for p in places:
            key = _normalize_text(p.get("poi_name"))
            if key and key not in seen:
                seen.add(key)
                all_places.append(p)
        time.sleep(0.2)

    return all_places[:max_results]


def build_google_supplement_for_area(
    *,
    area: str,
    purpose: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Collect Google Places supplement for one requested area."""
    if not api_key:
        return []

    purpose_lower = purpose.lower()
    supplement: list[dict[str, Any]] = []

    # Cafes are essential for Seoul travel and the current project use case.
    need_cafe = any(k in purpose_lower for k in ["cafe", "coffee", "relax", "카페"])
    if need_cafe:
        cafes = fetch_nearby_places(
            area=area,
            place_type="cafe",
            api_key=api_key,
            radius=1800,
            min_rating=4.1,
            max_results=5,
        )
        if len(cafes) < 3:
            cafes += fetch_text_places(
                area=area,
                query=f"best cafes in {_area_label(area)} Seoul",
                api_key=api_key,
                radius=2500,
                min_rating=4.0,
                max_results=5 - len(cafes),
                poi_type="cafe",
            )
        supplement.extend(cafes)
        print(f"[Google Places][{_area_label(area)}] 카페 {len(cafes)}개 추가")

    restaurants = fetch_nearby_places(
        area=area,
        place_type="restaurant",
        api_key=api_key,
        radius=1800,
        min_rating=4.0,
        max_results=5,
    )
    if len(restaurants) < 3:
        restaurants += fetch_text_places(
            area=area,
            query=f"popular restaurants in {_area_label(area)} Seoul",
            api_key=api_key,
            radius=2500,
            min_rating=4.0,
            max_results=5 - len(restaurants),
            poi_type="restaurant",
        )
    supplement.extend(restaurants)
    print(f"[Google Places][{_area_label(area)}] 식당 {len(restaurants)}개 추가")

    if any(k in purpose_lower for k in ["kpop", "k-pop", "bts", "blackpink", "idol", "아이돌"]):
        kpop_places = fetch_kpop_places_for_area(
            area=area,
            api_key=api_key,
            purpose=purpose,
            max_results=5,
        )
        supplement.extend(kpop_places)
        print(f"[Google Places][{_area_label(area)}] K-POP 장소 {len(kpop_places)}개 추가")

    if any(k in purpose_lower for k in ["shopping", "shop", "fashion", "쇼핑"]):
        shops = fetch_nearby_places(
            area=area,
            place_type="shopping_mall",
            api_key=api_key,
            radius=2200,
            min_rating=4.0,
            max_results=3,
        )
        if len(shops) < 2:
            shops += fetch_text_places(
                area=area,
                query=f"shopping in {_area_label(area)} Seoul",
                api_key=api_key,
                radius=2500,
                min_rating=4.0,
                max_results=3 - len(shops),
                poi_type="shopping",
            )
        supplement.extend(shops)
        print(f"[Google Places][{_area_label(area)}] 쇼핑 {len(shops)}개 추가")

    return _dedupe_places(supplement)


def build_google_supplement_by_areas(
    *,
    requested_areas: list[str],
    location: str,
    purpose: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Collect Google Places supplement for every requested area."""
    if not api_key:
        return []

    if not requested_areas:
        # Fallback: choose one area from location string or Seoul center.
        fallback_area = None
        text = _normalize_text(location)
        for area, aliases in AREA_ALIASES.items():
            if area in text or any(alias in text for alias in aliases):
                fallback_area = area
                break
        requested_areas = [fallback_area or "myeongdong"]

    print(f"[planner] 요청 지역별 Google Places 보완 시작: {[_area_label(a) for a in requested_areas]}")

    all_places: list[dict[str, Any]] = []
    for area in requested_areas:
        places = build_google_supplement_for_area(
            area=area,
            purpose=purpose,
            api_key=api_key,
        )
        all_places.extend(places)

    all_places = _dedupe_places(all_places)
    print(f"[planner] Google Places 총 {len(all_places)}개 보완 데이터 확보")
    return all_places


def _dedupe_places(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for p in places:
        name = _normalize_text(p.get("poi_name"))
        lat = p.get("lat")
        lng = p.get("lng")
        key = f"{name}|{round(float(lat), 4) if lat is not None else ''}|{round(float(lng), 4) if lng is not None else ''}"
        if not name or key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return deduped


# ---------------------------------------------------------------------------
# Formatting prompt context
# ---------------------------------------------------------------------------

def _format_google_supplement(places: list[dict[str, Any]]) -> str:
    if not places:
        return ""

    lines = [
        "",
        "",
        "=== REAL-TIME GOOGLE PLACES DATA ===",
        "These are verified real places. Use them for cafes, restaurants, K-POP spots, and shopping.",
        "Each Google Places POI has an `area` field. If the user requested that area, you MUST use some POIs from that area.",
        "",
    ]

    for p in places:
        rating = f"rating={p.get('rating')}" if p.get("rating") else ""
        lines.append(
            f"  - {p.get('poi_name', '')} "
            f"[{p.get('poi_type', '')}] "
            f"area={p.get('area', '')} "
            f"addr={p.get('address_en') or p.get('address_ko', '')} "
            f"lat={p.get('lat')} lng={p.get('lng')} "
            f"stay={p.get('estimated_stay_time', 60)}min "
            f"{rating} "
            f"source={p.get('source', '')}"
        )

    return "\n".join(lines)


def _format_requested_area_rules(requested_areas: list[str], duration: str) -> str:
    if not requested_areas:
        return ""

    labels = [_area_label(a) for a in requested_areas]

    lines = [
        "",
        "=== REQUESTED AREA COVERAGE RULES ===",
        f"The user explicitly requested these areas: {', '.join(labels)}.",
        "You MUST include at least 2 POIs from EACH requested area across the full itinerary.",
        "Do NOT omit a requested area.",
        "If candidate course data is weak for an area, use REAL-TIME GOOGLE PLACES DATA for that area.",
    ]

    if len(requested_areas) >= 2:
        lines.append(
            "For a 2-day trip with 2 requested areas, assign one main requested area per day. "
            "Example: Day 1 = Hongdae/Mangwon, Day 2 = Seongsu."
        )

    lines.append(
        "If you cannot find enough sightseeing POIs for an area, use cafes, restaurants, shops, or cultural spaces from Google Places."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class ItineraryPlanner(dspy.Signature):
    """Generate a personalized Seoul travel itinerary for foreign tourists.

    You are given:
    1. the user's trip details,
    2. a shortlist of candidate courses from Visit Seoul / Visit Korea,
    3. real-time Google Places data for requested neighborhoods.

    Build a realistic day-by-day itinerary following ALL rules below.

    STRUCTURE RULES:
    - One day entry per requested trip duration day.
    - Each day MUST have 5–8 POIs. Never fewer than 5.
    - Each day MUST include at least one restaurant or cafe POI.
    - Arrange POIs in chronological visit order starting around 09:00–10:00.
    - Total planned activity + travel time per day should be 7–10 hours.

    REQUESTED AREA RULES:
    - If the user mentions multiple neighborhoods, cover ALL requested neighborhoods.
    - Include at least 2 POIs from EACH requested neighborhood across the itinerary.
    - For a 2-day trip with Hongdae and Seongsu, Day 1 can focus on Hongdae/Mangwon and Day 2 MUST focus on Seongsu.
    - Do not say "no relevant POI data was available" if Google Places data is provided for that area.
    - If candidate course data lacks a requested area, use Google Places supplement for that requested area.

    CONTENT RULES:
    - Prioritize POIs that match the user's purpose.
      * cafe or coffee -> include cafes from Google Places
      * shopping -> include markets, streets, malls, fashion shops
      * K-POP, kpop, BTS, BLACKPINK, idol -> include kpop_landmark POIs and Google Places K-POP spots
      * local culture -> include markets, streets, local neighborhoods, cultural spaces
      * relaxing -> include parks, riverside spots, cafes, healing spaces
    - Honor dietary restrictions strictly.
    - Stay within the user's budget.
    - Notes must explain why the POI fits the user's purpose and include practical/cultural tips when relevant.

    GEOGRAPHY RULES:
    - Each day should stay within 1–2 adjacent neighborhoods.
    - Good pairs: Hongdae+Mangwon, Hongdae+Hapjeong, Seongsu+Wangsimni, Gangnam+Sinsa, Jongno+Insadong.
    - Do NOT mix distant areas in one day unless unavoidable.
    - Order POIs geographically to minimize backtracking.

    DATA INTEGRITY RULES:
    - Use ONLY POIs that appear in candidate_courses or REAL-TIME GOOGLE PLACES DATA.
    - Do NOT invent generic POIs such as "Hongdae Nightlife", "Street Food Stalls", or "Seongsu Cafe Street" unless they appear exactly in the data.
    - Copy name, lat, lng, and address from the provided data.
    - For cafes, restaurants, shopping, and K-POP, prefer Google Places because it provides real current places.
    - Only list a course in sources if you used at least one POI from that course.

    Return ONLY valid JSON with no markdown fences:
    {
      "summary": "<2-3 sentence overview mentioning all requested neighborhoods>",
      "days": [
        {
          "day": 1,
          "theme": "<short day theme>",
          "pois": [
            {
              "name": "<POI name exactly as provided>",
              "type": "<poi_type>",
              "address": "<address from provided data>",
              "lat": <number>,
              "lng": <number>,
              "stay_minutes": <integer>,
              "notes": "<purpose fit + cultural/practical tips>"
            }
          ],
          "estimated_cost": "<realistic day cost>"
        }
      ],
      "sources": [
        {
          "course_id": "<exact course_id>",
          "course_title": "<exact course_title>",
          "source": "<Visit Seoul or Visit Korea>",
          "source_url": "<exact source_url>"
        }
      ]
    }
    """

    duration: str = dspy.InputField(desc="Trip length, e.g. '2 days'.")
    location: str = dspy.InputField(desc="Destination or requested neighborhoods.")
    budget: str = dspy.InputField(desc="Total trip budget.")
    dietary: str = dspy.InputField(desc="Dietary restrictions or preferences.")
    purpose: str = dspy.InputField(desc="Trip purpose, e.g. cafes, shopping, K-POP.")
    candidate_courses: str = dspy.InputField(
        desc="Candidate courses and Google Places supplement as compact text."
    )
    itinerary_json: str = dspy.OutputField(
        desc="Strict JSON itinerary matching the schema."
    )


class FixJSON(dspy.Signature):
    """Repair a JSON document that failed to parse.

    Output ONLY the corrected JSON object. No prose, no markdown fences.
    Preserve all fields and values from the broken input; only fix syntax.
    """
    broken_json: str = dspy.InputField(desc="Malformed JSON text.")
    error_message: str = dspy.InputField(desc="Parser error.")
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
# Candidate formatting
# ---------------------------------------------------------------------------

def _format_courses_for_prompt(
    courses: list[dict[str, Any]],
    google_supplement: list[dict[str, Any]] | None = None,
    requested_areas: list[str] | None = None,
    duration: str = "",
) -> str:
    blocks: list[str] = []

    requested_areas = requested_areas or []

    for i, c in enumerate(courses, start=1):
        title = c.get("course_title", "")
        course_id = c.get("course_id", "")
        source = c.get("source", "")
        source_url = c.get("source_url", "")
        themes = c.get("theme_category", [])
        if isinstance(themes, list):
            themes_str = ", ".join(themes)
        else:
            themes_str = str(themes or "")

        poi_lines: list[str] = []
        for p in c.get("sequence", []) or []:
            name = p.get("poi_name", "")
            address = p.get("address_en") or p.get("address_ko", "")
            lat = p.get("lat")
            lng = p.get("lng")
            area = _infer_area_from_text_or_coords(name, address, lat, lng) or ""

            poi_lines.append(
                f"    - {name} "
                f"[{p.get('poi_type', '')}] "
                f"area={area} "
                f"addr={address} "
                f"lat={lat} lng={lng} "
                f"stay={p.get('estimated_stay_time')}min"
            )

        blocks.append(
            f"Course {i}: {title}\n"
            f"  course_id : {course_id}\n"
            f"  source    : {source}\n"
            f"  source_url: {source_url}\n"
            f"  Themes    : {themes_str}\n"
            f"  POIs:\n" + "\n".join(poi_lines)
        )

    result = "\n\n".join(blocks)
    result += _format_requested_area_rules(requested_areas, duration)

    if google_supplement:
        result += _format_google_supplement(google_supplement)

    return result


# ---------------------------------------------------------------------------
# Candidate pool and validation
# ---------------------------------------------------------------------------

def _poi_from_course_item(p: dict[str, Any]) -> dict[str, Any]:
    name = p.get("poi_name") or p.get("name") or ""
    address = p.get("address_en") or p.get("address_ko") or p.get("address") or ""
    area = _infer_area_from_text_or_coords(name, address, p.get("lat"), p.get("lng"))

    return {
        "name": name,
        "type": p.get("poi_type") or p.get("type") or "tourist_spot",
        "address": address,
        "lat": p.get("lat"),
        "lng": p.get("lng"),
        "stay_minutes": int(float(p.get("estimated_stay_time") or p.get("stay_minutes") or 60)),
        "notes": "",
        "area": area,
        "source_kind": "course",
    }


def _poi_from_google_item(p: dict[str, Any]) -> dict[str, Any]:
    area = p.get("area") or _infer_area_from_text_or_coords(
        p.get("poi_name"),
        p.get("address_en") or p.get("address_ko"),
        p.get("lat"),
        p.get("lng"),
    )

    return {
        "name": p.get("poi_name", ""),
        "type": p.get("poi_type", "tourist_spot"),
        "address": p.get("address_en") or p.get("address_ko") or "",
        "lat": p.get("lat"),
        "lng": p.get("lng"),
        "stay_minutes": int(float(p.get("estimated_stay_time") or 60)),
        "notes": _google_note_for_type(p),
        "area": area,
        "source_kind": "google",
    }


def _google_note_for_type(p: dict[str, Any]) -> str:
    ptype = p.get("poi_type", "")
    area = _area_label(p.get("area", ""))
    rating = p.get("rating")
    rating_text = f" It has a Google rating of {rating}." if rating else ""

    if ptype == "cafe":
        return f"Verified cafe in {area}; good for cafe hopping and a relaxed break.{rating_text}"
    if ptype == "restaurant":
        return f"Verified restaurant in {area}; useful for a clear meal slot in the itinerary.{rating_text}"
    if ptype == "kpop_landmark":
        return f"Verified K-POP related place around {area}; fits the user's interest in idols and Hallyu culture.{rating_text}"
    if ptype in {"shopping_mall", "shopping"}:
        return f"Verified shopping spot in {area}; fits shopping and local trend exploration.{rating_text}"
    return f"Verified Google Places POI in {area}.{rating_text}"


def _build_candidate_pool(
    courses: list[dict[str, Any]],
    google_supplement: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}

    for c in courses:
        for raw in c.get("sequence", []) or []:
            item = _poi_from_course_item(raw)
            key = _normalize_text(item["name"])
            if key:
                pool[key] = item

    for raw in google_supplement or []:
        item = _poi_from_google_item(raw)
        key = _normalize_text(item["name"])
        if key:
            pool[key] = item

    return pool


def _as_output_poi(item: dict[str, Any], extra_note: str | None = None) -> dict[str, Any]:
    notes = item.get("notes") or ""
    if extra_note:
        notes = f"{notes} {extra_note}".strip()

    return {
        "name": item.get("name", ""),
        "type": item.get("type", "tourist_spot"),
        "address": item.get("address", ""),
        "lat": item.get("lat"),
        "lng": item.get("lng"),
        "stay_minutes": int(float(item.get("stay_minutes") or 60)),
        "notes": notes,
        "area": item.get("area"),
    }


def _poi_area(poi: dict[str, Any]) -> str | None:
    if poi.get("area"):
        return str(poi.get("area")).lower()
    return _infer_area_from_text_or_coords(
        poi.get("name"),
        poi.get("address"),
        poi.get("lat"),
        poi.get("lng"),
    )


def _is_meal_poi(poi: dict[str, Any]) -> bool:
    ptype = _normalize_text(poi.get("type"))
    name = _normalize_text(poi.get("name"))
    return (
        ptype in {"restaurant", "cafe", "market", "food", "meal_takeaway"}
        or "restaurant" in ptype
        or "cafe" in ptype
        or "coffee" in name
    )


def _candidate_items_for_area(
    pool: dict[str, dict[str, Any]],
    area: str,
    *,
    preferred_types: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude_names = exclude_names or set()
    preferred_types = preferred_types or set()

    items: list[dict[str, Any]] = []

    for item in pool.values():
        name_key = _normalize_text(item.get("name"))
        if name_key in exclude_names:
            continue

        item_area = item.get("area")
        if not _area_matches_requested(item_area, area):
            continue

        if preferred_types:
            ptype = _normalize_text(item.get("type"))
            if not any(t in ptype for t in preferred_types):
                continue

        items.append(item)

    # Prefer Google Places and higher relevance.
    def sort_key(x: dict[str, Any]) -> tuple[int, int]:
        source_score = 0 if x.get("source_kind") == "google" else 1
        type_score = 0
        ptype = _normalize_text(x.get("type"))
        if ptype in {"cafe", "restaurant", "kpop_landmark", "shopping_mall", "shopping"}:
            type_score = -1
        return (source_score, type_score)

    return sorted(items, key=sort_key)


def _validate_and_repair_itinerary(
    itinerary: dict[str, Any],
    *,
    courses: list[dict[str, Any]],
    google_supplement: list[dict[str, Any]],
    requested_areas: list[str],
) -> dict[str, Any]:
    """Remove hallucinations and force requested area coverage."""
    pool = _build_candidate_pool(courses, google_supplement)
    valid_names = set(pool.keys())
    used_names: set[str] = set()

    days = itinerary.get("days") or []
    if not isinstance(days, list):
        days = []
    itinerary["days"] = days

    # 1. Remove hallucinated POIs.
    for day in days:
        original = day.get("pois") or []
        valid_pois: list[dict[str, Any]] = []
        removed: list[str] = []

        for poi in original:
            name_key = _normalize_text(poi.get("name"))
            if name_key in valid_names:
                # Normalize with canonical candidate data if possible.
                candidate = pool[name_key]
                out = _as_output_poi(candidate)
                # Preserve the model's note if useful.
                if poi.get("notes"):
                    out["notes"] = poi.get("notes")
                valid_pois.append(out)
                used_names.add(name_key)
            else:
                removed.append(str(poi.get("name", "")))

        if removed:
            print(f"[Validator] Day {day.get('day')} hallucinated POI 제거: {removed}")

        day["pois"] = valid_pois

    # 2. Force requested area coverage.
    if requested_areas and days:
        coverage = _area_coverage(days, requested_areas)

        for idx, area in enumerate(requested_areas):
            current_count = coverage.get(area, 0)
            if current_count >= 2:
                continue

            target_day_idx = min(idx, len(days) - 1)
            target_day = days[target_day_idx]

            needed = 2 - current_count
            candidates = _candidate_items_for_area(
                pool,
                area,
                exclude_names=used_names,
            )

            inserted = 0
            for item in candidates:
                if inserted >= needed:
                    break
                out = _as_output_poi(
                    item,
                    extra_note=f"Added to ensure the itinerary covers the requested area: {_area_label(area)}."
                )
                target_day.setdefault("pois", []).append(out)
                used_names.add(_normalize_text(out.get("name")))
                inserted += 1

            if inserted:
                print(f"[Validator] {_area_label(area)} 누락 보완: {inserted}개 POI 추가")

    # 3. Ensure each day has a meal slot.
    for day in days:
        pois = day.setdefault("pois", [])
        if any(_is_meal_poi(p) for p in pois):
            continue

        day_area = _primary_area_for_day(day, requested_areas)
        candidates = _candidate_items_for_area(
            pool,
            day_area,
            preferred_types={"restaurant", "cafe"},
            exclude_names=used_names,
        ) if day_area else []

        if not candidates:
            candidates = [
                item for item in pool.values()
                if _normalize_text(item.get("name")) not in used_names
                and _normalize_text(item.get("type")) in {"restaurant", "cafe"}
            ]

        if candidates:
            item = candidates[0]
            out = _as_output_poi(item, extra_note="Added as a clear meal or cafe slot.")
            insert_idx = min(2, len(pois))
            pois.insert(insert_idx, out)
            used_names.add(_normalize_text(out.get("name")))
            print(f"[Validator] Day {day.get('day')} 식사 슬롯 추가: {out.get('name')}")

    # 4. Fill under-populated days up to 5 POIs.
    for idx, day in enumerate(days):
        pois = day.setdefault("pois", [])
        if len(pois) >= 5:
            continue

        target_area = None
        if requested_areas:
            target_area = requested_areas[min(idx, len(requested_areas) - 1)]
        target_area = target_area or _primary_area_for_day(day, requested_areas)

        candidates = []
        if target_area:
            candidates = _candidate_items_for_area(
                pool,
                target_area,
                exclude_names=used_names,
            )

        if not candidates:
            candidates = [
                item for item in pool.values()
                if _normalize_text(item.get("name")) not in used_names
            ]

        while len(pois) < 5 and candidates:
            item = candidates.pop(0)
            out = _as_output_poi(item, extra_note="Added to make the day sufficiently complete.")
            pois.append(out)
            used_names.add(_normalize_text(out.get("name")))
            print(f"[Validator] Day {day.get('day')} POI 수 보완: {out.get('name')}")

    # 5. Reorder each day lightly by area grouping, preserving the LLM order mostly.
    for day in days:
        day["pois"] = day.get("pois") or []

    itinerary["requested_areas"] = requested_areas
    itinerary["area_coverage"] = _area_coverage(days, requested_areas)

    return itinerary


def _area_coverage(days: list[dict[str, Any]], requested_areas: list[str]) -> dict[str, int]:
    coverage = {area: 0 for area in requested_areas}
    for day in days:
        for poi in day.get("pois", []) or []:
            area = _poi_area(poi)
            for req in requested_areas:
                if _area_matches_requested(area, req):
                    coverage[req] += 1
    return coverage


def _primary_area_for_day(day: dict[str, Any], requested_areas: list[str]) -> str | None:
    if requested_areas:
        day_num = int(day.get("day") or 1)
        idx = min(max(day_num - 1, 0), len(requested_areas) - 1)
        return requested_areas[idx]

    counts: dict[str, int] = {}
    for poi in day.get("pois", []) or []:
        area = _poi_area(poi)
        if area:
            counts[area] = counts.get(area, 0) + 1

    if not counts:
        return None

    return max(counts.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _isolate_json_object(text: str) -> str:
    text = _FENCE_RE.sub("", text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def _simple_repair(text: str) -> str:
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _parse_itinerary_json(raw: str, *, use_llm_fallback: bool = True) -> dict[str, Any]:
    isolated = _isolate_json_object(raw)

    try:
        return json.loads(isolated)
    except json.JSONDecodeError as first_err:
        repaired = _simple_repair(isolated)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as second_err:
        if use_llm_fallback:
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
        raise second_err from first_err


def _dump_debug(raw: str) -> None:
    try:
        dbg_path = Path(__file__).resolve().parent / "planner_last_failed.txt"
        dbg_path.write_text(raw or "", encoding="utf-8")
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
        if not cid or cid in seen_ids:
            continue

        seen_ids.add(cid)
        cleaned.append({
            "course_id": cid,
            "course_title": match.get("course_title", ""),
            "source": match.get("source", ""),
            "source_url": match.get("source_url", ""),
        })

    if not cleaned and retrieved:
        for c in retrieved:
            if c.get("source_url"):
                cleaned.append({
                    "course_id": c.get("course_id"),
                    "course_title": c.get("course_title", ""),
                    "source": c.get("source", ""),
                    "source_url": c.get("source_url", ""),
                })

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
    duration = state.get("duration") or ""
    budget = state.get("budget") or ""
    dietary = state.get("dietary") or "none"

    requested_areas = _extract_requested_areas(location, purpose)
    print(f"[planner] requested_areas = {requested_areas}")

    google_supplement: list[dict[str, Any]] = []
    if GOOGLE_PLACES_API_KEY:
        google_supplement = build_google_supplement_by_areas(
            requested_areas=requested_areas,
            location=location,
            purpose=purpose,
            api_key=GOOGLE_PLACES_API_KEY,
        )
    else:
        print("[planner] GOOGLE_PLACES_API_KEY 없음 — Google Places 보완 생략")

    prompt_context = _format_courses_for_prompt(
        courses,
        google_supplement=google_supplement,
        requested_areas=requested_areas,
        duration=duration,
    )

    try:
        with lm_context():
            result = get_planner()(
                duration=duration,
                location=location,
                budget=budget,
                dietary=dietary,
                purpose=purpose,
                candidate_courses=prompt_context,
            )

        itinerary = _parse_itinerary_json(result.itinerary_json)

        itinerary = _validate_and_repair_itinerary(
            itinerary,
            courses=courses,
            google_supplement=google_supplement,
            requested_areas=requested_areas,
        )

        itinerary = _normalize_sources(itinerary, courses)

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

    summary = itinerary.get("summary", "")
    day_count = len(itinerary.get("days", []))
    area_text = ", ".join(_area_label(a) for a in requested_areas) if requested_areas else "Seoul"

    ack = (
        f"✅ Your {day_count}-day itinerary is ready!\n\n"
        f"{summary}\n\n"
        f"Requested area coverage checked: {area_text}.\n\n"
        "See the full plan below."
    )

    return {
        **state,
        "itinerary": itinerary,
        "planning_context": {
            "requested_areas": requested_areas,
            "google_supplement": google_supplement,
        },
        "current_step": "done",
        "messages": [AIMessage(content=ack)],
    }
