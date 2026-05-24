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

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import dspy
import requests
from langchain_core.messages import AIMessage

from llm import lm_context
from rag import build_query, retrieve_courses
from state import TravelState


# ---------------------------------------------------------------------------
# Google Places 설정
# ---------------------------------------------------------------------------

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

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


def fetch_nearby_places(
    lat: float,
    lng: float,
    place_type: str,
    api_key: str,
    radius: int = 1500,
    min_rating: float = 4.0,
    max_results: int = 5,
) -> list[dict]:
    """
    Google Places Nearby Search로 주변 장소 검색.
    place_type: "cafe" | "restaurant" | "bar"
    """
    if not api_key:
        return []
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
        filtered = [r for r in results if r.get("rating", 0) >= min_rating]
        return [
            {
                "poi_name": r["name"],
                "poi_type": place_type,
                "address_en": r.get("vicinity", ""),
                "address_ko": r.get("vicinity", ""),
                "lat": r["geometry"]["location"]["lat"],
                "lng": r["geometry"]["location"]["lng"],
                "rating": r.get("rating"),
                "estimated_stay_time": 60 if place_type == "restaurant" else 45,
                "source": "Google Places",
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
) -> list[dict]:
    """
    사용자 목적에 맞게 Google Places 보완 데이터 수집.
    카페, 식당, K-POP 장소 등을 실시간으로 가져옴.
    """
    if not api_key:
        return []

    center_lat, center_lng = _get_area_center(location)
    supplement = []
    purpose_lower = purpose.lower()

    # 카페 보완
    if any(k in purpose_lower for k in ["cafe", "coffee", "카페"]):
        cafes = fetch_nearby_places(
            center_lat, center_lng, "cafe", api_key,
            radius=1500, min_rating=4.2, max_results=5
        )
        supplement.extend(cafes)
        print(f"[Google Places] 카페 {len(cafes)}개 추가")

    # 식당 보완 (항상 추가 - 식사 슬롯용)
    restaurants = fetch_nearby_places(
        center_lat, center_lng, "restaurant", api_key,
        radius=1500, min_rating=4.0, max_results=5
    )
    supplement.extend(restaurants)
    print(f"[Google Places] 식당 {len(restaurants)}개 추가")

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
    """Google Places 보완 데이터를 프롬프트용 텍스트로 변환."""
    if not places:
        return ""
    lines = ["\n\n=== REAL-TIME GOOGLE PLACES DATA (USE THESE FOR CAFES/RESTAURANTS/KPOP) ==="]
    lines.append("These are verified real places. Prioritize these for cafe/restaurant/kpop slots.\n")
    for p in places:
        rating_str = f"rating={p['rating']}" if p.get("rating") else ""
        lines.append(
            f"  - {p['poi_name']} [{p['poi_type']}] "
            f"addr={p['address_en']} "
            f"lat={p['lat']} lng={p['lng']} "
            f"stay={p['estimated_stay_time']}min "
            f"{rating_str} source={p['source']}"
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

      DATA INTEGRITY RULES:
      - For sightseeing/parks/museums: use POIs from candidate_courses ONLY.
      - For cafes/restaurants/kpop: PREFER Google Places data (more accurate).
      - Do NOT invent or hallucinate POI names, addresses, or coordinates.
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
    hallucinated POI 제거 후 식사 슬롯 부족하면 Google Places 식당으로 보완.
    """
    google_restaurants = [
        p for p in (google_supplement or [])
        if p.get("poi_type") in ["restaurant", "cafe"]
    ]
    google_by_name = {
        p["poi_name"].lower(): p for p in (google_supplement or [])
    }

    for day in itinerary.get("days", []):
        original_pois = day.get("pois", [])
        validated = []
        hallucinated_names = []

        for poi in original_pois:
            poi_name = str(poi.get("name", "")).lower().strip()
            # 정확 매칭 또는 부분 매칭
            is_valid = (
                poi_name in valid_names
                or any(poi_name in v or v in poi_name for v in valid_names)
            )
            if is_valid:
                validated.append(poi)
            else:
                hallucinated_names.append(poi.get("name", ""))

        if hallucinated_names:
            print(f"[Validator] Day {day.get('day')} hallucinated POI 제거: {hallucinated_names}")

        # 식사 슬롯 체크 - restaurant/cafe가 없으면 Google Places에서 추가
        has_meal = any(
            p.get("type") in ["restaurant", "cafe", "market"]
            for p in validated
        )
        if not has_meal and google_restaurants:
            best = google_restaurants[0]
            meal_poi = {
                "name": best["poi_name"],
                "type": best["poi_type"],
                "address": best.get("address_en", ""),
                "lat": best["lat"],
                "lng": best["lng"],
                "stay_minutes": best.get("estimated_stay_time", 60),
                "notes": "Recommended restaurant nearby. Great for lunch or dinner.",
            }
            # 중간에 삽입 (3번째 POI 뒤)
            insert_idx = min(2, len(validated))
            validated.insert(insert_idx, meal_poi)
            print(f"[Validator] Day {day.get('day')} 식사 슬롯 자동 추가: {best['poi_name']}")

        day["pois"] = validated

    return itinerary


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
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as second_err:
        pass
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
    raise second_err


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
            courses = retrieve_courses(api_key=api_key, query=query, k=5)
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

    # Google Places로 카페/식당/K-POP 실시간 보완
    google_supplement = []
    if GOOGLE_PLACES_API_KEY:
        print("[planner] Google Places 보완 데이터 수집 중...")
        google_supplement = build_google_supplement(
            location=location,
            purpose=purpose,
            api_key=GOOGLE_PLACES_API_KEY,
        )
        print(f"[planner] Google Places 총 {len(google_supplement)}개 보완 데이터 확보")
    else:
        print("[planner] GOOGLE_PLACES_API_KEY 없음 — Google Places 보완 생략")

    # 유효한 POI 이름 집합 (hallucination 검증용)
    valid_names = _build_valid_poi_names(courses, google_supplement)

    try:
        with lm_context():
            result = get_planner()(
                duration=state.get("duration") or "",
                location=location,
                budget=state.get("budget") or "",
                dietary=state.get("dietary") or "none",
                purpose=purpose,
                candidate_courses=_format_courses_for_prompt(courses, google_supplement),
            )
        itinerary = _parse_itinerary_json(result.itinerary_json)

        # Hallucination 검증 + 식사 슬롯 보완
        itinerary = _validate_and_fix_pois(itinerary, valid_names, google_supplement)

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
