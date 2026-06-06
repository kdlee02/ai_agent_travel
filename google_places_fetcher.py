"""
google_places_fetcher.py

replanner.py가 meals pool이 비었을 때 직접 호출하는
Google Places 식당/카페 검색 모듈.

사용법:
    from google_places_fetcher import fetch_nearby_restaurants

    restaurants = fetch_nearby_restaurants(
        lat=37.5563, lng=126.9227,
        api_key="YOUR_KEY",
        dietary_restrictions=["seafood"],
        radius=1000,
        max_results=5,
    )
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# .env에서 API key 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

GOOGLE_PLACES_API_KEY = (
    os.getenv("GOOGLE_PLACES_API_KEY")
    or os.getenv("GOOGLE_MAPS_API_KEY")
    or ""
)

# ============================================================
# Dietary 필터
# ============================================================

DIETARY_EXCLUDE_KEYWORDS: dict[str, list[str]] = {
    "seafood":    ["seafood", "fish", "sushi", "sashimi", "crab", "shrimp", "lobster",
                   "해산물", "생선", "초밥", "회", "게", "새우", "랍스터", "해물"],
    "pork":       ["pork", "pig", "bacon", "ham", "삼겹살", "돼지", "베이컨", "햄", "족발"],
    "beef":       ["beef", "steak", "소고기", "스테이크", "육회", "한우"],
    "meat":       ["meat", "chicken", "beef", "pork", "육류", "고기", "닭", "소", "돼지"],
    "vegetarian": [],
    "vegan":      [],
    "halal":      ["pork", "돼지", "bacon", "ham", "삼겹살"],
    "nut":        ["nut", "peanut", "almond", "견과류", "땅콩"],
    "gluten":     ["ramen", "noodle", "bread", "pasta", "라멘", "국수", "빵", "파스타"],
}


def parse_dietary_restrictions(dietary: str) -> list[str]:
    if not dietary or dietary.lower() in {"none", "no", "없음", "없어요"}:
        return []
    d = dietary.lower()
    return [key for key in DIETARY_EXCLUDE_KEYWORDS if key in d]


def violates_dietary(name: str, types: list[str], restrictions: list[str]) -> bool:
    if not restrictions:
        return False
    combined = f"{name.lower()} {' '.join(types).lower()}"
    for r in restrictions:
        for word in DIETARY_EXCLUDE_KEYWORDS.get(r, []):
            if word in combined:
                return True
    return False


# ============================================================
# Google Places 호출
# ============================================================

def fetch_nearby_restaurants(
    lat: float,
    lng: float,
    api_key: str = "",
    dietary_restrictions: list[str] | None = None,
    radius: int = 1000,
    min_rating: float = 4.0,
    max_results: int = 5,
    place_type: str = "restaurant",
) -> list[dict[str, Any]]:
    """
    Google Places Nearby Search로 인근 식당/카페 검색.

    Parameters
    ----------
    lat, lng              : 검색 중심 좌표 (anchor POI 기준)
    api_key               : Google Places API key
    dietary_restrictions  : 제외할 음식 종류 (parse_dietary_restrictions 결과)
    radius                : 검색 반경 (미터)
    min_rating            : 최소 rating
    max_results           : 최대 결과 수
    place_type            : "restaurant" or "cafe"

    Returns
    -------
    list of dicts with keys: poi_name, poi_type, lat, lng, rating,
                              user_ratings_total, address_en, source
    """
    if not api_key:
        api_key = GOOGLE_PLACES_API_KEY
    if not api_key:
        return []
    if not _REQUESTS_OK:
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
    except Exception as e:
        print(f"[Google Places] 오류: {e}")
        return []

    out = []
    for r in results:
        # 운영 중인 곳만
        if r.get("business_status") and r["business_status"] != "OPERATIONAL":
            continue
        # rating 필터
        if r.get("rating", 0) < min_rating:
            continue
        # dietary 필터
        name = r.get("name", "")
        types = r.get("types", [])
        if violates_dietary(name, types, dietary_restrictions):
            continue

        out.append({
            "poi_name": name,
            "poi_type": place_type,
            "lat": r["geometry"]["location"]["lat"],
            "lng": r["geometry"]["location"]["lng"],
            "rating": r.get("rating"),
            "user_ratings_total": r.get("user_ratings_total", 0),
            "address_en": r.get("vicinity", ""),
            "place_id": r.get("place_id", ""),
            "opening_hours": r.get("opening_hours"),
            "source": "Google Places (live)",
            "dietary_safe": True,
        })

    # 카페는 리뷰 수(외국인 많이 가는 곳), 식당은 rating 순
    if place_type == "cafe":
        out.sort(key=lambda x: x.get("user_ratings_total", 0), reverse=True)
    else:
        out.sort(key=lambda x: x.get("rating") or 0, reverse=True)

    return out[:max_results]


def fetch_restaurants_for_area(
    area_key: str,
    anchor_lat: float,
    anchor_lng: float,
    api_key: str = "",
    dietary_restrictions: list[str] | None = None,
    radius: int = 1000,
) -> list[dict[str, Any]]:
    """
    replanner가 meals pool 비었을 때 호출하는 편의 함수.
    식당 5개 + 카페 3개를 가져옴.

    dietary 제한이 있을 때(vegetarian 등)는 min_reviews를 50으로 낮춰서
    조건 맞는 식당이 부족한 문제를 완화.
    """
    if not api_key:
        api_key = GOOGLE_PLACES_API_KEY
    if not api_key:
        print(f"[Google Places] API key 없음 — {area_key} 식당 검색 스킵")
        return []

    dietary_restrictions = dietary_restrictions or []
    results = []

    # dietary 제한 있으면 min_reviews 완화 (채식/할랄 식당이 적은 문제 대응)
    has_dietary = bool(dietary_restrictions)
    restaurant_min_reviews = 50 if has_dietary else 100
    restaurant_min_rating = 4.0 if has_dietary else 4.3

    # 식당
    restaurants = fetch_nearby_restaurants(
        lat=anchor_lat, lng=anchor_lng,
        api_key=api_key,
        dietary_restrictions=dietary_restrictions,
        radius=radius,
        min_rating=restaurant_min_rating,
        max_results=5,
        place_type="restaurant",
    )
    # dietary 있는데 결과 부족하면 반경 확장 재시도
    if has_dietary and len(restaurants) < 2:
        restaurants = fetch_nearby_restaurants(
            lat=anchor_lat, lng=anchor_lng,
            api_key=api_key,
            dietary_restrictions=dietary_restrictions,
            radius=radius * 2,
            min_rating=3.8,
            max_results=5,
            place_type="restaurant",
        )
    results.extend(restaurants)

    # 카페
    cafes = fetch_nearby_restaurants(
        lat=anchor_lat, lng=anchor_lng,
        api_key=api_key,
        dietary_restrictions=[],
        radius=radius,
        min_rating=4.2,
        max_results=3,
        place_type="cafe",
    )
    results.extend(cafes)

    print(f"[Google Places] {area_key} 인근 식당 {len(restaurants)}개, 카페 {len(cafes)}개 검색 "
          f"(dietary={dietary_restrictions or 'none'})")
    return results


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    key = GOOGLE_PLACES_API_KEY
    if not key:
        print("GOOGLE_PLACES_API_KEY 없음. .env 파일 확인.")
    else:
        print("홍대 인근 식당 검색 테스트...")
        results = fetch_restaurants_for_area(
            area_key="hongdae_area",
            anchor_lat=37.5563,
            anchor_lng=126.9227,
            api_key=key,
            dietary_restrictions=[],
            radius=800,
        )
        for r in results:
            print(f"  {r['poi_name']} ({r['poi_type']}) rating={r['rating']} | {r['address_en']}")