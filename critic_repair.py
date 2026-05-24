"""Critic-Repair node for SeoulMate LangGraph.

This module evaluates and repairs generated itineraries.

Main checks:
1. Requested area coverage: e.g. Hongdae + Seongsu must both appear.
2. Per-day POI count: each day should have at least 5 POIs.
3. Meal slot: each day should include cafe/restaurant/market.
4. Duplicate POIs.
5. Basic geographic coherence.
6. Basic foreigner readiness from available labels if present.

The node is intentionally robust:
- It does not crash on NaN.
- It works even if enriched fields are missing.
- It can repair using planning_context["google_supplement"] and retrieved_courses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

try:
    from state import TravelState
except Exception:
    TravelState = dict


# ---------------------------------------------------------------------------
# Area configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Safe utilities
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        text = str(value)
        if text.lower() in {"nan", "none", "null", ""}:
            return default
        return text
    except Exception:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null", ""}:
            return default
        return float(text)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null", ""}:
            return default
        return int(float(text))
    except Exception:
        return default


def area_label(area: str | None) -> str:
    if not area:
        return "Unknown"
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


def extract_requested_areas(location: str | None, purpose: str | None = None) -> list[str]:
    text = f"{location or ''} {purpose or ''}".lower()
    found: list[str] = []

    for area, aliases in AREA_ALIASES.items():
        if any(alias.lower() in text for alias in aliases):
            if area not in found:
                found.append(area)

    return found


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
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


def infer_area_from_poi(poi: dict[str, Any]) -> str | None:
    if poi.get("area"):
        area = normalize_text(poi.get("area"))
        if area in SEOUL_AREA_CENTERS:
            return area

    text = f"{poi.get('name', '')} {poi.get('address', '')} {poi.get('poi_name', '')} {poi.get('address_en', '')} {poi.get('address_ko', '')}".lower()

    for area, aliases in AREA_ALIASES.items():
        if area in text or any(alias.lower() in text for alias in aliases):
            return area

    lat = safe_float(poi.get("lat"))
    lng = safe_float(poi.get("lng"))
    if lat is None or lng is None:
        return None

    nearest_area = None
    nearest_dist = 9999.0

    for area, (center_lat, center_lng) in SEOUL_AREA_CENTERS.items():
        dist = haversine_km(lat, lng, center_lat, center_lng)
        if dist < nearest_dist:
            nearest_area = area
            nearest_dist = dist

    if nearest_dist <= 3.2:
        return nearest_area

    return None


def area_matches_requested(area: str | None, requested: str) -> bool:
    if not area:
        return False

    area = normalize_text(area)
    requested = normalize_text(requested)

    if area == requested:
        return True

    adjacent = {
        "hongdae": {"hongdae", "hapjeong", "mangwon", "yeonnam", "mapo"},
        "seongsu": {"seongsu", "wangsimni"},
        "gangnam": {"gangnam", "sinsa", "garosu-gil"},
        "jongno": {"jongno", "insadong", "myeongdong"},
    }

    return area in adjacent.get(requested, {requested})


def is_meal_poi(poi: dict[str, Any]) -> bool:
    ptype = normalize_text(poi.get("type") or poi.get("poi_type"))
    name = normalize_text(poi.get("name") or poi.get("poi_name"))
    return (
        ptype in {"restaurant", "cafe", "market", "food", "meal_takeaway"}
        or "restaurant" in ptype
        or "cafe" in ptype
        or "coffee" in name
        or "market" in ptype
    )


def poi_name(poi: dict[str, Any]) -> str:
    return safe_str(poi.get("name") or poi.get("poi_name"))


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def candidate_from_course_poi(raw: dict[str, Any]) -> dict[str, Any]:
    name = raw.get("poi_name") or raw.get("name") or ""
    ptype = raw.get("poi_type") or raw.get("type") or "tourist_spot"
    address = raw.get("address_en") or raw.get("address_ko") or raw.get("address") or ""
    lat = raw.get("lat")
    lng = raw.get("lng")

    item = {
        "name": name,
        "type": ptype,
        "address": address,
        "lat": lat,
        "lng": lng,
        "stay_minutes": safe_int(raw.get("estimated_stay_time") or raw.get("stay_minutes"), 60),
        "notes": "",
        "area": raw.get("area"),
        "source_kind": "course",
    }
    item["area"] = item["area"] or infer_area_from_poi(item)
    return item


def candidate_from_google(raw: dict[str, Any]) -> dict[str, Any]:
    name = raw.get("poi_name") or raw.get("name") or ""
    ptype = raw.get("poi_type") or raw.get("type") or "tourist_spot"
    address = raw.get("address_en") or raw.get("address_ko") or raw.get("address") or ""
    lat = raw.get("lat")
    lng = raw.get("lng")

    item = {
        "name": name,
        "type": ptype,
        "address": address,
        "lat": lat,
        "lng": lng,
        "stay_minutes": safe_int(raw.get("estimated_stay_time") or raw.get("stay_minutes"), 60),
        "notes": google_note(raw),
        "area": raw.get("area"),
        "source_kind": "google",
    }
    item["area"] = item["area"] or infer_area_from_poi(item)
    return item


def google_note(raw: dict[str, Any]) -> str:
    ptype = normalize_text(raw.get("poi_type") or raw.get("type"))
    area = area_label(raw.get("area"))
    rating = raw.get("rating")
    rating_text = f" Google rating: {rating}." if rating else ""

    if ptype == "cafe":
        return f"Verified cafe in {area}; useful for cafe hopping and a relaxed break.{rating_text}"
    if ptype == "restaurant":
        return f"Verified restaurant in {area}; added to make the day executable with a clear meal slot.{rating_text}"
    if ptype == "kpop_landmark":
        return f"Verified K-POP related place in or near {area}; fits Hallyu and idol interests.{rating_text}"
    if ptype in {"shopping", "shopping_mall"}:
        return f"Verified shopping spot in {area}; fits shopping and trend exploration.{rating_text}"

    return f"Verified place in {area}.{rating_text}"


def build_candidate_pool(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}

    for course in state.get("retrieved_courses") or []:
        for raw in course.get("sequence", []) or []:
            item = candidate_from_course_poi(raw)
            key = normalize_text(item.get("name"))
            if key:
                pool[key] = item

    planning_context = state.get("planning_context") or {}
    for raw in planning_context.get("google_supplement") or []:
        item = candidate_from_google(raw)
        key = normalize_text(item.get("name"))
        if key:
            pool[key] = item

    return pool


def as_output_poi(item: dict[str, Any], note_suffix: str = "") -> dict[str, Any]:
    notes = safe_str(item.get("notes"))
    if note_suffix:
        notes = f"{notes} {note_suffix}".strip()

    return {
        "name": item.get("name", ""),
        "type": item.get("type", "tourist_spot"),
        "address": item.get("address", ""),
        "lat": item.get("lat"),
        "lng": item.get("lng"),
        "stay_minutes": safe_int(item.get("stay_minutes"), 60),
        "notes": notes,
        "area": item.get("area"),
    }


def used_name_set(itinerary: dict[str, Any]) -> set[str]:
    used = set()
    for day in itinerary.get("days") or []:
        for poi in day.get("pois") or []:
            key = normalize_text(poi_name(poi))
            if key:
                used.add(key)
    return used


def candidates_for_area(
    pool: dict[str, dict[str, Any]],
    area: str,
    *,
    exclude: set[str] | None = None,
    preferred_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude = exclude or set()
    preferred_types = preferred_types or set()

    items: list[dict[str, Any]] = []

    for item in pool.values():
        name_key = normalize_text(item.get("name"))
        if name_key in exclude:
            continue

        item_area = item.get("area")
        if not area_matches_requested(item_area, area):
            continue

        if preferred_types:
            ptype = normalize_text(item.get("type"))
            if not any(t in ptype for t in preferred_types):
                continue

        items.append(item)

    def sort_key(x: dict[str, Any]) -> tuple[int, int]:
        # Prefer Google current places, then food/cafe/kpop/shopping.
        source_score = 0 if x.get("source_kind") == "google" else 1
        ptype = normalize_text(x.get("type"))
        type_score = 0
        if ptype in {"cafe", "restaurant", "kpop_landmark", "shopping", "shopping_mall"}:
            type_score = -1
        return (source_score, type_score)

    return sorted(items, key=sort_key)


# ---------------------------------------------------------------------------
# Critic data structures
# ---------------------------------------------------------------------------

@dataclass
class CriticIssue:
    code: str
    severity: str
    message: str
    day: int | None = None
    area: str | None = None


class CriticAgent:
    def evaluate(self, state: dict[str, Any]) -> dict[str, Any]:
        itinerary = state.get("itinerary") or {}
        requested_areas = self._get_requested_areas(state)

        issues: list[CriticIssue] = []

        area_report = self._evaluate_area_coverage(itinerary, requested_areas, issues)
        day_report = self._evaluate_days(itinerary, requested_areas, issues)
        duplicate_report = self._evaluate_duplicates(itinerary, issues)
        foreigner_report = self._evaluate_foreigner_readiness(itinerary, issues)

        feasibility_score = day_report["score"]
        area_score = area_report["score"]
        duplicate_score = duplicate_report["score"]
        foreigner_score = foreigner_report["score"]

        overall = round(
            0.35 * feasibility_score
            + 0.35 * area_score
            + 0.15 * duplicate_score
            + 0.15 * foreigner_score,
            3,
        )

        return {
            "overall_score": overall,
            "feasibility_score": feasibility_score,
            "requested_area_coverage_score": area_score,
            "duplicate_score": duplicate_score,
            "foreigner_readiness_score": foreigner_score,
            "requested_areas": requested_areas,
            "area_coverage": area_report["coverage"],
            "missing_areas": area_report["missing_areas"],
            "issues": [issue.__dict__ for issue in issues],
        }

    def _get_requested_areas(self, state: dict[str, Any]) -> list[str]:
        planning_context = state.get("planning_context") or {}
        requested = planning_context.get("requested_areas")
        if requested:
            return list(requested)

        return extract_requested_areas(
            state.get("location"),
            state.get("purpose"),
        )

    def _evaluate_area_coverage(
        self,
        itinerary: dict[str, Any],
        requested_areas: list[str],
        issues: list[CriticIssue],
    ) -> dict[str, Any]:
        if not requested_areas:
            return {
                "score": 1.0,
                "coverage": {},
                "missing_areas": [],
            }

        coverage = {area: 0 for area in requested_areas}

        for day in itinerary.get("days") or []:
            for poi in day.get("pois") or []:
                area = infer_area_from_poi(poi)
                for req in requested_areas:
                    if area_matches_requested(area, req):
                        coverage[req] += 1

        missing = [area for area, count in coverage.items() if count < 2]

        for area in missing:
            issues.append(CriticIssue(
                code="REQUESTED_AREA_UNDER_COVERED",
                severity="high",
                message=f"Requested area {area_label(area)} has fewer than 2 POIs.",
                area=area,
            ))

        score = 1.0
        if requested_areas:
            satisfied = sum(1 for area in requested_areas if coverage.get(area, 0) >= 2)
            score = satisfied / len(requested_areas)

        return {
            "score": round(score, 3),
            "coverage": coverage,
            "missing_areas": missing,
        }

    def _evaluate_days(
        self,
        itinerary: dict[str, Any],
        requested_areas: list[str],
        issues: list[CriticIssue],
    ) -> dict[str, Any]:
        days = itinerary.get("days") or []
        if not days:
            issues.append(CriticIssue(
                code="NO_DAYS",
                severity="critical",
                message="No days were generated in the itinerary.",
            ))
            return {"score": 0.0}

        penalties = 0.0
        checks = 0

        for day in days:
            day_num = safe_int(day.get("day"), 0)
            pois = day.get("pois") or []

            checks += 1
            if len(pois) < 5:
                penalties += 0.35
                issues.append(CriticIssue(
                    code="TOO_FEW_POIS",
                    severity="medium",
                    message=f"Day {day_num} has only {len(pois)} POIs; at least 5 are recommended.",
                    day=day_num,
                ))

            checks += 1
            if not any(is_meal_poi(p) for p in pois):
                penalties += 0.25
                issues.append(CriticIssue(
                    code="NO_MEAL_SLOT",
                    severity="medium",
                    message=f"Day {day_num} has no clear cafe/restaurant/market slot.",
                    day=day_num,
                ))

            checks += 1
            if self._day_is_geographically_scattered(pois):
                penalties += 0.20
                issues.append(CriticIssue(
                    code="SCATTERED_DAY_ROUTE",
                    severity="medium",
                    message=f"Day {day_num} appears geographically scattered.",
                    day=day_num,
                ))

        if checks == 0:
            return {"score": 0.0}

        score = max(0.0, 1.0 - penalties / max(len(days), 1))
        return {"score": round(score, 3)}

    def _day_is_geographically_scattered(self, pois: list[dict[str, Any]]) -> bool:
        coords: list[tuple[float, float]] = []

        for p in pois:
            lat = safe_float(p.get("lat"))
            lng = safe_float(p.get("lng"))
            if lat is not None and lng is not None:
                coords.append((lat, lng))

        if len(coords) < 2:
            return False

        max_dist = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                max_dist = max(max_dist, dist)

        return max_dist > 8.0

    def _evaluate_duplicates(
        self,
        itinerary: dict[str, Any],
        issues: list[CriticIssue],
    ) -> dict[str, Any]:
        seen: set[str] = set()
        duplicates: list[str] = []

        for day in itinerary.get("days") or []:
            for poi in day.get("pois") or []:
                key = normalize_text(poi_name(poi))
                if not key:
                    continue
                if key in seen:
                    duplicates.append(poi_name(poi))
                seen.add(key)

        if duplicates:
            issues.append(CriticIssue(
                code="DUPLICATE_POIS",
                severity="low",
                message=f"Duplicate POIs found: {', '.join(duplicates)}.",
            ))

        return {"score": 0.85 if duplicates else 1.0}

    def _evaluate_foreigner_readiness(
        self,
        itinerary: dict[str, Any],
        issues: list[CriticIssue],
    ) -> dict[str, Any]:
        total = 0
        risky = 0

        for day in itinerary.get("days") or []:
            for poi in day.get("pois") or []:
                total += 1
                english = safe_int(poi.get("english_support"), 1)
                cash_only = safe_int(poi.get("cash_only"), 0)
                friction = safe_int(poi.get("cultural_friction"), 0)
                confidence = normalize_text(poi.get("label_confidence"))

                if english == 0 or cash_only == 1 or friction == 1:
                    risky += 1

                if confidence == "low":
                    risky += 0.25

        if total == 0:
            return {"score": 0.7}

        ratio = risky / total
        score = max(0.0, 1.0 - ratio * 0.5)

        if ratio >= 0.4:
            issues.append(CriticIssue(
                code="HIGH_FOREIGNER_FRICTION",
                severity="medium",
                message="Many POIs may have language, payment, or cultural friction for foreign visitors.",
            ))

        return {"score": round(score, 3)}


# ---------------------------------------------------------------------------
# Repair Agent
# ---------------------------------------------------------------------------

class RepairAgent:
    def repair(self, state: dict[str, Any], report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        itinerary = state.get("itinerary") or {}
        pool = build_candidate_pool(state)
        logs: list[str] = []

        if not itinerary.get("days"):
            return itinerary, logs

        requested_areas = report.get("requested_areas") or extract_requested_areas(
            state.get("location"),
            state.get("purpose"),
        )

        itinerary = self._repair_missing_areas(
            itinerary=itinerary,
            pool=pool,
            requested_areas=requested_areas,
            logs=logs,
        )

        itinerary = self._repair_missing_meals(
            itinerary=itinerary,
            pool=pool,
            requested_areas=requested_areas,
            logs=logs,
        )

        itinerary = self._repair_underfilled_days(
            itinerary=itinerary,
            pool=pool,
            requested_areas=requested_areas,
            logs=logs,
        )

        itinerary = self._remove_duplicates(
            itinerary=itinerary,
            logs=logs,
        )

        itinerary["repair_log"] = logs
        return itinerary, logs

    def _repair_missing_areas(
        self,
        *,
        itinerary: dict[str, Any],
        pool: dict[str, dict[str, Any]],
        requested_areas: list[str],
        logs: list[str],
    ) -> dict[str, Any]:
        if not requested_areas:
            return itinerary

        days = itinerary.get("days") or []
        if not days:
            return itinerary

        coverage = self._coverage(itinerary, requested_areas)
        used = used_name_set(itinerary)

        for idx, area in enumerate(requested_areas):
            current = coverage.get(area, 0)
            if current >= 2:
                continue

            target_day_idx = min(idx, len(days) - 1)
            target_day = days[target_day_idx]
            needed = 2 - current

            candidates = candidates_for_area(
                pool,
                area,
                exclude=used,
            )

            inserted = 0
            for item in candidates:
                if inserted >= needed:
                    break

                poi = as_output_poi(
                    item,
                    note_suffix=f"Added by Repair Agent to cover requested area: {area_label(area)}."
                )
                target_day.setdefault("pois", []).append(poi)
                used.add(normalize_text(poi.get("name")))
                inserted += 1

            if inserted:
                logs.append(f"Added {inserted} POI(s) for requested area {area_label(area)}.")

        return itinerary

    def _repair_missing_meals(
        self,
        *,
        itinerary: dict[str, Any],
        pool: dict[str, dict[str, Any]],
        requested_areas: list[str],
        logs: list[str],
    ) -> dict[str, Any]:
        used = used_name_set(itinerary)

        for idx, day in enumerate(itinerary.get("days") or []):
            pois = day.setdefault("pois", [])
            if any(is_meal_poi(p) for p in pois):
                continue

            target_area = self._target_area_for_day(day, requested_areas, idx)
            candidates = []
            if target_area:
                candidates = candidates_for_area(
                    pool,
                    target_area,
                    exclude=used,
                    preferred_types={"restaurant", "cafe"},
                )

            if not candidates:
                candidates = [
                    item for item in pool.values()
                    if normalize_text(item.get("name")) not in used
                    and normalize_text(item.get("type")) in {"restaurant", "cafe"}
                ]

            if candidates:
                item = candidates[0]
                poi = as_output_poi(
                    item,
                    note_suffix="Added by Repair Agent as a required meal/cafe slot."
                )
                insert_idx = min(2, len(pois))
                pois.insert(insert_idx, poi)
                used.add(normalize_text(poi.get("name")))
                logs.append(f"Added meal slot on Day {day.get('day')}: {poi.get('name')}.")

        return itinerary

    def _repair_underfilled_days(
        self,
        *,
        itinerary: dict[str, Any],
        pool: dict[str, dict[str, Any]],
        requested_areas: list[str],
        logs: list[str],
    ) -> dict[str, Any]:
        used = used_name_set(itinerary)

        for idx, day in enumerate(itinerary.get("days") or []):
            pois = day.setdefault("pois", [])
            if len(pois) >= 5:
                continue

            target_area = self._target_area_for_day(day, requested_areas, idx)
            candidates = []

            if target_area:
                candidates = candidates_for_area(
                    pool,
                    target_area,
                    exclude=used,
                )

            if not candidates:
                candidates = [
                    item for item in pool.values()
                    if normalize_text(item.get("name")) not in used
                ]

            added = 0
            while len(pois) < 5 and candidates:
                item = candidates.pop(0)
                poi = as_output_poi(
                    item,
                    note_suffix="Added by Repair Agent to make the day sufficiently complete."
                )
                pois.append(poi)
                used.add(normalize_text(poi.get("name")))
                added += 1

            if added:
                logs.append(f"Added {added} POI(s) to Day {day.get('day')} because the day was under-filled.")

        return itinerary

    def _remove_duplicates(
        self,
        *,
        itinerary: dict[str, Any],
        logs: list[str],
    ) -> dict[str, Any]:
        seen: set[str] = set()

        for day in itinerary.get("days") or []:
            cleaned: list[dict[str, Any]] = []

            for poi in day.get("pois") or []:
                key = normalize_text(poi_name(poi))
                if not key:
                    continue

                if key in seen:
                    logs.append(f"Removed duplicate POI: {poi_name(poi)}.")
                    continue

                seen.add(key)
                cleaned.append(poi)

            day["pois"] = cleaned

        return itinerary

    def _coverage(self, itinerary: dict[str, Any], requested_areas: list[str]) -> dict[str, int]:
        coverage = {area: 0 for area in requested_areas}

        for day in itinerary.get("days") or []:
            for poi in day.get("pois") or []:
                area = infer_area_from_poi(poi)
                for req in requested_areas:
                    if area_matches_requested(area, req):
                        coverage[req] += 1

        return coverage

    def _target_area_for_day(
        self,
        day: dict[str, Any],
        requested_areas: list[str],
        day_index: int,
    ) -> str | None:
        if requested_areas:
            return requested_areas[min(day_index, len(requested_areas) - 1)]

        counts: dict[str, int] = {}
        for poi in day.get("pois") or []:
            area = infer_area_from_poi(poi)
            if area:
                counts[area] = counts.get(area, 0) + 1

        if not counts:
            return None

        return max(counts.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Public graph node
# ---------------------------------------------------------------------------

def make_critic_repair_node(base_dir: Any | None = None):
    critic = CriticAgent()
    repairer = RepairAgent()

    def critic_repair_node(state: TravelState) -> TravelState:
        itinerary = state.get("itinerary")

        if not itinerary:
            return {
                **state,
                "messages": list(state.get("messages") or []) + [
                    AIMessage(content="⚠️ Critic-Repair skipped because no itinerary was found.")
                ],
            }

        try:
            before_report = critic.evaluate(state)

            repaired_itinerary, repair_logs = repairer.repair(
                state={**state, "itinerary": itinerary},
                report=before_report,
            )

            after_state = {**state, "itinerary": repaired_itinerary}
            after_report = critic.evaluate(after_state)

            repaired_itinerary["critic_report"] = {
                "before": before_report,
                "after": after_report,
                "repair_applied": bool(repair_logs),
                "repair_log": repair_logs,
            }

            requested = after_report.get("requested_areas") or []
            coverage = after_report.get("area_coverage") or {}
            coverage_text = ", ".join(
                f"{area_label(area)}={coverage.get(area, 0)}"
                for area in requested
            ) if requested else "No specific requested areas"

            score = after_report.get("overall_score")

            msg = (
                f"✅ Critic-Repair completed.\n"
                f"- Overall score: {score}\n"
                f"- Requested area coverage: {coverage_text}\n"
                f"- Repairs applied: {len(repair_logs)}"
            )

            return {
                **state,
                "itinerary": repaired_itinerary,
                "critic_report": repaired_itinerary["critic_report"],
                "messages": list(state.get("messages") or []) + [AIMessage(content=msg)],
            }

        except Exception as e:
            return {
                **state,
                "messages": list(state.get("messages") or []) + [
                    AIMessage(content=f"⚠️ Critic-Repair 오류: {e}")
                ],
            }

    return critic_repair_node
