"""
repair.py

SeoulMate / AI 여행 플래너용 Repair 모듈.

역할
----
critic.py가 감지한 작은 이슈들만 수정한다.
일정을 구조적으로 바꾸지 않는다 — 그건 replanner.py의 역할이다.

처리하는 이슈
-------------
- duration_out_of_range  : 체류시간을 권장 범위로 조정
- lunch_missing          : lunch 시간대에 식사 가능 POI 삽입
- dinner_missing         : dinner 시간대에 식사 가능 POI 삽입
- oh_conflict            : 운영시간 충돌 POI의 방문 시간 조정
- missing_foreigner_tip  : 외국인용 notes 보강

처리하지 않는 이슈 (replanner 담당)
-------------------------------------
- vague_poi, off_theme_cluster_poi, cluster_scattered
- too_sparse_day, no_representative_anchor, bad_general_poi
- not_in_profile, consecutive_meal_like_pois

실행 예시
---------
python repair.py --input output/replanned_itinerary.json --critic output/critic_result.json --output output/repaired_itinerary.json
python repair.py  # 기본값으로 실행
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# ============================================================
# 정책값
# ============================================================

MEAL_SLOTS = {
    "lunch": (11 * 60, 14 * 60),
    "dinner": (17 * 60, 20 * 60),
}

TRAVEL_BUFFER_MINUTES = 25
DEFAULT_START_TIME = "10:00"

DURATION_RANGE = {
    "museum": (60, 120),
    "history": (60, 180),
    "palace": (90, 180),
    "park": (45, 120),
    "nature": (45, 120),
    "kpop_landmark": (30, 90),
    "shopping": (60, 150),
    "street": (45, 120),
    "tourist_spot": (45, 120),
    "culture": (60, 120),
    "market": (45, 90),
    "restaurant": (45, 90),
    "food": (45, 90),
    "cafe": (45, 90),
    "nightlife": (60, 150),
    "beauty": (60, 120),
}

# 외국인 여행자용 기본 notes 템플릿
FOREIGNER_TIPS: dict[str, str] = {
    "history":      "Traditional Korean historical site. Dress modestly and follow posted rules.",
    "palace":       "Historic royal palace. Audio guides available in English. Arrive early to avoid crowds.",
    "temple":       "Active Buddhist temple. Dress modestly, remove shoes where indicated, speak quietly.",
    "museum":       "English signage available at most major museums. Check last entry times.",
    "market":       "Traditional market — great for street food and local goods. Cash recommended.",
    "street":       "Lively street area. Best explored on foot. Many shops open late.",
    "park":         "Public park, free entry. Good spot for a rest between sightseeing.",
    "nature":       "Natural area. Comfortable walking shoes recommended.",
    "shopping":     "Shopping area. Most stores open 10:00–22:00. Credit cards widely accepted.",
    "kpop_landmark":"K-pop culture spot. Popular for fans and photo opportunities.",
    "culture":      "Cultural venue. Check current exhibitions before visiting.",
    "cafe":         "Café area. Great place to rest and experience local coffee culture.",
    "restaurant":   "Local dining spot. English menus often available at tourist-friendly areas.",
    "food":         "Local food spot. Try the specialties — pointing at menu photos works fine.",
    "tourist_spot": "Popular tourist destination. Visit early morning or late afternoon for fewer crowds.",
    "beauty":       "Beauty and cosmetics shopping area. Many international brands and Korean brands available.",
}

SAFE_REPAIR_TYPES = {
    "duration_out_of_range",
    "lunch_missing",
    "dinner_missing",
    "oh_conflict",
    "oh_missing",
    "missing_foreigner_tip",
}

AREA_ALIASES = {
    "hongdae_area": {
        "hongdae", "hongik", "yeonnam", "mangwon", "mangnidan", "mapo", "gyeongui",
        "홍대", "홍익대", "연남", "망원", "망리단", "마포", "경의선",
    },
    "gangnam_area": {
        "gangnam", "coex", "samseong", "starfield", "bongeunsa", "garosu", "apgujeong",
        "sinsa", "dosan", "seocho", "seolleung",
        "강남", "코엑스", "삼성", "스타필드", "봉은사", "가로수", "압구정", "신사", "도산", "서초", "선릉",
    },
    "seongsu": {"seongsu", "seoul forest", "성수", "성수동", "서울숲"},
    "jongno_area": {
        "jongno", "insadong", "bukchon", "gyeongbokgung", "gwanghwamun", "samcheong",
        "ikseon", "daehangno", "종로", "인사동", "북촌", "경복궁", "광화문", "삼청", "익선", "대학로",
    },
    "yongsan_itaewon_area": {
        "yongsan", "itaewon", "hannam", "hybe", "leeum", "용산", "이태원", "한남",
    },
    "myeongdong_euljiro_area": {
        "myeongdong", "euljiro", "namdaemun", "dongdaemun", "ddp",
        "명동", "을지로", "남대문", "동대문",
    },
    "yeouido": {"yeouido", "ifc", "hyundai seoul", "hangang", "여의도", "더현대", "한강"},
    "jamsil": {"jamsil", "lotte world", "songnidan", "잠실", "롯데월드", "송리단"},
}

# 권역별 기본 식사 후보 (profile에 없을 때 fallback용)
AREA_MEAL_FALLBACKS: dict[str, list[dict[str, Any]]] = {
    "hongdae_area": [
        {"name": "Mangwon Market (망원시장)", "type": "market", "lat": 37.5549, "lng": 126.9014, "stay_minutes": 60,
         "notes": "Traditional market in Mangwon. Great for street food and local snacks. Cash recommended."},
        {"name": "Yeonnam-dong Cafe Street (연남동 카페거리)", "type": "cafe", "lat": 37.5615, "lng": 126.9228, "stay_minutes": 60,
         "notes": "Trendy cafe street in Yeonnam-dong. Perfect for brunch or coffee break."},
    ],
    "gangnam_area": [
        {"name": "Gangnam Food Alley (강남 먹자골목)", "type": "restaurant", "lat": 37.4979, "lng": 127.0276, "stay_minutes": 60,
         "notes": "Variety of Korean restaurants near Gangnam station. English menus available."},
        {"name": "Coex Food Court (코엑스 푸드코트)", "type": "restaurant", "lat": 37.5118, "lng": 127.0592, "stay_minutes": 60,
         "notes": "Food court inside COEX mall with diverse options. Convenient and budget-friendly."},
    ],
    "jongno_area": [
        {"name": "Insadong Ssamziegil Food (인사동 쌈지길 식당)", "type": "restaurant", "lat": 37.5741, "lng": 126.9861, "stay_minutes": 60,
         "notes": "Traditional Korean food options in Insadong cultural area."},
    ],
    "seongsu": [
        {"name": "Seongsu Cafe (성수동 카페)", "type": "cafe", "lat": 37.5447, "lng": 127.0558, "stay_minutes": 60,
         "notes": "Trendy cafe in Seongsu. Known for unique interiors and specialty coffee."},
    ],
    "yongsan_itaewon_area": [
        {"name": "Itaewon Restaurant Street (이태원 음식거리)", "type": "restaurant", "lat": 37.5347, "lng": 126.9946, "stay_minutes": 60,
         "notes": "International cuisine street in Itaewon. Wide variety of options for all tastes."},
    ],
    "myeongdong_euljiro_area": [
        {"name": "Myeongdong Street Food (명동 길거리 음식)", "type": "market", "lat": 37.5636, "lng": 126.9857, "stay_minutes": 60,
         "notes": "Famous street food area in Myeongdong. Try Korean snacks like tteokbokki and hotteok."},
    ],
    "yeouido": [
        {"name": "Yeouido IFC Mall Food Court (여의도 IFC 푸드코트)", "type": "restaurant", "lat": 37.5217, "lng": 126.9244, "stay_minutes": 60,
         "notes": "Food court in IFC Mall with Korean and international options."},
    ],
    "jamsil": [
        {"name": "Lotte World Food Street (롯데월드 식당가)", "type": "restaurant", "lat": 37.5133, "lng": 127.1028, "stay_minutes": 60,
         "notes": "Variety of restaurants near Lotte World. Convenient for families."},
    ],
}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class RepairAction:
    action_type: str
    day: int | None
    poi_index: int | None
    issue_type: str
    status: str          # applied / skipped / failed
    description: str
    before: Any = None
    after: Any = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class RepairResult:
    changed: bool
    actions: list[RepairAction]
    warnings: list[str]
    skipped_issues: list[str]


# ============================================================
# 유틸
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


def time_to_minutes(t: str) -> int:
    try:
        h, m = t.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 10 * 60


def minutes_to_time(m: int) -> str:
    m = max(0, min(23 * 60 + 59, int(m)))
    return f"{m // 60:02d}:{m % 60:02d}"


def interval_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def poi_name(poi: dict[str, Any]) -> str:
    return clean_str(poi.get("name") or poi.get("poi_name") or poi.get("title"))


def poi_type_str(poi: dict[str, Any]) -> str:
    return clean_str(poi.get("type") or poi.get("poi_type"), "tourist_spot")


def get_day_number(day: dict[str, Any], idx: int) -> int:
    return safe_int(day.get("day"), idx + 1) or idx + 1


def issue_get(issue: Any, key: str, default: Any = None) -> Any:
    if isinstance(issue, dict):
        return issue.get(key, default)
    return getattr(issue, key, default)


def infer_area_from_text(text: str) -> str | None:
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
    return None


def infer_day_area(day: dict[str, Any]) -> str | None:
    texts = [
        clean_str(day.get("theme")),
        clean_str(day.get("title")),
        clean_str(day.get("area")),
        clean_str(day.get("replan_target_area")),
    ]
    for text in texts:
        area = infer_area_from_text(text)
        if area:
            return area
    joined = " ".join(poi_name(p) for p in day.get("pois", []))
    return infer_area_from_text(joined)


def get_poi_schedule(poi: dict[str, Any]) -> tuple[int | None, int | None]:
    """POI의 시작/종료 시간(분)을 반환."""
    start = poi.get("estimated_start_time") or poi.get("start_time")
    end = poi.get("estimated_end_time") or poi.get("end_time")
    s = time_to_minutes(start) if start else None
    e = time_to_minutes(end) if end else None
    return s, e


def is_meal_like_poi(poi: dict[str, Any]) -> bool:
    t = normalize_name(poi_type_str(poi))
    name = normalize_name(poi_name(poi))
    if t in {"restaurant", "food", "cafe", "market"}:
        return True
    if any(k in name for k in ["restaurant", "cafe", "food", "식당", "카페", "맛집",
                                "칼국수", "냉면", "라멘", "kalguksu"]):
        return True
    if "market" in name or "시장" in name:
        return True
    return False


def rebuild_schedule(pois: list[dict[str, Any]], start_time: str = DEFAULT_START_TIME) -> list[dict[str, Any]]:
    """POI 리스트의 시간표를 처음부터 재계산."""
    current = time_to_minutes(start_time)
    out = []
    for p in pois:
        q = copy.deepcopy(p)
        stay = safe_int(q.get("stay_minutes"), 60) or 60

        # 두 번째 POI가 meal-like면 lunch 시간대로 맞춤
        if len(out) == 1 and is_meal_like_poi(q):
            current = max(current, time_to_minutes("11:30"))

        q["estimated_start_time"] = minutes_to_time(current)
        q["estimated_end_time"] = minutes_to_time(current + stay)
        current = current + stay + TRAVEL_BUFFER_MINUTES
        out.append(q)
    return out


# ============================================================
# Repair functions
# ============================================================

def repair_duration(
    poi: dict[str, Any],
    day_num: int,
    poi_idx: int,
) -> RepairAction:
    """체류시간을 권장 범위로 조정."""
    t = normalize_name(poi_type_str(poi)).replace(" ", "_")
    stay = safe_int(poi.get("stay_minutes"), 60) or 60
    lo, hi = DURATION_RANGE.get(t, (30, 180))

    if stay < lo:
        new_stay = lo
    elif stay > hi:
        new_stay = hi
    else:
        return RepairAction(
            action_type="repair_duration",
            day=day_num, poi_index=poi_idx,
            issue_type="duration_out_of_range",
            status="skipped",
            description=f"{poi_name(poi)} 체류시간이 이미 범위 내입니다.",
            before=stay, after=stay,
        )

    return RepairAction(
        action_type="repair_duration",
        day=day_num, poi_index=poi_idx,
        issue_type="duration_out_of_range",
        status="applied",
        description=f"{poi_name(poi)} 체류시간 {stay}분 → {new_stay}분으로 조정",
        before=stay, after=new_stay,
        evidence=[f"type={t}", f"range={lo}-{hi}"],
    )


def repair_missing_foreigner_tip(
    poi: dict[str, Any],
    day_num: int,
    poi_idx: int,
) -> RepairAction:
    """외국인용 notes가 부족한 POI에 기본 tip을 추가."""
    t = normalize_name(poi_type_str(poi))
    name = poi_name(poi)
    current_notes = clean_str(poi.get("notes"))

    tip = FOREIGNER_TIPS.get(t, "")
    if not tip:
        # 이름 기반으로 타입 추정
        n = normalize_name(name)
        if any(k in n for k in ["market", "시장"]):
            tip = FOREIGNER_TIPS["market"]
        elif any(k in n for k in ["temple", "절", "사원", "봉은사"]):
            tip = FOREIGNER_TIPS["temple"]
        elif any(k in n for k in ["palace", "궁"]):
            tip = FOREIGNER_TIPS["palace"]
        elif any(k in n for k in ["park", "공원"]):
            tip = FOREIGNER_TIPS["park"]
        elif any(k in n for k in ["library", "도서관"]):
            tip = FOREIGNER_TIPS["culture"]
        elif any(k in n for k in ["mall", "몰", "coex", "코엑스", "starfield"]):
            tip = FOREIGNER_TIPS["shopping"]
        else:
            tip = FOREIGNER_TIPS["tourist_spot"]

    if current_notes and len(current_notes) >= 20:
        # 이미 충분한 notes가 있으면 tip을 append
        new_notes = f"{current_notes} | {tip}"
    else:
        new_notes = tip

    return RepairAction(
        action_type="repair_foreigner_tip",
        day=day_num, poi_index=poi_idx,
        issue_type="missing_foreigner_tip",
        status="applied",
        description=f"{name}에 외국인용 tip 추가",
        before=current_notes,
        after=new_notes,
        evidence=[f"type={t}"],
    )


def repair_insert_meal(
    day: dict[str, Any],
    day_num: int,
    meal_name: str,
    slot_start: int,
    slot_end: int,
    target_area: str | None,
) -> tuple[dict[str, Any] | None, RepairAction]:
    """
    식사 슬롯이 비어있을 때 fallback meal POI를 삽입.
    삽입 위치: 식사 시간대(slot_start ~ slot_end) 안에 들어갈 수 있는 가장 적합한 자리.
    """
    fallbacks = AREA_MEAL_FALLBACKS.get(target_area or "", [])
    if not fallbacks:
        # 모든 area에 공통으로 쓸 수 있는 generic fallback
        fallbacks = [
            {"name": "Local Restaurant (현지 식당)", "type": "restaurant",
             "stay_minutes": 60,
             "notes": "Local Korean restaurant. English menus often available. Try the daily special."}
        ]

    meal_poi = copy.deepcopy(fallbacks[0])
    meal_poi["source"] = "repair_meal_insert"
    meal_poi["estimated_start_time"] = minutes_to_time(slot_start + 15)
    meal_poi["estimated_end_time"] = minutes_to_time(slot_start + 15 + (meal_poi.get("stay_minutes") or 60))

    action = RepairAction(
        action_type="repair_insert_meal",
        day=day_num, poi_index=None,
        issue_type=f"{meal_name}_missing",
        status="applied",
        description=f"Day {day_num} {meal_name} 슬롯에 {poi_name(meal_poi)} 삽입",
        before=None,
        after=poi_name(meal_poi),
        evidence=[
            f"meal_slot={minutes_to_time(slot_start)}-{minutes_to_time(slot_end)}",
            f"target_area={target_area}",
            f"inserted={poi_name(meal_poi)}",
        ],
    )
    return meal_poi, action


def repair_oh_conflict(
    poi: dict[str, Any],
    day_num: int,
    poi_idx: int,
) -> RepairAction:
    """
    운영시간 충돌 POI의 방문 시간을 일반적인 오전(10:00)으로 조정.
    실제 운영시간 데이터가 없으면 10:00 시작으로 안전하게 이동.
    """
    old_start = poi.get("estimated_start_time", "")
    old_end = poi.get("estimated_end_time", "")
    stay = safe_int(poi.get("stay_minutes"), 60) or 60

    # 안전한 시간대: 10:00 시작
    new_start = "10:00"
    new_end = minutes_to_time(time_to_minutes(new_start) + stay)

    return RepairAction(
        action_type="repair_oh_conflict",
        day=day_num, poi_index=poi_idx,
        issue_type="oh_conflict",
        status="applied",
        description=f"{poi_name(poi)} 방문 시간 {old_start}-{old_end} → {new_start}-{new_end} 조정",
        before=f"{old_start}-{old_end}",
        after=f"{new_start}-{new_end}",
        evidence=["adjusted_to_safe_morning_slot"],
    )


# ============================================================
# Main Repairer
# ============================================================

class ItineraryRepairer:
    def __init__(self, profile_path: str | Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        if profile_path is None:
            profile_path = base_dir / "output" / "area_profiles_v2.json"
        else:
            profile_path = Path(profile_path)
            if not profile_path.is_absolute():
                profile_path = base_dir / profile_path
        self.profile_path = profile_path

    def repair(
        self,
        itinerary: dict[str, Any],
        critic_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        critic_result의 repair 대상 이슈를 처리해서 수정된 itinerary를 반환.
        replanner 대상 이슈(structural)는 무시한다.
        """
        repaired = copy.deepcopy(itinerary)
        actions: list[RepairAction] = []
        warnings: list[str] = []
        skipped: list[str] = []

        issues = self._extract_repair_issues(critic_result)

        if not issues:
            return repaired, self._result_to_dict(RepairResult(
                changed=False, actions=[], warnings=[], skipped_issues=[]
            ))

        # day별로 이슈 그룹핑
        day_issues: dict[int, list[dict[str, Any]]] = {}
        for issue in issues:
            day_num = safe_int(issue_get(issue, "day"), None)
            if day_num is None:
                continue
            day_issues.setdefault(day_num, []).append(issue)

        # 각 day 수정
        changed = False
        for day_idx, day in enumerate(repaired.get("days", [])):
            day_num = get_day_number(day, day_idx)
            if day_num not in day_issues:
                continue

            day_changed, day_actions, day_warnings = self._repair_day(
                day, day_num, day_issues[day_num]
            )
            actions.extend(day_actions)
            warnings.extend(day_warnings)
            if day_changed:
                changed = True

        repaired.setdefault("repair_log", [])
        repaired["repair_log"].extend([asdict(a) for a in actions])
        repaired["repaired"] = changed

        result = RepairResult(
            changed=changed,
            actions=actions,
            warnings=warnings,
            skipped_issues=skipped,
        )
        return repaired, self._result_to_dict(result)

    def _repair_day(
        self,
        day: dict[str, Any],
        day_num: int,
        issues: list[dict[str, Any]],
    ) -> tuple[bool, list[RepairAction], list[str]]:
        actions: list[RepairAction] = []
        warnings: list[str] = []
        changed = False
        pois = day.get("pois", [])
        target_area = infer_day_area(day)

        issue_types = {clean_str(issue_get(i, "issue_type")).lower() for i in issues}

        # --------------------------------------------------
        # 1) duration_out_of_range: 체류시간 조정
        # --------------------------------------------------
        if "duration_out_of_range" in issue_types:
            duration_issues = [
                i for i in issues
                if clean_str(issue_get(i, "issue_type")).lower() == "duration_out_of_range"
            ]
            for issue in duration_issues:
                poi_idx = safe_int(issue_get(issue, "poi_index"), None)
                if poi_idx is None or poi_idx >= len(pois):
                    continue
                poi = pois[poi_idx]
                action = repair_duration(poi, day_num, poi_idx)
                if action.status == "applied":
                    pois[poi_idx]["stay_minutes"] = action.after
                    changed = True
                actions.append(action)

        # --------------------------------------------------
        # 2) missing_foreigner_tip: notes 보강
        # --------------------------------------------------
        if "missing_foreigner_tip" in issue_types:
            tip_issues = [
                i for i in issues
                if clean_str(issue_get(i, "issue_type")).lower() == "missing_foreigner_tip"
            ]
            for issue in tip_issues:
                poi_idx = safe_int(issue_get(issue, "poi_index"), None)
                if poi_idx is None or poi_idx >= len(pois):
                    continue
                poi = pois[poi_idx]
                action = repair_missing_foreigner_tip(poi, day_num, poi_idx)
                if action.status == "applied":
                    pois[poi_idx]["notes"] = action.after
                    changed = True
                actions.append(action)

        # --------------------------------------------------
        # 3) oh_conflict: 운영시간 충돌 조정
        # --------------------------------------------------
        if "oh_conflict" in issue_types:
            oh_issues = [
                i for i in issues
                if clean_str(issue_get(i, "issue_type")).lower() == "oh_conflict"
            ]
            for issue in oh_issues:
                poi_idx = safe_int(issue_get(issue, "poi_index"), None)
                if poi_idx is None or poi_idx >= len(pois):
                    continue
                poi = pois[poi_idx]
                action = repair_oh_conflict(poi, day_num, poi_idx)
                if action.status == "applied":
                    pois[poi_idx]["estimated_start_time"] = action.after.split("-")[0]
                    pois[poi_idx]["estimated_end_time"] = action.after.split("-")[1]
                    changed = True
                actions.append(action)

        # --------------------------------------------------
        # 4) lunch_missing / dinner_missing: 식사 POI 삽입
        # --------------------------------------------------
        for meal_name, (slot_start, slot_end) in MEAL_SLOTS.items():
            if f"{meal_name}_missing" not in issue_types:
                continue

            # 이미 해당 시간대에 식사 가능한 POI가 있는지 재확인
            already_covered = False
            for poi in pois:
                s, e = get_poi_schedule(poi)
                if s is not None and e is not None:
                    if interval_overlaps(s, e, slot_start, slot_end) and is_meal_like_poi(poi):
                        already_covered = True
                        break
                elif is_meal_like_poi(poi):
                    # 시간 정보 없으면 meal-like POI 있으면 covered로 간주
                    already_covered = True
                    break

            if already_covered:
                actions.append(RepairAction(
                    action_type="repair_insert_meal",
                    day=day_num, poi_index=None,
                    issue_type=f"{meal_name}_missing",
                    status="skipped",
                    description=f"Day {day_num} {meal_name} — 이미 식사 가능 POI 있음",
                    evidence=["already_covered"],
                ))
                continue

            meal_poi, action = repair_insert_meal(
                day, day_num, meal_name, slot_start, slot_end, target_area
            )
            if meal_poi:
                # lunch는 두 번째 자리, dinner는 마지막 자리에 삽입
                if meal_name == "lunch":
                    insert_pos = min(1, len(pois))
                else:
                    insert_pos = len(pois)
                pois.insert(insert_pos, meal_poi)
                changed = True
            actions.append(action)

        # --------------------------------------------------
        # 5) 변경사항이 있으면 시간표 재계산
        # --------------------------------------------------
        if changed:
            day["pois"] = rebuild_schedule(pois)
            day["repaired"] = True

        return changed, actions, warnings

    def _extract_repair_issues(
        self, critic_result: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not critic_result:
            return []

        all_issues = []
        for key in ["issues", "unresolved_warnings"]:
            val = critic_result.get(key)
            if isinstance(val, list):
                all_issues.extend(val)

        # repair 대상만 필터 (replanner 대상 제외)
        repair_issues = []
        seen = set()
        for issue in all_issues:
            issue_type = clean_str(issue_get(issue, "issue_type")).lower()
            target = clean_str(issue_get(issue, "target_module")).lower()
            day = safe_int(issue_get(issue, "day"), None)
            poi_idx = safe_int(issue_get(issue, "poi_index"), None)

            if issue_type not in SAFE_REPAIR_TYPES:
                continue
            if target not in {"repair", "critic", ""}:
                continue

            # 중복 제거
            key = (issue_type, day, poi_idx)
            if key in seen:
                continue
            seen.add(key)
            repair_issues.append(issue)

        return repair_issues

    def _result_to_dict(self, result: RepairResult) -> dict[str, Any]:
        return {
            "changed": result.changed,
            "actions": [asdict(a) for a in result.actions],
            "warnings": result.warnings,
            "skipped_issues": result.skipped_issues,
        }


# ============================================================
# Public function
# ============================================================

def repair_itinerary(
    itinerary: dict[str, Any],
    critic_result: dict[str, Any] | None = None,
    profile_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repairer = ItineraryRepairer(profile_path=profile_path)
    return repairer.repair(itinerary=itinerary, critic_result=critic_result)


# ============================================================
# CLI
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply small repairs to itinerary based on critic result."
    )
    parser.add_argument("--input", type=str, default="",
                        help="Input itinerary JSON. Default: output/replanned_itinerary.json")
    parser.add_argument("--critic", type=str, default="",
                        help="Critic result JSON. Default: output/critic_result.json")
    parser.add_argument("--output", type=str, default="output/repaired_itinerary.json")
    parser.add_argument("--profile", type=str, default="output/area_profiles_v2.json")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # input itinerary
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = base_dir / input_path
    else:
        input_path = base_dir / "output" / "replanned_itinerary.json"
        if not input_path.exists():
            # v3_1 fallback
            input_path = base_dir / "output" / "replanned_itinerary_v3_1.json"

    # critic result
    if args.critic:
        critic_path = Path(args.critic)
        if not critic_path.is_absolute():
            critic_path = base_dir / critic_path
    else:
        critic_path = base_dir / "output" / "critic_result.json"

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = base_dir / profile_path

    itinerary = load_json(input_path)
    critic_result = load_json(critic_path) if critic_path.exists() else None

    repaired, result = repair_itinerary(
        itinerary=itinerary,
        critic_result=critic_result,
        profile_path=profile_path,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[repair] 완료")
    print(f"  changed  : {result['changed']}")
    print(f"  output   : {output_path}")

    print("\n[actions]")
    for a in result["actions"]:
        status_mark = "✓" if a["status"] == "applied" else "–"
        print(f"  {status_mark} Day {a['day']} [{a['issue_type']}] {a['description']}")

    if result["warnings"]:
        print("\n[warnings]")
        for w in result["warnings"]:
            print(f"  - {w}")

    print("\n[repaired itinerary]")
    for day in repaired.get("days", []):
        print(f"\nDay {day.get('day')} — {day.get('theme')}")
        for p in day.get("pois", []):
            print(
                f"  - {p.get('estimated_start_time', '')}-{p.get('estimated_end_time', '')} "
                f"{p.get('name')} ({p.get('type')})"
            )


if __name__ == "__main__":
    main()