"""
critic.py

SeoulMate / AI 여행 플래너용 독립 Critic 모듈.

목표
----
- 기존 critic_repair.py에서 Critic 역할만 분리한다.
- itinerary를 절대 수정하지 않는다.
- 문제를 issue로만 감지하고, 그 문제가 repair.py 대상인지 replanner.py 대상인지 표시한다.
- area_profiles_v2.json을 사용해 서울 전체 POI/권역 기준으로 검수한다.

역할 분리
---------
1. critic.py
   - itinerary 문제 감지
   - off_theme_cluster_poi, vague_poi, too_sparse_day, meal_missing, oh_conflict 등 출력

2. repair.py
   - 작은 수정만 수행
   - notes 보강, 체류시간 조정, 식사 시간대 삽입, 운영시간 충돌 시간 조정 등

3. replanner.py
   - 구조적으로 잘못된 day를 area_profiles_v2.json 기반으로 재구성

실행 예시
---------
cd "C:\\Users\\leechaewon\\Desktop\\홍익대 자료\\시스템분석\\project\\ai_agent_travel"

python critic.py
python critic.py --input output/replanned_itinerary.json
python critic.py --input output/replanned_itinerary.json --output output/critic_result.json
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
# 기본 정책
# ============================================================

MEAL_SLOTS = {
    "lunch": (11 * 60, 14 * 60),
    "dinner": (17 * 60, 20 * 60),
}

DEFAULT_DAY_START_MINUTES = 10 * 60
DEFAULT_TRAVEL_BUFFER_MINUTES = 25
DEFAULT_STAY_MINUTES = 60

# 하루 일정 밀도.
# half-day가 명시되지 않은 일반 여행 day는 최소 4개 POI를 권장한다.
MIN_POIS_FULL_DAY = 4
MIN_POIS_HALF_DAY = 3
MAX_POIS_PER_DAY = 7

# 외국인 여행에서 너무 긴 단일 이동은 warning.
MAX_SIMPLE_DISTANCE_KM = 3.0
MAX_AVG_DISTANCE_KM = 4.0

# 권장 체류시간 범위
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

# 구조적 문제: repair.py가 아니라 replanner.py가 처리해야 하는 issue
STRUCTURAL_REPLAN_TYPES = {
    "vague_poi",
    "area_anchor_needs_concrete_poi",
    "off_theme_cluster_poi",
    "cluster_scattered",
    "too_sparse_day",
    "weak_representative_anchor",
    "no_representative_anchor",
    "not_in_profile",
    "bad_general_poi",
    "consecutive_meal_like_pois",
}

# 작은 수정 문제: repair.py가 처리하는 issue
SAFE_REPAIR_TYPES = {
    "duration_out_of_range",
    "lunch_missing",
    "dinner_missing",
    "oh_conflict",
    "oh_missing",
    "cultural_friction_unexplained",
    "missing_foreigner_tip",
}

# final passed를 막는 핵심 문제
# not_in_profile은 BLOCKING에서 제외:
#   Google Places 식당/카페는 profile에 없어도 정상이므로
#   not_in_profile이 passed를 막으면 안 됨
BLOCKING_TYPES = {
    "vague_poi",
    "area_anchor_needs_concrete_poi",
    "off_theme_cluster_poi",
    "too_sparse_day",
    "no_representative_anchor",
    "oh_conflict",
    "lunch_missing",
    "dinner_missing",
}

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

BAD_GENERAL_PATTERNS = {
    "karaoke", "노래방",
    "guest house", "guesthouse", "게스트하우스",
    "station", "역",
    "university", "univ", "대학교", "대학",
    "airport", "공항",
}

ACTIVITY_SPECIAL_PATTERNS = {
    "water sports", "수상스포츠", "class", "one-day class", "원데이", "체험",
    "photo booth", "self-photo", "셀프사진", "포토부스",
}

MEAL_LIKE_ROLES = {"meal", "cafe", "market"}
ATTRACTION_LIKE_ROLES = {"attraction", "shopping", "history", "culture", "nature", "kpop", "family", "market", "beauty"}

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
# Dataclasses
# ============================================================

@dataclass
class IssueItem:
    issue_type: str
    day: int | None
    poi_index: int | None
    poi_name: str
    category: str
    severity: str
    status: str
    description: str
    evidence: list[str] = field(default_factory=list)
    target_module: str = "critic"  # repair / replanner / data / critic
    repairable: bool = False
    suggested_action: str = ""


@dataclass
class MetricResult:
    score: float
    status: str
    confidence: str
    evidence: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CriticResult:
    overall_score: float
    summary_status: str
    passed: bool
    needs_repair: bool
    needs_replan: bool
    score_breakdown: dict[str, dict[str, Any]]
    category_scores: dict[str, float]
    issues: list[dict[str, Any]]
    unresolved_warnings: list[dict[str, Any]]
    days_needing_repair: list[int]
    days_needing_replan: list[int]
    source_summary: dict[str, Any]
    debug: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Utilities
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


def safe_int(value: Any, default: int | None = None) -> int | None:
    if is_missing(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def parse_time_to_minutes(value: Any) -> int | None:
    if is_missing(value):
        return None
    s = str(value).strip().lower()

    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*(am|pm)?", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour * 60 + minute

    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", s)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23:
            return hour * 60

    return None


def minutes_to_hhmm(minutes: int | float | None) -> str:
    if minutes is None:
        return ""
    minutes = int(round(float(minutes))) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def interval_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_travel_minutes(a: dict[str, Any], b: dict[str, Any]) -> int:
    lat1 = safe_float(a.get("lat"))
    lng1 = safe_float(a.get("lng"))
    lat2 = safe_float(b.get("lat"))
    lng2 = safe_float(b.get("lng"))
    if None in {lat1, lng1, lat2, lng2}:
        return DEFAULT_TRAVEL_BUFFER_MINUTES
    km = haversine_km(lat1, lng1, lat2, lng2)
    return int(round((km / 20.0) * 60 + 10))


def poi_name(poi: dict[str, Any]) -> str:
    return clean_str(poi.get("name") or poi.get("poi_name") or poi.get("title"))


def poi_type(poi: dict[str, Any]) -> str:
    return clean_str(poi.get("type") or poi.get("poi_type"), "tourist_spot")


def get_day_number(day: dict[str, Any], idx: int) -> int:
    return safe_int(day.get("day"), idx + 1) or idx + 1


def issue_to_dict(issue: IssueItem) -> dict[str, Any]:
    return asdict(issue)


def metric_status(score: float) -> str:
    if score >= 0.85:
        return "pass"
    if score >= 0.60:
        return "warning"
    return "fail"


def severity_weight(severity: str) -> float:
    return {"high": 1.0, "medium": 0.55, "low": 0.25}.get(str(severity).lower(), 0.35)


def dedupe_key(name: str) -> str:
    n = normalize_name(name)
    aliases = [
        ("starfield coex mall", "coex"),
        ("convention and exhibition center coex", "coex"),
        ("coex mall", "coex"),
        ("gyeongui line forest park yeontral park", "gyeongui line forest park"),
        ("yeonnam dong cafe street", "yeonnam cafe street"),
    ]
    for src, dst in aliases:
        if src in n:
            return dst
    return n


def json_safe_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# Opening hours
# ============================================================

def parse_opening_slot_minutes(slot: Any) -> tuple[int, int] | None:
    if not isinstance(slot, (list, tuple)) or len(slot) < 2:
        return None
    start = parse_time_to_minutes(slot[0])
    end = parse_time_to_minutes(slot[1])
    if start is None or end is None:
        return None
    if end <= start:
        end += 24 * 60
    return start, end


def check_open_during_interval(opening_hours: Any, weekday: str, start_min: int | None, end_min: int | None) -> tuple[str, str]:
    if start_min is None or end_min is None:
        return "unknown", "visit time is missing"

    if not isinstance(opening_hours, dict) or not opening_hours:
        return "missing", "opening_hours is missing"

    day_key = weekday or "mon"
    if day_key not in opening_hours:
        return "missing", f"opening_hours for {day_key} is missing"

    slots = opening_hours.get(day_key)
    if slots is None:
        return "closed", f"{day_key} appears closed"

    if not isinstance(slots, list) or not slots:
        return "missing", f"opening_hours for {day_key} is empty"

    parsed = []
    for slot in slots:
        ps = parse_opening_slot_minutes(slot)
        if ps:
            parsed.append(ps)

    if not parsed:
        return "unknown", f"opening_hours exists but cannot be parsed: {slots}"

    visit_start = int(start_min)
    visit_end = int(end_min)
    if visit_end <= visit_start:
        visit_end += 24 * 60

    for open_start, open_end in parsed:
        if open_start <= visit_start and visit_end <= open_end:
            return "pass", f"visit {minutes_to_hhmm(visit_start)}-{minutes_to_hhmm(visit_end)} within {minutes_to_hhmm(open_start)}-{minutes_to_hhmm(open_end)}"

    readable = ", ".join(f"{minutes_to_hhmm(s)}-{minutes_to_hhmm(e)}" for s, e in parsed)
    return "conflict", f"visit {minutes_to_hhmm(visit_start)}-{minutes_to_hhmm(visit_end)} outside opening hours ({readable})"


# ============================================================
# Area profile index
# ============================================================

def infer_area_from_text(text: str, profile: dict[str, Any]) -> str | None:
    n = normalize_name(text)
    if not n:
        return None

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


class AreaProfileIndex:
    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = profile
        self.clusters = profile.get("clusters", {})
        self.name_index: dict[str, dict[str, Any]] = {}
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
                key = normalize_name(name)
                if key:
                    self.name_index.setdefault(key, p)

    def find_poi(self, name: str) -> dict[str, Any] | None:
        n = normalize_name(name)
        if not n:
            return None

        if n in self.name_index:
            return self.name_index[n]

        best = None
        best_score = 0.0
        n_tokens = set(n.split())

        for key, p in self.name_index.items():
            if not key:
                continue

            score = 0.0
            if key in n or n in key:
                score = min(len(key), len(n)) / max(len(key), len(n))
            else:
                k_tokens = set(key.split())
                if n_tokens and k_tokens:
                    overlap = len(n_tokens & k_tokens)
                    union = len(n_tokens | k_tokens)
                    score = overlap / union

            if score > best_score:
                best_score = score
                best = p

        if best_score >= 0.55:
            return best
        return None

    def get_cluster(self, area_key: str | None) -> dict[str, Any]:
        if not area_key:
            return {}
        return self.clusters.get(area_key, {})

    def compatible_clusters(self, area_key: str | None) -> set[str]:
        if not area_key:
            return set()

        cdata = self.get_cluster(area_key)
        compatible = set(cdata.get("compatible_clusters") or [])
        compatible |= set(cdata.get("members") or [])
        compatible.add(area_key)
        return compatible

    def is_profile_poi_compatible(self, profile_poi: dict[str, Any] | None, area_key: str | None) -> bool:
        if not area_key or not profile_poi:
            return True
        cluster = clean_str(profile_poi.get("cluster"))
        if not cluster:
            return False
        return cluster in self.compatible_clusters(area_key) or cluster == area_key

    def target_area_for_day(self, day: dict[str, Any], user_state: dict[str, Any]) -> str | None:
        texts = [
            clean_str(day.get("theme")),
            clean_str(day.get("title")),
            clean_str(day.get("area")),
            clean_str(day.get("location")),
            clean_str(user_state.get("location")),
            clean_str(user_state.get("area")),
        ]

        for text in texts:
            area = infer_area_from_text(text, self.profile)
            if area:
                return area

        joined = " ".join(poi_name(p) for p in day.get("pois", []))
        return infer_area_from_text(joined, self.profile)

    def top_representative_names(self, area_key: str | None, top_k: int = 8) -> list[str]:
        cdata = self.get_cluster(area_key)
        reps = cdata.get("representative_pois", [])[:top_k]
        return [clean_str(p.get("name")) for p in reps if clean_str(p.get("name"))]

    def top_representative_keys(self, area_key: str | None, top_k: int = 8) -> set[str]:
        return {dedupe_key(n) for n in self.top_representative_names(area_key, top_k=top_k)}


# ============================================================
# Itinerary helpers
# ============================================================

def estimate_schedule_for_day(day: dict[str, Any]) -> list[dict[str, Any]]:
    current = DEFAULT_DAY_START_MINUTES
    schedule = []
    prev = None

    for idx, p in enumerate(day.get("pois", [])):
        start = parse_time_to_minutes(
            p.get("estimated_start_time")
            or p.get("estimated_start")
            or p.get("start_time")
            or p.get("start")
            or p.get("time")
        )

        end = parse_time_to_minutes(
            p.get("estimated_end_time")
            or p.get("estimated_end")
            or p.get("end_time")
            or p.get("end")
        )

        if start is None:
            if prev is not None:
                current += estimate_travel_minutes(prev, p)
            start = current

        stay = safe_int(p.get("stay_minutes") or p.get("duration_minutes"), None)
        if stay is None or stay <= 0:
            stay = default_stay_minutes(p)

        if end is None:
            end = start + stay
        elif end > start:
            stay = end - start

        q = copy.deepcopy(p)
        q["_schedule_idx"] = idx
        q["_start_min"] = start
        q["_end_min"] = end
        q["_start"] = minutes_to_hhmm(start)
        q["_end"] = minutes_to_hhmm(end)
        q["_stay_minutes"] = stay
        schedule.append(q)

        current = end
        prev = p

    return schedule


def default_stay_minutes(poi: dict[str, Any]) -> int:
    t = normalize_name(poi_type(poi))
    name = normalize_name(poi_name(poi))

    if "photo" in name or "포토" in name:
        return 40

    defaults = {
        "restaurant": 60,
        "food": 60,
        "cafe": 60,
        "market": 75,
        "shopping": 90,
        "park": 75,
        "nature": 75,
        "history": 90,
        "culture": 90,
        "museum": 90,
        "street": 60,
        "tourist spot": 60,
        "tourist_spot": 60,
    }
    return defaults.get(t, DEFAULT_STAY_MINUTES)


def is_broad_or_vague_poi(poi: dict[str, Any]) -> bool:
    name = normalize_name(poi_name(poi))
    ptype = normalize_name(poi_type(poi))

    if not name:
        return True

    if any(normalize_name(m) in name for m in CONCRETE_MARKERS):
        return False

    if name in BROAD_AREA_EXACT_ALIASES:
        return True

    if ptype in {"area", "district", "neighborhood"}:
        return True

    tokens = name.split()
    if len(tokens) <= 3:
        for alias in BROAD_AREA_EXACT_ALIASES:
            a = normalize_name(alias)
            if a and a in name:
                return True

    return False


def is_meal_like(poi: dict[str, Any], profile_poi: dict[str, Any] | None = None) -> bool:
    roles = set(profile_poi.get("roles") or []) if profile_poi else set()
    t = normalize_name(poi_type(poi))
    name = normalize_name(poi_name(poi))

    if roles & MEAL_LIKE_ROLES:
        return True
    if t in {"restaurant", "food", "cafe", "market"}:
        return True
    if any(k in name for k in ["restaurant", "cafe", "market", "food", "식당", "카페", "시장", "맛집"]):
        return True
    return False


def is_inside_meal_window(poi: dict[str, Any]) -> bool:
    """POI 방문 시간이 lunch/dinner 시간대와 겹치는지 확인."""
    start = poi.get("_start_min")
    end = poi.get("_end_min")
    if start is None or end is None:
        return False
    for slot_start, slot_end in MEAL_SLOTS.values():
        if interval_overlaps(int(start), int(end), slot_start, slot_end):
            return True
    return False


def purpose_allows_profile_poi(profile_poi: dict[str, Any] | None, purpose: str) -> bool:
    """사용자 목적상 허용 가능한 특수 후보인지 판단."""
    if not profile_poi:
        return False
    roles = set(profile_poi.get("roles") or [])
    purpose = normalize_name(purpose)

    if purpose in {"food"} and roles & {"meal", "market", "cafe"}:
        return True
    if purpose in {"cafe_hopping", "cafe"} and roles & {"cafe", "attraction", "shopping"}:
        return True
    if purpose in {"shopping"} and roles & {"shopping", "market", "beauty", "attraction"}:
        return True
    if purpose in {"kpop"} and roles & {"kpop", "shopping", "culture", "attraction"}:
        return True
    if purpose in {"nature"} and roles & {"nature", "park", "family", "attraction"}:
        return True
    if purpose in {"family"} and roles & {"family", "indoor_leisure", "nature", "attraction"}:
        return True
    if purpose in {"nightlife"} and roles & {"nightlife", "meal"}:
        return True
    if purpose in {"beauty"} and roles & {"beauty", "shopping"}:
        return True
    if purpose in {"history", "culture"} and roles & {"history", "culture", "attraction"}:
        return True

    return False


def is_attraction_like(poi: dict[str, Any], profile_poi: dict[str, Any] | None = None) -> bool:
    roles = set(profile_poi.get("roles") or []) if profile_poi else set()
    t = normalize_name(poi_type(poi))

    if roles & ATTRACTION_LIKE_ROLES:
        return True
    if t in {"tourist spot", "tourist_spot", "street", "market", "shopping", "park", "nature", "history", "culture", "museum"}:
        return True
    return False


def is_bad_general_poi(poi: dict[str, Any], profile_poi: dict[str, Any] | None, purpose: str) -> bool:
    """일반 대표 anchor로 부적합한 후보인지 판단.

    v1.1 보정:
    - 식사 시간대에 들어간 restaurant/cafe/market은 bad_general_poi로 보지 않는다.
    - 사용자가 shopping/kpop/food/cafe_hopping 등 명확한 목적을 준 경우 해당 role 후보는 허용한다.
    - 이 함수는 '대표 anchor' 품질 검사용이지, 식사 슬롯 후보를 벌주기 위한 함수가 아니다.
    """
    text = " ".join([poi_name(poi), poi_type(poi), clean_str(poi.get("notes"))])
    roles = set(profile_poi.get("roles") or []) if profile_poi else set()
    purpose = normalize_name(purpose or "general")

    # 실제 식사 시간대에 배치된 식당/카페/시장은 정상 meal slot으로 본다.
    if is_meal_like(poi, profile_poi) and is_inside_meal_window(poi):
        return False

    # purpose-specific 후보는 해당 목적에서 허용한다.
    if purpose_allows_profile_poi(profile_poi, purpose):
        return False

    # General 목적에서 아래 후보는 대표 anchor로 부적합하다.
    if purpose not in {"nightlife"} and contains_any(text, {"karaoke", "노래방"}):
        return True

    if purpose not in {"family"} and contains_any(text, {"bowling", "볼링"}):
        return True

    if purpose not in {"education"} and contains_any(text, {"university", "대학교", "대학"}):
        return True

    if purpose not in {"transport"} and contains_any(text, {"station", "역"}):
        return True

    if purpose == "general" and roles & {"accommodation", "transport", "education"}:
        return True

    # 일반 목적에서는 수상스포츠/원데이클래스 같은 특수체험은 대표 anchor로 약함.
    if purpose == "general" and contains_any(text, ACTIVITY_SPECIAL_PATTERNS):
        return True

    return False

def infer_purpose(user_state: dict[str, Any], day: dict[str, Any] | None = None) -> str:
    texts = []
    for key in ["purpose", "travel_purpose", "theme", "interest", "interests"]:
        val = user_state.get(key)
        if val:
            texts.append(" ".join(val) if isinstance(val, list) else str(val))
    if day:
        texts.append(clean_str(day.get("theme")))

    blob = normalize_name(" ".join(texts))
    mapping = [
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

    for purpose, keywords in mapping:
        if any(normalize_name(k) in blob for k in keywords):
            return purpose
    return "general"


def is_half_day(user_state: dict[str, Any], day: dict[str, Any], schedule: list[dict[str, Any]]) -> bool:
    text = normalize_name(" ".join([
        clean_str(user_state.get("duration")),
        clean_str(user_state.get("travel_duration")),
        clean_str(day.get("theme")),
        clean_str(day.get("title")),
    ]))

    if any(k in text for k in ["half day", "halfday", "반나절", "오전", "오후"]):
        return True

    if schedule:
        start = min(p["_start_min"] for p in schedule)
        end = max(p["_end_min"] for p in schedule)
        if end - start <= 240 and len(schedule) <= 3:
            return True

    return False


# ============================================================
# Critic Agent
# ============================================================

class CriticAgent:
    def __init__(
        self,
        profile_path: str | Path = "output/area_profiles_v2.json",
        user_state: dict[str, Any] | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        path = Path(profile_path)
        if not path.is_absolute():
            path = base_dir / path

        self.profile_path = path
        self.user_state = user_state or {}

        if path.exists():
            self.index = AreaProfileIndex.from_path(path)
            self.profile_loaded = True
        else:
            self.index = AreaProfileIndex({"clusters": {}, "metadata": {}})
            self.profile_loaded = False

    def evaluate(self, itinerary: dict[str, Any]) -> dict[str, Any]:
        issues: list[IssueItem] = []
        debug: dict[str, Any] = {
            "profile_loaded": self.profile_loaded,
            "profile_path": str(self.profile_path),
            "day_targets": {},
        }

        self._check_schema(itinerary, issues)
        self._check_each_day(itinerary, issues, debug)

        score_breakdown = self._build_score_breakdown(itinerary, issues)
        category_scores = self._build_category_scores(score_breakdown)

        overall_score = round(sum(category_scores.values()) / max(len(category_scores), 1), 3)
        days_repair = sorted({
            i.day for i in issues
            if i.day is not None and i.target_module == "repair"
        })
        days_replan = sorted({
            i.day for i in issues
            if i.day is not None and i.target_module == "replanner"
        })

        blocking_issues = [
            i for i in issues
            if self._is_blocking_issue(i)
        ]

        metric_failures = [
            name for name, metric in score_breakdown.items()
            if metric["status"] == "fail"
            and name in {"within_sandbox", "opening_hours", "meal_coverage", "day_density", "cluster_preservation", "representative_anchor"}
        ]

        if blocking_issues or metric_failures:
            summary_status = "fail"
        elif any(i.severity in {"medium", "low"} for i in issues) or overall_score < 0.85:
            summary_status = "warning"
        else:
            summary_status = "pass"

        # passed 판단:
        # - blocking_issues나 metric_failures가 없고
        # - 남은 이슈가 모두 LOW severity이면 통과 허용
        # - score 기준 0.75로 완화 (Google Places POI가 not_in_profile로 걸리던 문제 해결)
        remaining_medium_high = [
            i for i in issues
            if i.severity in {"medium", "high"}
            and i.issue_type not in {"not_in_profile", "consecutive_meal_like_pois",
                                      "weak_representative_anchor", "complex_transfer",
                                      "route_too_spread"}
        ]
        passed = (
            not blocking_issues
            and not metric_failures
            and overall_score >= 0.75
            and not remaining_medium_high
        )

        source_summary = {
            "total_days": len(itinerary.get("days", [])),
            "total_pois": sum(len(d.get("pois", [])) for d in itinerary.get("days", [])),
            "profile_loaded": self.profile_loaded,
            "profile_path": str(self.profile_path),
            "blocking_issue_count": len(blocking_issues),
            "metric_failures": metric_failures,
            "repair_issue_count": len([i for i in issues if i.target_module == "repair"]),
            "replan_issue_count": len([i for i in issues if i.target_module == "replanner"]),
        }

        result = CriticResult(
            overall_score=overall_score,
            summary_status=summary_status,
            passed=passed,
            needs_repair=bool(days_repair),
            needs_replan=bool(days_replan),
            score_breakdown=score_breakdown,
            category_scores=category_scores,
            issues=[issue_to_dict(i) for i in issues],
            unresolved_warnings=[
                issue_to_dict(i) for i in issues
                if i.severity in {"high", "medium"} or i.status in {"fail", "warning"}
            ],
            days_needing_repair=days_repair,
            days_needing_replan=days_replan,
            source_summary=source_summary,
            debug=debug,
        )

        return asdict(result)

    def _add_issue(self, issues: list[IssueItem], issue: IssueItem) -> None:
        issues.append(issue)

    def _is_blocking_issue(self, issue: IssueItem) -> bool:
        if issue.severity == "high" or issue.status == "fail":
            return True
        if issue.issue_type in BLOCKING_TYPES and issue.severity in {"medium", "high"}:
            return True
        return False

    # ------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------

    def _check_schema(self, itinerary: dict[str, Any], issues: list[IssueItem]) -> None:
        if not isinstance(itinerary, dict):
            self._add_issue(issues, IssueItem(
                issue_type="invalid_itinerary_schema",
                day=None,
                poi_index=None,
                poi_name="",
                category="feasibility",
                severity="high",
                status="fail",
                description="itinerary가 dict 형식이 아닙니다.",
                evidence=[f"type={type(itinerary)}"],
                target_module="critic",
            ))
            return

        if not isinstance(itinerary.get("days"), list) or not itinerary.get("days"):
            self._add_issue(issues, IssueItem(
                issue_type="missing_days",
                day=None,
                poi_index=None,
                poi_name="",
                category="feasibility",
                severity="high",
                status="fail",
                description="itinerary에 days가 없거나 비어 있습니다.",
                evidence=[],
                target_module="critic",
            ))

    def _check_each_day(self, itinerary: dict[str, Any], issues: list[IssueItem], debug: dict[str, Any]) -> None:
        for day_idx, day in enumerate(itinerary.get("days", [])):
            day_num = get_day_number(day, day_idx)
            schedule = estimate_schedule_for_day(day)
            target_area = self.index.target_area_for_day(day, self.user_state)
            purpose = infer_purpose(self.user_state, day)

            debug["day_targets"][str(day_num)] = {
                "theme": day.get("theme"),
                "target_area": target_area,
                "purpose": purpose,
                "poi_count": len(day.get("pois", [])),
            }

            self._check_day_density(day, day_num, schedule, target_area, purpose, issues)
            self._check_poi_validity(day, day_num, schedule, target_area, purpose, issues)
            self._check_cluster_consistency(day, day_num, schedule, target_area, issues)
            self._check_representative_anchor(day, day_num, schedule, target_area, purpose, issues)
            self._check_meal_coverage(day, day_num, schedule, target_area, issues)
            self._check_opening_hours(day, day_num, schedule, issues)
            self._check_durations(day, day_num, schedule, issues)
            self._check_route(day, day_num, schedule, issues)
            self._check_flow(day, day_num, schedule, purpose, issues)
            self._check_foreignness(day, day_num, schedule, issues)

    def _check_day_density(
        self,
        day: dict[str, Any],
        day_num: int,
        schedule: list[dict[str, Any]],
        target_area: str | None,
        purpose: str,
        issues: list[IssueItem],
    ) -> None:
        poi_count = len(schedule)
        half_day = is_half_day(self.user_state, day, schedule)
        min_required = MIN_POIS_HALF_DAY if half_day else MIN_POIS_FULL_DAY

        if poi_count < min_required:
            self._add_issue(issues, IssueItem(
                issue_type="too_sparse_day",
                day=day_num,
                poi_index=None,
                poi_name="",
                category="planning_quality",
                severity="medium",
                status="warning",
                description=(
                    f"Day {day_num}의 POI 수가 {poi_count}개로 부족합니다. "
                    f"{'반나절' if half_day else '일반 하루 일정'} 기준 최소 {min_required}개가 필요합니다."
                ),
                evidence=[
                    f"poi_count={poi_count}",
                    f"min_required={min_required}",
                    f"target_area={target_area}",
                    f"purpose={purpose}",
                ],
                target_module="replanner",
                repairable=True,
                suggested_action="area_profiles_v2 기반으로 같은 권역 후보를 추가해 day를 재구성",
            ))

        if poi_count > MAX_POIS_PER_DAY:
            self._add_issue(issues, IssueItem(
                issue_type="too_dense_day",
                day=day_num,
                poi_index=None,
                poi_name="",
                category="planning_quality",
                severity="low",
                status="warning",
                description=f"Day {day_num}의 POI 수가 {poi_count}개로 많아 일정이 과밀할 수 있습니다.",
                evidence=[f"poi_count={poi_count}", f"max={MAX_POIS_PER_DAY}"],
                target_module="replanner",
                repairable=True,
                suggested_action="체류시간과 이동시간을 고려해 후보를 줄이거나 하루를 분리",
            ))

    def _check_poi_validity(
        self,
        day: dict[str, Any],
        day_num: int,
        schedule: list[dict[str, Any]],
        target_area: str | None,
        purpose: str,
        issues: list[IssueItem],
    ) -> None:
        for idx, p in enumerate(schedule):
            name = poi_name(p)
            prof = self.index.find_poi(name)

            if not name:
                self._add_issue(issues, IssueItem(
                    issue_type="missing_poi_name",
                    day=day_num,
                    poi_index=idx,
                    poi_name="",
                    category="feasibility",
                    severity="high",
                    status="fail",
                    description=f"Day {day_num}의 {idx+1}번째 POI 이름이 없습니다.",
                    evidence=[],
                    target_module="repair",
                    repairable=True,
                    suggested_action="POI 이름을 보강하거나 후보를 교체",
                ))
                continue

            if is_broad_or_vague_poi(p):
                self._add_issue(issues, IssueItem(
                    issue_type="vague_poi",
                    day=day_num,
                    poi_index=idx,
                    poi_name=name,
                    category="feasibility",
                    severity="medium",
                    status="warning",
                    description=f"{name}은 구체 방문 장소가 아니라 넓은 권역/거리/동네명에 가깝습니다.",
                    evidence=[
                        f"name={name}",
                        f"type={poi_type(p)}",
                        "broad_or_vague_name_detected",
                    ],
                    target_module="replanner",
                    repairable=True,
                    suggested_action="area profile 기반의 구체 POI 묶음으로 day 재구성",
                ))

            if prof is None and self.profile_loaded:
                # Google Places에서 가져온 식당/카페는 not_in_profile 체크 면제
                # source 필드가 "Google Places"로 시작하거나, poi_type이 식사 관련이면 허용
                poi_source = clean_str(p.get("source", "")).lower()
                poi_type_val = normalize_name(poi_type(p))
                is_google_places_poi = (
                    "google places" in poi_source
                    or "google" in poi_source
                )
                is_meal_type = poi_type_val in {"restaurant", "cafe", "food", "market"}

                if is_google_places_poi and is_meal_type:
                    # Google Places 식당/카페는 profile에 없어도 허용
                    pass
                else:
                    self._add_issue(issues, IssueItem(
                        issue_type="not_in_profile",
                        day=day_num,
                        poi_index=idx,
                        poi_name=name,
                        category="feasibility",
                        severity="medium",
                        status="warning",
                        description=f"{name}은 area_profiles_v2 후보 pool에서 확인되지 않았습니다.",
                        evidence=[f"profile={self.profile_path}"],
                        target_module="replanner",
                        repairable=True,
                        suggested_action="POI master/Google Places 후보를 확인하거나 profile 기반 후보로 교체",
                    ))
                    continue

            if prof and is_bad_general_poi(p, prof, purpose):
                self._add_issue(issues, IssueItem(
                    issue_type="bad_general_poi",
                    day=day_num,
                    poi_index=idx,
                    poi_name=name,
                    category="planning_quality",
                    severity="medium",
                    status="warning",
                    description=f"{name}은 일반 여행 목적의 대표 POI로는 부적절하거나 특수 목적 후보입니다.",
                    evidence=[
                        f"purpose={purpose}",
                        f"profile_roles={prof.get('roles')}",
                        f"profile_cluster={prof.get('cluster')}",
                    ],
                    target_module="replanner",
                    repairable=True,
                    suggested_action="사용자 목적이 명확하지 않다면 대표 관광/문화/쇼핑 후보로 교체",
                ))

    def _check_cluster_consistency(
        self,
        day: dict[str, Any],
        day_num: int,
        schedule: list[dict[str, Any]],
        target_area: str | None,
        issues: list[IssueItem],
    ) -> None:
        if not target_area:
            return

        clusters = []
        incompatible = []

        for idx, p in enumerate(schedule):
            prof = self.index.find_poi(poi_name(p))
            if not prof:
                continue

            cluster = clean_str(prof.get("cluster"))
            if cluster:
                clusters.append(cluster)

            if not self.index.is_profile_poi_compatible(prof, target_area):
                incompatible.append((idx, p, prof))

        for idx, p, prof in incompatible:
            # Google Places에서 가져온 식당/카페는 off_theme 체크 면제
            # (인근 실시간 검색 결과는 권역 일관성보다 실용성이 우선)
            poi_source = clean_str(p.get("source", "")).lower()
            poi_type_val = normalize_name(poi_type(p))
            if "google" in poi_source and poi_type_val in {"restaurant", "cafe", "food", "market"}:
                continue

            self._add_issue(issues, IssueItem(
                issue_type="off_theme_cluster_poi",
                day=day_num,
                poi_index=idx,
                poi_name=poi_name(p),
                category="spatial_consistency",
                severity="medium",
                status="warning",
                description=(
                    f"Day {day_num}의 {poi_name(p)}은 day target area({target_area})와 다른 권역 "
                    f"({prof.get('cluster')})의 POI입니다."
                ),
                evidence=[
                    f"target_area={target_area}",
                    f"poi_cluster={prof.get('cluster')}",
                    f"compatible_clusters={sorted(self.index.compatible_clusters(target_area))}",
                ],
                target_module="replanner",
                repairable=True,
                suggested_action="target area와 호환되는 후보만으로 day 재구성",
            ))

        if clusters:
            most_common = max(set(clusters), key=clusters.count)
            ratio = clusters.count(most_common) / len(clusters)
            if ratio < 0.60:
                self._add_issue(issues, IssueItem(
                    issue_type="cluster_scattered",
                    day=day_num,
                    poi_index=None,
                    poi_name="",
                    category="spatial_consistency",
                    severity="medium",
                    status="warning",
                    description=f"Day {day_num}의 권역 일관성이 낮습니다. 주요 권역 유지율이 {ratio:.0%}입니다.",
                    evidence=[
                        f"clusters={clusters}",
                        f"dominant_cluster={most_common}",
                        f"dominant_ratio={ratio:.3f}",
                    ],
                    target_module="replanner",
                    repairable=True,
                    suggested_action="하루를 하나의 target area 기준으로 다시 구성",
                ))

    def _check_representative_anchor(
        self,
        day: dict[str, Any],
        day_num: int,
        schedule: list[dict[str, Any]],
        target_area: str | None,
        purpose: str,
        issues: list[IssueItem],
    ) -> None:
        if not target_area or not self.profile_loaded:
            return

        top5 = self.index.top_representative_keys(target_area, top_k=5)
        top12 = self.index.top_representative_keys(target_area, top_k=12)
        if not top12:
            return

        day_keys = {dedupe_key(poi_name(p)) for p in schedule}
        top5_hits = sorted(day_keys & top5)
        top12_hits = sorted(day_keys & top12)

        if len(top5_hits) == 0:
            self._add_issue(issues, IssueItem(
                issue_type="no_representative_anchor",
                day=day_num,
                poi_index=None,
                poi_name="",
                category="planning_quality",
                severity="medium",
                status="warning",
                description=f"Day {day_num}에는 {target_area}의 상위 대표 POI가 포함되어 있지 않습니다.",
                evidence=[
                    f"target_area={target_area}",
                    f"top_representatives={self.index.top_representative_names(target_area, top_k=5)}",
                    f"day_pois={[poi_name(p) for p in schedule]}",
                ],
                target_module="replanner",
                repairable=True,
                suggested_action="상위 representative POI를 최소 1개 이상 포함하도록 day 재구성",
            ))
        elif len(top12_hits) < 2 and len(schedule) >= 4:
            self._add_issue(issues, IssueItem(
                issue_type="weak_representative_anchor",
                day=day_num,
                poi_index=None,
                poi_name="",
                category="planning_quality",
                severity="low",
                status="warning",
                description=f"Day {day_num}의 {target_area} 대표 후보 비중이 낮습니다.",
                evidence=[
                    f"top5_hits={top5_hits}",
                    f"top12_hits={top12_hits}",
                    f"top_representatives={self.index.top_representative_names(target_area, top_k=8)}",
                ],
                target_module="replanner",
                repairable=True,
                suggested_action="대표 POI를 1개 더 포함하도록 후보 조합 개선",
            ))

    def _check_meal_coverage(
        self,
        day: dict[str, Any],
        day_num: int,
        schedule: list[dict[str, Any]],
        target_area: str | None,
        issues: list[IssueItem],
    ) -> None:
        if not schedule:
            return

        start = min(p["_start_min"] for p in schedule)
        end = max(p["_end_min"] for p in schedule)
        duration = end - start
        poi_count = len(schedule)

        for meal_name, (slot_start, slot_end) in MEAL_SLOTS.items():
            if meal_name == "lunch":
                required = interval_overlaps(start, end, slot_start, slot_end) and (poi_count >= 2 or duration >= 180)
            else:
                required = interval_overlaps(start, end, slot_start, slot_end) and (poi_count >= 5 or end >= 17 * 60 or duration >= 420)

            if not required:
                continue

            covered = []
            for idx, p in enumerate(schedule):
                prof = self.index.find_poi(poi_name(p))
                if interval_overlaps(p["_start_min"], p["_end_min"], slot_start, slot_end) and is_meal_like(p, prof):
                    covered.append((idx, p))

            if not covered:
                self._add_issue(issues, IssueItem(
                    issue_type=f"{meal_name}_missing",
                    day=day_num,
                    poi_index=None,
                    poi_name="",
                    category="feasibility",
                    severity="medium",
                    status="warning",
                    description=(
                        f"Day {day_num}의 {meal_name} 시간대 "
                        f"({minutes_to_hhmm(slot_start)}-{minutes_to_hhmm(slot_end)})에 식사 가능한 후보가 없습니다."
                    ),
                    evidence=[
                        f"day_schedule={[(poi_name(p), p['_start'], p['_end'], poi_type(p)) for p in schedule]}",
                        f"target_area={target_area}",
                    ],
                    target_module="repair",
                    repairable=True,
                    suggested_action="같은 target area 안에서 식사/시장 후보를 해당 시간대에 삽입",
                ))

    def _check_opening_hours(self, day: dict[str, Any], day_num: int, schedule: list[dict[str, Any]], issues: list[IssueItem]) -> None:
        for idx, p in enumerate(schedule):
            opening_hours = p.get("opening_hours")
            if opening_hours is None:
                # profile에는 opening_hours가 없을 수 있으므로 low warning으로만 둔다.
                continue

            status, reason = check_open_during_interval(opening_hours, "mon", p["_start_min"], p["_end_min"])
            if status == "conflict" or status == "closed":
                self._add_issue(issues, IssueItem(
                    issue_type="oh_conflict",
                    day=day_num,
                    poi_index=idx,
                    poi_name=poi_name(p),
                    category="feasibility",
                    severity="high",
                    status="fail",
                    description=f"{poi_name(p)}의 예상 방문 시간이 운영시간과 충돌합니다.",
                    evidence=[reason, f"visit={p['_start']}-{p['_end']}"],
                    target_module="repair",
                    repairable=True,
                    suggested_action="방문 시간을 조정하거나 같은 권역 내 운영시간이 맞는 후보로 교체",
                ))
            elif status in {"missing", "unknown"}:
                self._add_issue(issues, IssueItem(
                    issue_type="oh_missing",
                    day=day_num,
                    poi_index=idx,
                    poi_name=poi_name(p),
                    category="feasibility",
                    severity="low",
                    status="warning",
                    description=f"{poi_name(p)}의 운영시간 정보가 부족합니다.",
                    evidence=[reason],
                    target_module="repair",
                    repairable=False,
                    suggested_action="Google Places 또는 공식 출처로 운영시간 보강",
                ))

    def _check_durations(self, day: dict[str, Any], day_num: int, schedule: list[dict[str, Any]], issues: list[IssueItem]) -> None:
        for idx, p in enumerate(schedule):
            t = normalize_name(poi_type(p)).replace(" ", "_")
            stay = p.get("_stay_minutes") or default_stay_minutes(p)
            lo, hi = DURATION_RANGE.get(t, (30, 180))

            if stay < lo or stay > hi:
                self._add_issue(issues, IssueItem(
                    issue_type="duration_out_of_range",
                    day=day_num,
                    poi_index=idx,
                    poi_name=poi_name(p),
                    category="flow_quality",
                    severity="low",
                    status="warning",
                    description=f"{poi_name(p)}의 체류시간 {stay}분이 권장 범위 {lo}-{hi}분을 벗어납니다.",
                    evidence=[f"type={t}", f"stay={stay}", f"range={lo}-{hi}"],
                    target_module="repair",
                    repairable=True,
                    suggested_action="stay_minutes를 권장 범위에 맞게 조정",
                ))

    def _check_route(self, day: dict[str, Any], day_num: int, schedule: list[dict[str, Any]], issues: list[IssueItem]) -> None:
        distances = []
        for idx in range(len(schedule) - 1):
            a = schedule[idx]
            b = schedule[idx + 1]
            lat1 = safe_float(a.get("lat"))
            lng1 = safe_float(a.get("lng"))
            lat2 = safe_float(b.get("lat"))
            lng2 = safe_float(b.get("lng"))

            if None in {lat1, lng1, lat2, lng2}:
                continue

            km = haversine_km(lat1, lng1, lat2, lng2)
            distances.append(km)

            if km > MAX_SIMPLE_DISTANCE_KM:
                self._add_issue(issues, IssueItem(
                    issue_type="complex_transfer",
                    day=day_num,
                    poi_index=idx,
                    poi_name=f"{poi_name(a)} → {poi_name(b)}",
                    category="mobility",
                    severity="low",
                    status="warning",
                    description=f"Day {day_num}의 {poi_name(a)} → {poi_name(b)} 이동거리가 {km:.1f}km로 길 수 있습니다.",
                    evidence=[f"distance_km={km:.3f}"],
                    target_module="replanner",
                    repairable=True,
                    suggested_action="같은 권역 후보로 재구성하거나 이동 순서 재조정",
                ))

        if distances:
            avg = sum(distances) / len(distances)
            if avg > MAX_AVG_DISTANCE_KM:
                self._add_issue(issues, IssueItem(
                    issue_type="route_too_spread",
                    day=day_num,
                    poi_index=None,
                    poi_name="",
                    category="mobility",
                    severity="medium",
                    status="warning",
                    description=f"Day {day_num}의 평균 POI 간 거리가 {avg:.1f}km로 넓게 퍼져 있습니다.",
                    evidence=[f"distances={[round(d, 2) for d in distances]}", f"avg={avg:.3f}"],
                    target_module="replanner",
                    repairable=True,
                    suggested_action="권역 일관성을 유지하도록 day 재구성",
                ))

    def _check_flow(self, day: dict[str, Any], day_num: int, schedule: list[dict[str, Any]], purpose: str, issues: list[IssueItem]) -> None:
        meal_like_flags = []
        attraction_count = 0

        for p in schedule:
            prof = self.index.find_poi(poi_name(p))
            meal_like = is_meal_like(p, prof)
            meal_like_flags.append(meal_like)
            if is_attraction_like(p, prof) and not meal_like:
                attraction_count += 1

        for idx in range(len(meal_like_flags) - 1):
            if meal_like_flags[idx] and meal_like_flags[idx + 1]:
                # Google Places에서 가져온 식당/카페가 연속이면 severity를 low로 낮춤
                # (Google Places 식당은 일정 중 자연스러운 식사 슬롯으로 허용)
                p_a = schedule[idx]
                p_b = schedule[idx + 1]
                src_a = clean_str(p_a.get("source", "")).lower()
                src_b = clean_str(p_b.get("source", "")).lower()
                either_google = "google" in src_a or "google" in src_b
                sev = "low" if either_google else "medium"

                self._add_issue(issues, IssueItem(
                    issue_type="consecutive_meal_like_pois",
                    day=day_num,
                    poi_index=idx,
                    poi_name=f"{poi_name(schedule[idx])} → {poi_name(schedule[idx + 1])}",
                    category="flow_quality",
                    severity=sev,
                    status="warning",
                    description=f"Day {day_num}에 식사/카페/시장 성격의 POI가 연속 배치되어 흐름이 단조롭습니다.",
                    evidence=[
                        f"poi_a={poi_name(schedule[idx])}",
                        f"poi_b={poi_name(schedule[idx + 1])}",
                    ],
                    target_module="replanner",
                    repairable=True,
                    suggested_action="attraction과 meal/cafe가 교차되도록 day 재구성",
                ))
                break

        if len(schedule) >= 4 and attraction_count < 2 and purpose != "food":
            self._add_issue(issues, IssueItem(
                issue_type="weak_attraction_mix",
                day=day_num,
                poi_index=None,
                poi_name="",
                category="flow_quality",
                severity="low",
                status="warning",
                description=f"Day {day_num}의 attraction 성격 후보가 {attraction_count}개로 적습니다.",
                evidence=[f"attraction_count={attraction_count}", f"purpose={purpose}"],
                target_module="replanner",
                repairable=True,
                suggested_action="대표 관광/문화/쇼핑 anchor를 추가해 day 구성 보완",
            ))

    def _check_foreignness(self, day: dict[str, Any], day_num: int, schedule: list[dict[str, Any]], issues: list[IssueItem]) -> None:
        for idx, p in enumerate(schedule):
            notes = clean_str(p.get("notes"))
            name = poi_name(p)
            ptype = normalize_name(poi_type(p))
            prof = self.index.find_poi(name)

            roles = set(prof.get("roles") or []) if prof else set()
            needs_tip = (
                ptype in {"history", "culture", "market", "street"}
                or roles & {"history", "culture", "market", "kpop"}
            )

            if needs_tip and len(notes) < 20:
                self._add_issue(issues, IssueItem(
                    issue_type="missing_foreigner_tip",
                    day=day_num,
                    poi_index=idx,
                    poi_name=name,
                    category="foreignness",
                    severity="low",
                    status="warning",
                    description=f"{name}은 외국인에게 설명이 있으면 좋은 장소지만 notes가 부족합니다.",
                    evidence=[f"type={ptype}", f"roles={sorted(roles)}", f"notes_len={len(notes)}"],
                    target_module="repair",
                    repairable=True,
                    suggested_action="외국인용 문화/이용 팁을 notes에 추가",
                ))

    # ------------------------------------------------------------
    # Score building
    # ------------------------------------------------------------

    def _build_score_breakdown(self, itinerary: dict[str, Any], issues: list[IssueItem]) -> dict[str, dict[str, Any]]:
        total_pois = max(1, sum(len(d.get("pois", [])) for d in itinerary.get("days", [])))
        total_days = max(1, len(itinerary.get("days", [])))

        metric_defs = {
            "within_sandbox": {
                "category": "feasibility",
                "types": {"vague_poi", "not_in_profile", "missing_poi_name"},
                "denom": total_pois,
            },
            "complete_info": {
                "category": "feasibility",
                "types": {"missing_days", "invalid_itinerary_schema"},
                "denom": total_days,
            },
            "opening_hours": {
                "category": "feasibility",
                "types": {"oh_conflict", "oh_missing"},
                "denom": total_pois,
            },
            "meal_coverage": {
                "category": "feasibility",
                "types": {"lunch_missing", "dinner_missing"},
                "denom": total_days,
            },
            "day_density": {
                "category": "planning_quality",
                "types": {"too_sparse_day", "too_dense_day"},
                "denom": total_days,
            },
            "cluster_preservation": {
                "category": "spatial_consistency",
                "types": {"off_theme_cluster_poi", "cluster_scattered"},
                "denom": total_days,
            },
            "representative_anchor": {
                "category": "planning_quality",
                "types": {"no_representative_anchor", "weak_representative_anchor", "bad_general_poi"},
                "denom": total_days,
            },
            "mobility_complexity": {
                "category": "mobility",
                "types": {"complex_transfer", "route_too_spread"},
                "denom": total_days,
            },
            "flow_balance": {
                "category": "flow_quality",
                "types": {"consecutive_meal_like_pois", "weak_attraction_mix", "duration_out_of_range"},
                "denom": total_pois,
            },
            "foreignness": {
                "category": "foreignness",
                "types": {"missing_foreigner_tip", "cultural_friction_unexplained"},
                "denom": total_pois,
            },
        }

        breakdown = {}
        for metric, spec in metric_defs.items():
            related = [i for i in issues if i.issue_type in spec["types"]]
            penalty = sum(severity_weight(i.severity) for i in related)
            score = max(0.0, 1.0 - penalty / max(spec["denom"], 1))
            confidence = "high" if self.profile_loaded else "medium"
            if any(i.severity == "high" for i in related):
                confidence = "medium"

            breakdown[metric] = asdict(MetricResult(
                score=round(score, 3),
                status=metric_status(score),
                confidence=confidence,
                evidence=[
                    f"related_issue_count={len(related)}",
                    f"penalty={penalty:.3f}",
                    f"denom={spec['denom']}",
                ],
                issues=[issue_to_dict(i) for i in related],
            ))

        return breakdown

    def _build_category_scores(self, score_breakdown: dict[str, dict[str, Any]]) -> dict[str, float]:
        groups = {
            "feasibility": ["within_sandbox", "complete_info", "opening_hours", "meal_coverage"],
            "planning_quality": ["day_density", "representative_anchor", "flow_balance"],
            "spatial_mobility": ["cluster_preservation", "mobility_complexity"],
            "foreignness": ["foreignness"],
        }

        out = {}
        for group, metrics in groups.items():
            vals = [score_breakdown[m]["score"] for m in metrics if m in score_breakdown]
            out[group] = round(sum(vals) / max(len(vals), 1), 3)
        return out


# ============================================================
# Public function
# ============================================================

def evaluate_itinerary(
    itinerary: dict[str, Any],
    user_state: dict[str, Any] | None = None,
    profile_path: str | Path = "output/area_profiles_v2.json",
) -> dict[str, Any]:
    critic = CriticAgent(profile_path=profile_path, user_state=user_state or {})
    return critic.evaluate(itinerary)


# ============================================================
# CLI
# ============================================================

def sample_itinerary() -> dict[str, Any]:
    return {
        "summary": "critic test itinerary",
        "days": [
            {
                "day": 1,
                "theme": "홍대 & 연남동",
                "pois": [
                    {"name": "Hongdae (Hongik University Street) (홍대)", "type": "street", "lat": 37.5563, "lng": 126.9227, "stay_minutes": 90},
                    {"name": "Yeonnam-dong Cafe Street (연남동 카페거리)", "type": "cafe", "lat": 37.562, "lng": 126.923, "stay_minutes": 60},
                    {"name": "Gyeongbokgung Palace (경복궁)", "type": "history", "lat": 37.5796, "lng": 126.9770, "stay_minutes": 120},
                ],
            },
            {
                "day": 2,
                "theme": "강남",
                "pois": [
                    {"name": "Gangnam (강남)", "type": "street", "lat": 37.4979, "lng": 127.0276, "stay_minutes": 90},
                ],
            },
        ],
    }


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate itinerary without modifying it.")
    parser.add_argument("--profile", type=str, default="output/area_profiles_v2.json")
    parser.add_argument("--input", type=str, default="", help="Input itinerary JSON. If omitted, output/replanned_itinerary.json or sample is used.")
    parser.add_argument("--output", type=str, default="output/critic_result.json")
    parser.add_argument("--purpose", type=str, default="", help="Optional user purpose, e.g. general, shopping, kpop")
    parser.add_argument("--location", type=str, default="", help="Optional user location/area hint")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = base_dir / input_path
        itinerary = load_json(input_path)
    else:
        default_input = base_dir / "output" / "replanned_itinerary.json"
        if default_input.exists():
            itinerary = load_json(default_input)
            print(f"[critic] default input: {default_input}")
        else:
            itinerary = sample_itinerary()
            print("[critic] built-in sample input")

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = base_dir / profile_path

    user_state = {}
    if args.purpose:
        user_state["purpose"] = args.purpose
    if args.location:
        user_state["location"] = args.location

    result = evaluate_itinerary(
        itinerary=itinerary,
        user_state=user_state,
        profile_path=profile_path,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    json_safe_dump(output_path, result)

    print("\n[critic] 완료")
    print(f"  overall_score : {result['overall_score']}")
    print(f"  summary_status: {result['summary_status']}")
    print(f"  passed        : {result['passed']}")
    print(f"  needs_repair  : {result['needs_repair']} days={result['days_needing_repair']}")
    print(f"  needs_replan  : {result['needs_replan']} days={result['days_needing_replan']}")
    print(f"  output        : {output_path}")

    print("\n[category_scores]")
    for k, v in result["category_scores"].items():
        print(f"  - {k}: {v}")

    print("\n[score_breakdown]")
    for k, v in result["score_breakdown"].items():
        print(f"  - {k}: {v['score']} ({v['status']}) issues={len(v['issues'])}")

    print("\n[issues]")
    for issue in result["issues"]:
        print(
            f"  - Day {issue.get('day')} "
            f"[{issue.get('severity').upper()}] "
            f"{issue.get('issue_type')} → {issue.get('target_module')}: "
            f"{issue.get('description')}"
        )


if __name__ == "__main__":
    main()