"""
critic_repair.py

Generator가 생성한 itinerary를 평가(Critic)하고 자동 수정(Repair)하는 모듈.

평가 지표 (중간보고서 + TripTailor 논문 기반):
  1. Feasibility Score  F(p) = 1/6 * (WS + CI + OH + MC + TF + BV)
  2. ECS               ECS(p) = 1/5 * (RD + VD + SS + CP + FS)
  3. Foreignness Score FG(p)  = 1/5 * (LA + AB + MB + CF + CC)

입력:
  - itinerary: Generator가 생성한 일정 dict (planner.py 스키마)
  - poi_db: poi_master_step3.csv (Google + Gemini 라벨 포함)
  - user_state: TravelState (budget, dietary, duration 등)
  - reference_courses: course_data.json (ECS 계산용 reference)

출력:
  - CriticResult: 각 점수 + 문제 목록
  - RepairResult: 수정된 itinerary + 변경 이력
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ============================================================
# 상수 및 설정
# ============================================================

# 하루 이동시간 threshold (분)
MAX_DAILY_TRAVEL_MINUTES = 180

# POI 간 단일 이동 threshold (분)
MAX_SINGLE_TRAVEL_MINUTES = 60

# 하루 최대 POI 수
MAX_POIS_PER_DAY = 8

# 환승 복잡도 threshold (고복잡 이동으로 판단하는 환승 횟수)
HIGH_COMPLEXITY_TRANSFERS = 2

# 식사 필요 시간대 (이 범위 POI 중 restaurant/cafe 없으면 식사 누락)
MEAL_SLOTS = {
    "lunch": (11, 14),   # 11~14시
    "dinner": (17, 20),  # 17~20시
}

# POI type → activity category (FS 계산용)
POI_TYPE_TO_CATEGORY = {
    "museum": "sightseeing",
    "history": "sightseeing",
    "park": "sightseeing",
    "kpop_landmark": "sightseeing",
    "shopping": "sightseeing",
    "street": "sightseeing",
    "tourist_spot": "sightseeing",
    "culture": "sightseeing",
    "market": "sightseeing",
    "nature": "sightseeing",
    "nightlife": "sightseeing",
    "cafe": "rest",
    "restaurant": "meal",
}

# POI type별 권장 체류시간 (분) [min, max]
DURATION_RANGE = {
    "museum":       (60, 120),
    "history":      (60, 240),
    "park":         (30, 90),
    "kpop_landmark":(30, 60),
    "shopping":     (60, 180),
    "street":       (30, 60),
    "tourist_spot": (30, 90),
    "cafe":         (30, 90),
    "restaurant":   (60, 90),
    "culture":      (45, 120),
    "market":       (45, 90),
    "nature":       (30, 90),
    "nightlife":    (60, 180),
}

# 서울 권역 클러스터 (lat/lng 중심점 기반)
SEOUL_CLUSTERS = {
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
}

CLUSTER_RADIUS_KM = 1.5  # 이 반경 내면 같은 권역으로 간주

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class POIInfo:
    """itinerary의 POI 하나를 표현."""
    name: str
    poi_type: str
    address: str
    lat: float | None
    lng: float | None
    stay_minutes: int
    notes: str = ""

    # step3에서 채워지는 enrichment 필드
    opening_hours: dict | None = None   # {"mon": [["09:00","18:00"]], ...}
    price_level: int | None = None
    english_support: int = 0
    reservation_required: int = 0
    cash_only: int = 0
    cultural_friction: int = 0
    cultural_friction_reason: str = ""
    label_confidence: str = "low"


@dataclass
class DayPlan:
    """하루 일정."""
    day: int
    theme: str
    pois: list[POIInfo]
    estimated_cost: str = ""


@dataclass
class IssueItem:
    """Critic이 발견한 문제 하나."""
    category: str       # feasibility / ecs / foreignness
    issue_type: str     # oh_conflict / travel_time / meal_missing 등
    day: int
    poi_index: int | None
    description: str
    severity: str       # high / medium / low


@dataclass
class CriticResult:
    """Critic Agent 평가 결과."""
    feasibility_score: float
    ecs_score: float
    foreignness_score: float
    total_score: float

    # 세부 점수
    ws: float = 0.0   # Within Sandbox
    ci: float = 0.0   # Complete Information
    oh: float = 0.0   # Opening Hours
    mc: float = 0.0   # Meal Coverage
    tf: float = 0.0   # Transport Feasibility
    bv: float = 0.0   # Budget Validity

    rd: float = 0.0   # Route Distance
    vd: float = 0.0   # Visit Duration
    ss: float = 0.0   # Sequence Similarity
    cp: float = 0.0   # Cluster Preservation
    fs: float = 0.0   # Flow Similarity

    la: float = 0.0   # Language Accessibility
    ab: float = 0.0   # Access Barrier
    mb: float = 0.0   # Mobility Complexity
    cf: float = 0.0   # Cultural Friction
    cc: float = 0.0   # Constraint Compatibility

    issues: list[IssueItem] = field(default_factory=list)
    passed: bool = False


@dataclass
class RepairAction:
    """Repair Agent가 수행한 수정 하나."""
    action_type: str    # replace_poi / insert_slot / remove_poi / reorder
    day: int
    poi_index: int | None
    description: str
    before: str = ""
    after: str = ""


@dataclass
class RepairResult:
    """Repair Agent 수정 결과."""
    itinerary: dict
    actions: list[RepairAction]
    success: bool
    message: str = ""


# ============================================================
# 유틸 함수
# ============================================================

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 좌표 간 거리 (km)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_minutes(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    두 좌표 간 예상 이동시간 (분).
    서울 대중교통 평균 속도 약 20km/h + 환승 버퍼 10분.
    """
    km = haversine_km(lat1, lng1, lat2, lng2)
    return (km / 20.0) * 60 + 10


def assign_cluster(lat: float, lng: float) -> str:
    """좌표를 서울 권역 클러스터에 배정."""
    min_dist = float("inf")
    best = "other"
    for name, (clat, clng) in SEOUL_CLUSTERS.items():
        d = haversine_km(lat, lng, clat, clng)
        if d < min_dist:
            min_dist = d
            best = name
    return best if min_dist <= CLUSTER_RADIUS_KM else "other"


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def is_open_at(opening_hours: dict, day_of_week: str, hour: int) -> bool | None:
    """
    특정 요일/시간에 영업 중인지 확인.
    opening_hours: {"mon": [["09:00","18:00"]], "tue": null, ...}
    day_of_week: "mon", "tue", ...
    hour: 0~23 (정수)
    반환: True=영업중, False=영업안함, None=정보없음
    """
    if not opening_hours:
        return None
    slots = opening_hours.get(day_of_week)
    if slots is None:
        return False  # 해당 요일 null → 휴무
    if not slots:
        return None
    for slot in slots:
        try:
            open_h = int(slot[0].split(":")[0])
            close_h = int(slot[1].split(":")[0])
            if open_h <= hour < close_h:
                return True
        except Exception:
            continue
    return False


def parse_budget_krw(budget_str: str | None) -> float | None:
    """
    예산 문자열 → KRW 정수.
    예: "$500" → 670000, "50만원" → 500000, "1000000" → 1000000
    """
    if not budget_str:
        return None
    s = str(budget_str).strip()
    # USD
    m = re.search(r"\$\s*([\d,]+)", s)
    if m:
        usd = float(m.group(1).replace(",", ""))
        return usd * 1340  # 환율 근사값
    # 만원 단위
    m = re.search(r"([\d.]+)\s*만", s)
    if m:
        return float(m.group(1)) * 10000
    # 원 단위
    m = re.search(r"([\d,]+)\s*원", s)
    if m:
        return float(m.group(1).replace(",", ""))
    # 순수 숫자
    m = re.search(r"[\d,]+", s)
    if m:
        return float(m.group(0).replace(",", ""))
    return None


def call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Gemini API 호출 → 텍스트 반환."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 없음")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=60,
            )
            data = resp.json()
            if resp.status_code in [429, 500, 503]:
                time.sleep(min(2 ** attempt, 30))
                continue
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini 호출 실패")


# ============================================================
# POI DB 로더
# ============================================================

class POIDatabase:
    """
    poi_master_step3.csv를 로드해서 POI 정보를 제공.
    step3가 없으면 step2, 그것도 없으면 base poi_master.csv 사용.
    """

    def __init__(self, base_dir: str | Path = "."):
        base_dir = Path(base_dir)
        paths = [
            base_dir / "output" / "poi_master_step3.csv",
            base_dir / "output" / "poi_master_step2.csv",
            base_dir / "output" / "poi_master.csv",
        ]
        self._df = None
        for p in paths:
            if p.exists():
                self._df = pd.read_csv(p)
                print(f"[POIDatabase] 로드: {p} ({len(self._df)}개)")
                break
        if self._df is None:
            print("[POIDatabase] CSV 없음 — enrichment 없이 동작")
            self._df = pd.DataFrame()

        # poi_name 기준 인덱스
        self._name_index: dict[str, dict] = {}
        if not self._df.empty and "poi_name" in self._df.columns:
            for _, row in self._df.iterrows():
                self._name_index[str(row["poi_name"]).lower()] = row.to_dict()

    def get(self, poi_name: str) -> dict | None:
        """POI 이름으로 DB 조회 (퍼지 매칭)."""
        key = poi_name.lower()
        if key in self._name_index:
            return self._name_index[key]
        # 부분 매칭
        for db_name, row in self._name_index.items():
            if key in db_name or db_name in key:
                return row
        return None

    def get_by_type_and_cluster(
        self,
        poi_type: str,
        cluster: str,
        exclude_names: list[str],
        dietary: str | None = None,
    ) -> list[dict]:
        """
        특정 타입 + 권역의 대체 POI 후보 반환.
        exclude_names: 이미 일정에 있는 POI 제외.
        """
        if self._df.empty:
            return []
        candidates = []
        excl = [n.lower() for n in exclude_names]
        for _, row in self._df.iterrows():
            if str(row.get("poi_type", "")) != poi_type:
                continue
            name = str(row.get("poi_name", ""))
            if name.lower() in excl:
                continue
            lat = row.get("lat")
            lng = row.get("lng")
            if pd.isna(lat) or pd.isna(lng):
                continue
            if assign_cluster(float(lat), float(lng)) != cluster:
                continue
            # 식단 제약 체크 (vegetarian이면 cash_only 아닌 식당 우선)
            if dietary and "vegetarian" in dietary.lower():
                if poi_type == "restaurant" and row.get("cash_only") == 1:
                    continue
            candidates.append(row.to_dict())
        return candidates


# ============================================================
# itinerary 파서
# ============================================================

def parse_itinerary(itinerary: dict, poi_db: POIDatabase) -> list[DayPlan]:
    """Generator itinerary dict → DayPlan 리스트."""
    days = []
    for day_data in itinerary.get("days", []):
        pois = []
        for p in day_data.get("pois", []):
            lat = p.get("lat")
            lng = p.get("lng")
            try:
                lat = float(lat) if lat else None
                lng = float(lng) if lng else None
            except Exception:
                lat = lng = None

            poi = POIInfo(
                name=p.get("name", ""),
                poi_type=p.get("type", "tourist_spot"),
                address=p.get("address", ""),
                lat=lat,
                lng=lng,
                stay_minutes=int(p.get("stay_minutes") or 60),
                notes=p.get("notes", ""),
            )

            # DB에서 enrichment 데이터 주입
            db_row = poi_db.get(poi.name)
            if db_row:
                oh_raw = db_row.get("opening_hours")
                poi.opening_hours = safe_json_loads(oh_raw)
                poi.price_level = db_row.get("price_level")
                poi.english_support = int(db_row.get("english_support") or 0)
                poi.reservation_required = int(db_row.get("reservation_required") or 0)
                poi.cash_only = int(db_row.get("cash_only") or 0)
                poi.cultural_friction = int(db_row.get("cultural_friction") or 0)
                poi.cultural_friction_reason = str(db_row.get("cultural_friction_reason") or "")
                poi.label_confidence = str(db_row.get("label_confidence") or "low")
                # DB 좌표가 더 정확하면 덮어씀
                if pd.notna(db_row.get("lat")) and not lat:
                    poi.lat = float(db_row["lat"])
                    poi.lng = float(db_row["lng"])

            pois.append(poi)

        days.append(DayPlan(
            day=day_data.get("day", len(days) + 1),
            theme=day_data.get("theme", ""),
            pois=pois,
            estimated_cost=day_data.get("estimated_cost", ""),
        ))
    return days


# ============================================================
# Critic Agent
# ============================================================

class CriticAgent:
    """
    생성된 itinerary를 3개 축으로 평가.
    F(p), ECS(p), FG(p) 계산 + 문제 탐지.
    """

    def __init__(
        self,
        poi_db: POIDatabase,
        reference_courses: list[dict],
        user_state: dict,
    ):
        self.poi_db = poi_db
        self.reference_courses = reference_courses
        self.user_state = user_state
        # reference 코스의 평균 POI 간 이동거리 계산
        self._ref_avg_distance = self._calc_ref_avg_distance()
        self._ref_bigrams = self._calc_ref_bigrams()

    def evaluate(self, itinerary: dict) -> CriticResult:
        """itinerary 전체 평가."""
        days = parse_itinerary(itinerary, self.poi_db)
        issues: list[IssueItem] = []

        # ── 1. Feasibility ──────────────────────────────
        ws = self._score_within_sandbox(days, issues)
        ci = self._score_complete_info(itinerary, issues)
        oh = self._score_opening_hours(days, issues)
        mc = self._score_meal_coverage(days, issues)
        tf = self._score_transport_feasibility(days, issues)
        bv = self._score_budget_validity(days, issues)
        feasibility = (ws + ci + oh + mc + tf + bv) / 6

        # ── 2. ECS ──────────────────────────────────────
        rd = self._score_route_distance(days, issues)
        vd = self._score_visit_duration(days, issues)
        ss = self._score_sequence_similarity(days, issues)
        cp = self._score_cluster_preservation(days, issues)
        fs = self._score_flow_similarity(days, issues)
        ecs = (rd + vd + ss + cp + fs) / 5

        # ── 3. Foreignness ──────────────────────────────
        la = self._score_language_accessibility(days, issues)
        ab = self._score_access_barrier(days, issues)
        mb = self._score_mobility_complexity(days, issues)
        cf = self._score_cultural_friction(days, issues)
        cc = self._score_constraint_compatibility(days, issues)
        foreignness = (la + ab + mb + cf + cc) / 5

        total = (feasibility + ecs + foreignness) / 3
        passed = (
            feasibility >= 0.7
            and ecs >= 0.6
            and foreignness >= 0.6
            and total >= 0.65
        )

        return CriticResult(
            feasibility_score=round(feasibility, 3),
            ecs_score=round(ecs, 3),
            foreignness_score=round(foreignness, 3),
            total_score=round(total, 3),
            ws=ws, ci=ci, oh=oh, mc=mc, tf=tf, bv=bv,
            rd=rd, vd=vd, ss=ss, cp=cp, fs=fs,
            la=la, ab=ab, mb=mb, cf=cf, cc=cc,
            issues=issues,
            passed=passed,
        )

    # ── Feasibility 세부 지표 ────────────────────────────

    def _score_within_sandbox(self, days: list[DayPlan], issues: list) -> float:
        """WS: 모든 POI가 DB 내에 있는지."""
        total = sum(len(d.pois) for d in days)
        if total == 0:
            return 0.0
        matched = 0
        for d in days:
            for i, p in enumerate(d.pois):
                if self.poi_db.get(p.name):
                    matched += 1
                else:
                    issues.append(IssueItem(
                        category="feasibility", issue_type="not_in_sandbox",
                        day=d.day, poi_index=i,
                        description=f"{p.name} — POI DB에서 찾을 수 없음",
                        severity="low",
                    ))
        return matched / total

    def _score_complete_info(self, itinerary: dict, issues: list) -> float:
        """CI: 필수 필드(days, pois, lat/lng)가 모두 있는지."""
        score = 1.0
        if not itinerary.get("days"):
            issues.append(IssueItem("feasibility", "missing_days", 0, None,
                                    "days 필드 없음", "high"))
            return 0.0
        for day_data in itinerary["days"]:
            for p in day_data.get("pois", []):
                if not p.get("lat") or not p.get("lng"):
                    score -= 0.05  # 좌표 누락 시 감점
        return max(0.0, score)

    def _score_opening_hours(self, days: list[DayPlan], issues: list) -> float:
        """OH: 방문 시각이 운영시간 안에 있는지."""
        violations = 0
        scheduled = 0
        # 방문 시각 추정: 09:00 시작, 체류시간 누적
        for d in days:
            current_hour = 9
            for i, p in enumerate(d.pois):
                scheduled += 1
                if p.opening_hours:
                    day_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
                    # 요일 알 수 없으므로 평일(mon)로 체크
                    result = is_open_at(p.opening_hours, "mon", current_hour)
                    if result is False:
                        violations += 1
                        issues.append(IssueItem(
                            category="feasibility", issue_type="oh_conflict",
                            day=d.day, poi_index=i,
                            description=(
                                f"{p.name} — 방문 예정 {current_hour}시가 "
                                f"운영시간과 충돌 가능"
                            ),
                            severity="high",
                        ))
                # 다음 POI 방문 시각 업데이트
                current_hour += p.stay_minutes // 60
                if i + 1 < len(d.pois) and p.lat and d.pois[i + 1].lat:
                    current_hour += int(travel_minutes(
                        p.lat, p.lng, d.pois[i + 1].lat, d.pois[i + 1].lng
                    ) // 60)
        if scheduled == 0:
            return 1.0
        return 1.0 - violations / scheduled

    def _score_meal_coverage(self, days: list[DayPlan], issues: list) -> float:
        """MC: 점심/저녁 식사 슬롯이 있는지."""
        required = 0
        covered = 0
        for d in days:
            for meal_name in ["lunch", "dinner"]:
                required += 1
                has_meal = any(
                    p.poi_type in ["restaurant", "cafe", "market"]
                    for p in d.pois
                )
                if has_meal:
                    covered += 1
                else:
                    issues.append(IssueItem(
                        category="feasibility", issue_type="meal_missing",
                        day=d.day, poi_index=None,
                        description=f"Day {d.day} — {meal_name} 식사 슬롯 없음",
                        severity="medium",
                    ))
                    break  # 한 번만 체크
        if required == 0:
            return 1.0
        return covered / required

    def _score_transport_feasibility(self, days: list[DayPlan], issues: list) -> float:
        """TF: 연속 POI 간 이동시간이 물리적으로 충분한지."""
        total_transitions = 0
        feasible = 0
        for d in days:
            daily_travel = 0
            for i in range(len(d.pois) - 1):
                cur = d.pois[i]
                nxt = d.pois[i + 1]
                if cur.lat and nxt.lat:
                    total_transitions += 1
                    tm = travel_minutes(cur.lat, cur.lng, nxt.lat, nxt.lng)
                    daily_travel += tm
                    if tm <= MAX_SINGLE_TRAVEL_MINUTES:
                        feasible += 1
                    else:
                        issues.append(IssueItem(
                            category="feasibility", issue_type="travel_too_far",
                            day=d.day, poi_index=i,
                            description=(
                                f"Day {d.day} {cur.name} → {nxt.name}: "
                                f"이동시간 약 {tm:.0f}분 (기준 {MAX_SINGLE_TRAVEL_MINUTES}분 초과)"
                            ),
                            severity="high",
                        ))
            if daily_travel > MAX_DAILY_TRAVEL_MINUTES:
                issues.append(IssueItem(
                    category="feasibility", issue_type="daily_travel_excess",
                    day=d.day, poi_index=None,
                    description=(
                        f"Day {d.day} 하루 총 이동시간 약 {daily_travel:.0f}분 "
                        f"(기준 {MAX_DAILY_TRAVEL_MINUTES}분 초과)"
                    ),
                    severity="medium",
                ))
        if total_transitions == 0:
            return 1.0
        return feasible / total_transitions

    def _score_budget_validity(self, days: list[DayPlan], issues: list) -> float:
        """BV: 예산 초과 여부."""
        budget_krw = parse_budget_krw(self.user_state.get("budget"))
        if budget_krw is None:
            return 1.0  # 예산 정보 없으면 패스

        total_cost = 0
        price_map = {0: 0, 1: 10000, 2: 25000, 3: 50000, 4: 100000}
        for d in days:
            for p in d.pois:
                if p.price_level is not None:
                    total_cost += price_map.get(int(p.price_level), 15000)
                else:
                    total_cost += 15000  # 기본값

        if total_cost <= budget_krw:
            return 1.0
        else:
            ratio = budget_krw / total_cost
            issues.append(IssueItem(
                category="feasibility", issue_type="budget_exceeded",
                day=0, poi_index=None,
                description=(
                    f"예상 총 비용 약 {total_cost:,.0f}원이 "
                    f"예산 {budget_krw:,.0f}원 초과"
                ),
                severity="medium",
            ))
            return ratio

    # ── ECS 세부 지표 ────────────────────────────────────

    def _calc_ref_avg_distance(self) -> float:
        """reference 코스들의 평균 POI 간 이동거리 (km)."""
        distances = []
        for course in self.reference_courses:
            seq = course.get("sequence", [])
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if a.get("lat") and b.get("lat"):
                    distances.append(haversine_km(
                        float(a["lat"]), float(a["lng"]),
                        float(b["lat"]), float(b["lng"])
                    ))
        return sum(distances) / len(distances) if distances else 2.0

    def _calc_ref_bigrams(self) -> dict[tuple, int]:
        """reference 코스의 activity type bigram 빈도."""
        bigrams: dict[tuple, int] = {}
        for course in self.reference_courses:
            seq = course.get("sequence", [])
            cats = [POI_TYPE_TO_CATEGORY.get(p.get("poi_type", ""), "sightseeing") for p in seq]
            for i in range(len(cats) - 1):
                key = (cats[i], cats[i + 1])
                bigrams[key] = bigrams.get(key, 0) + 1
        return bigrams

    def _score_route_distance(self, days: list[DayPlan], issues: list) -> float:
        """RD: 하루 평균 이동거리가 reference와 유사한지."""
        daily_avgs = []
        for d in days:
            dists = []
            for i in range(len(d.pois) - 1):
                a, b = d.pois[i], d.pois[i + 1]
                if a.lat and b.lat:
                    dists.append(haversine_km(a.lat, a.lng, b.lat, b.lng))
            if dists:
                daily_avgs.append(sum(dists) / len(dists))

        if not daily_avgs:
            return 0.5
        gen_avg = sum(daily_avgs) / len(daily_avgs)
        ref_avg = self._ref_avg_distance

        if gen_avg > ref_avg * 2:
            issues.append(IssueItem(
                category="ecs", issue_type="route_too_spread",
                day=0, poi_index=None,
                description=(
                    f"평균 POI 간 거리 {gen_avg:.1f}km — "
                    f"reference 평균 {ref_avg:.1f}km의 2배 초과"
                ),
                severity="medium",
            ))
        return min(1.0, ref_avg / max(gen_avg, 0.1))

    def _score_visit_duration(self, days: list[DayPlan], issues: list) -> float:
        """VD: 각 POI 체류시간이 권장 범위 안에 있는지."""
        total = 0
        valid = 0
        for d in days:
            for i, p in enumerate(d.pois):
                total += 1
                lo, hi = DURATION_RANGE.get(p.poi_type, (30, 180))
                if lo <= p.stay_minutes <= hi:
                    valid += 1
                else:
                    issues.append(IssueItem(
                        category="ecs", issue_type="duration_out_of_range",
                        day=d.day, poi_index=i,
                        description=(
                            f"{p.name} 체류시간 {p.stay_minutes}분 — "
                            f"권장 범위 {lo}~{hi}분 벗어남"
                        ),
                        severity="low",
                    ))
        return valid / total if total > 0 else 1.0

    def _score_sequence_similarity(self, days: list[DayPlan], issues: list) -> float:
        """SS: 생성 POI 순서와 reference 코스의 Levenshtein 유사도."""
        gen_names = []
        for d in days:
            gen_names.extend([p.name for p in d.pois])

        best_sim = 0.0
        for course in self.reference_courses:
            ref_names = [p["poi_name"] for p in course.get("sequence", [])]
            if not ref_names:
                continue
            dist = _levenshtein(gen_names, ref_names)
            sim = 1.0 - dist / max(len(gen_names), len(ref_names))
            best_sim = max(best_sim, sim)

        return best_sim

    def _score_cluster_preservation(self, days: list[DayPlan], issues: list) -> float:
        """CP: 하루 안에서 동일 권역 POI 비율."""
        scores = []
        for d in days:
            if not d.pois:
                continue
            clusters = []
            for p in d.pois:
                if p.lat and p.lng:
                    clusters.append(assign_cluster(p.lat, p.lng))
                else:
                    clusters.append("other")

            if not clusters:
                continue
            from collections import Counter
            most_common_count = Counter(clusters).most_common(1)[0][1]
            ratio = most_common_count / len(clusters)
            scores.append(ratio)

            if ratio < 0.5:
                issues.append(IssueItem(
                    category="ecs", issue_type="cluster_scattered",
                    day=d.day, poi_index=None,
                    description=(
                        f"Day {d.day} 권역 유지율 {ratio:.0%} — "
                        "동선이 여러 지역에 분산됨"
                    ),
                    severity="medium",
                ))

        return sum(scores) / len(scores) if scores else 1.0

    def _score_flow_similarity(self, days: list[DayPlan], issues: list) -> float:
        """FS: 관광→식사→휴식 패턴이 reference bigram과 유사한지."""
        if not self._ref_bigrams:
            return 0.5

        gen_bigrams: dict[tuple, int] = {}
        for d in days:
            cats = [POI_TYPE_TO_CATEGORY.get(p.poi_type, "sightseeing") for p in d.pois]
            for i in range(len(cats) - 1):
                key = (cats[i], cats[i + 1])
                gen_bigrams[key] = gen_bigrams.get(key, 0) + 1

        if not gen_bigrams:
            return 0.5

        ref_keys = set(self._ref_bigrams.keys())
        gen_keys = set(gen_bigrams.keys())
        overlap = len(ref_keys & gen_keys)
        return overlap / len(ref_keys) if ref_keys else 0.5

    # ── Foreignness 세부 지표 ────────────────────────────

    def _score_language_accessibility(self, days: list[DayPlan], issues: list) -> float:
        """LA: 영어/다국어 안내 가능한 POI 비율."""
        total = sum(len(d.pois) for d in days)
        if total == 0:
            return 1.0
        supported = sum(
            p.english_support
            for d in days for p in d.pois
        )
        ratio = supported / total
        if ratio < 0.5:
            for d in days:
                for i, p in enumerate(d.pois):
                    if not p.english_support and p.label_confidence in ["high", "medium"]:
                        issues.append(IssueItem(
                            category="foreignness", issue_type="no_english",
                            day=d.day, poi_index=i,
                            description=f"{p.name} — 영어 안내 없음",
                            severity="medium",
                        ))
        return ratio

    def _score_access_barrier(self, days: list[DayPlan], issues: list) -> float:
        """AB: 예약필수/현금only 장벽이 없는 POI 비율."""
        total = sum(len(d.pois) for d in days)
        if total == 0:
            return 1.0
        barriers = 0
        for d in days:
            for i, p in enumerate(d.pois):
                if p.reservation_required or p.cash_only:
                    barriers += 1
                    issues.append(IssueItem(
                        category="foreignness", issue_type="access_barrier",
                        day=d.day, poi_index=i,
                        description=(
                            f"{p.name} — "
                            + ("예약 필수 " if p.reservation_required else "")
                            + ("현금 only" if p.cash_only else "")
                        ),
                        severity="medium",
                    ))
        return 1.0 - barriers / total

    def _score_mobility_complexity(self, days: list[DayPlan], issues: list) -> float:
        """MB: 복잡 환승(2회 이상 추정)이 없는 이동 비율."""
        total_transitions = 0
        simple = 0
        for d in days:
            for i in range(len(d.pois) - 1):
                cur, nxt = d.pois[i], d.pois[i + 1]
                if cur.lat and nxt.lat:
                    total_transitions += 1
                    km = haversine_km(cur.lat, cur.lng, nxt.lat, nxt.lng)
                    # 3km 이하는 단순 이동, 초과는 환승 복잡도 높음으로 판단
                    if km <= 3.0:
                        simple += 1
                    else:
                        issues.append(IssueItem(
                            category="foreignness", issue_type="complex_transfer",
                            day=d.day, poi_index=i,
                            description=(
                                f"Day {d.day} {cur.name} → {nxt.name}: "
                                f"{km:.1f}km — 환승 복잡도 높음"
                            ),
                            severity="low",
                        ))
        return simple / total_transitions if total_transitions > 0 else 1.0

    def _score_cultural_friction(self, days: list[DayPlan], issues: list) -> float:
        """CF: 문화 설명이 필요한 POI에 설명이 제공되는 비율."""
        friction_pois = [
            (d.day, i, p)
            for d in days
            for i, p in enumerate(d.pois)
            if p.cultural_friction == 1
        ]
        if not friction_pois:
            return 1.0
        # notes에 설명이 있으면 설명 제공된 것으로 간주
        explained = sum(
            1 for _, _, p in friction_pois
            if len(p.notes) > 20  # 20자 이상이면 설명 있다고 봄
        )
        ratio = explained / len(friction_pois)
        for day_num, i, p in friction_pois:
            if len(p.notes) <= 20:
                issues.append(IssueItem(
                    category="foreignness", issue_type="cultural_friction_unexplained",
                    day=day_num, poi_index=i,
                    description=(
                        f"{p.name} — 문화 설명 필요: {p.cultural_friction_reason or '문화 규범 존재'}"
                    ),
                    severity="medium",
                ))
        return ratio

    def _score_constraint_compatibility(self, days: list[DayPlan], issues: list) -> float:
        """CC: 사용자 제약조건(식단 등) 위반 여부."""
        dietary = self.user_state.get("dietary", "")
        if not dietary or dietary.lower() in ["none", "no", "없음", "n/a"]:
            return 1.0

        total = sum(len(d.pois) for d in days)
        violations = 0

        is_vegetarian = "vegetarian" in dietary.lower()
        is_vegan = "vegan" in dietary.lower()

        for d in days:
            for i, p in enumerate(d.pois):
                if p.poi_type in ["restaurant", "market", "cafe"]:
                    if is_vegetarian or is_vegan:
                        if p.cash_only == 1:
                            violations += 1
                            issues.append(IssueItem(
                                category="foreignness", issue_type="dietary_constraint",
                                day=d.day, poi_index=i,
                                description=(
                                    f"{p.name} — {dietary} 제약과 충돌 가능 "
                                    "(채식 옵션 불명확)"
                                ),
                                severity="high",
                            ))

        if total == 0:
            return 1.0
        return 1.0 - violations / total


# ============================================================
# Levenshtein distance (POI 이름 리스트용)
# ============================================================

def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


# ============================================================
# Repair Agent
# ============================================================

class RepairAgent:
    """
    Critic이 탐지한 문제를 기반으로 itinerary를 자동 수정.
    Rule-based 탐지 + LLM 기반 대체 POI 선택.
    """

    def __init__(self, poi_db: POIDatabase, user_state: dict):
        self.poi_db = poi_db
        self.user_state = user_state

    def repair(self, itinerary: dict, critic_result: CriticResult) -> RepairResult:
        """문제 목록을 보고 itinerary 수정."""
        import copy
        fixed = copy.deepcopy(itinerary)
        actions: list[RepairAction] = []

        high_issues = [i for i in critic_result.issues if i.severity == "high"]
        medium_issues = [i for i in critic_result.issues if i.severity == "medium"]

        # 우선순위: high → medium 순으로 처리
        for issue in high_issues + medium_issues:
            action = self._fix_issue(fixed, issue)
            if action:
                actions.append(action)

        return RepairResult(
            itinerary=fixed,
            actions=actions,
            success=len(actions) > 0,
            message=f"{len(actions)}개 문제 수정 완료",
        )

    def _fix_issue(self, itinerary: dict, issue: IssueItem) -> RepairAction | None:
        """개별 문제 수정."""

        # ── 운영시간 충돌 → 동일 권역 대체 POI 교체 ──
        if issue.issue_type == "oh_conflict":
            return self._fix_oh_conflict(itinerary, issue)

        # ── 이동시간 과다 → 가장 먼 POI 제거 ──
        if issue.issue_type == "travel_too_far":
            return self._fix_travel_too_far(itinerary, issue)

        # ── 식사 누락 → 근처 식당/카페 삽입 ──
        if issue.issue_type == "meal_missing":
            return self._fix_meal_missing(itinerary, issue)

        # ── 식단 제약 위반 → vegetarian 식당으로 교체 ──
        if issue.issue_type == "dietary_constraint":
            return self._fix_dietary_constraint(itinerary, issue)

        # ── 일정 과밀 → activity 제거 ──
        if issue.issue_type == "daily_travel_excess":
            return self._fix_daily_excess(itinerary, issue)

        return None

    def _get_day_pois(self, itinerary: dict, day_num: int) -> list:
        for d in itinerary.get("days", []):
            if d.get("day") == day_num:
                return d.get("pois", [])
        return []

    def _fix_oh_conflict(self, itinerary: dict, issue: IssueItem) -> RepairAction | None:
        """운영시간 충돌: 동일 권역 동일 타입 대체 POI로 교체."""
        day_pois = self._get_day_pois(itinerary, issue.day)
        if issue.poi_index is None or issue.poi_index >= len(day_pois):
            return None

        target = day_pois[issue.poi_index]
        target_name = target.get("name", "")
        target_type = target.get("type", "tourist_spot")
        lat = target.get("lat")
        lng = target.get("lng")

        cluster = assign_cluster(float(lat), float(lng)) if lat and lng else "other"
        existing = [p.get("name", "") for p in day_pois]
        all_existing = [
            p.get("name", "")
            for d in itinerary.get("days", [])
            for p in d.get("pois", [])
        ]

        candidates = self.poi_db.get_by_type_and_cluster(
            target_type, cluster, all_existing
        )

        if not candidates:
            return None

        best = candidates[0]
        replacement = {
            "name": best.get("poi_name", ""),
            "type": best.get("poi_type", target_type),
            "address": best.get("address_en") or best.get("address_ko", ""),
            "lat": best.get("lat"),
            "lng": best.get("lng"),
            "stay_minutes": target.get("stay_minutes", 60),
            "notes": f"운영시간 충돌로 인한 대체 장소 (원래: {target_name})",
        }
        day_pois[issue.poi_index] = replacement

        return RepairAction(
            action_type="replace_poi",
            day=issue.day,
            poi_index=issue.poi_index,
            description=f"운영시간 충돌 → {target_name}을 {replacement['name']}으로 교체",
            before=target_name,
            after=replacement["name"],
        )

    def _fix_travel_too_far(self, itinerary: dict, issue: IssueItem) -> RepairAction | None:
        """이동시간 과다: 더 먼 POI 제거."""
        day_pois = self._get_day_pois(itinerary, issue.day)
        if issue.poi_index is None or issue.poi_index + 1 >= len(day_pois):
            return None

        # 두 POI 중 인접 POI와의 총 거리가 더 큰 것 제거
        idx = issue.poi_index + 1  # 다음 POI 제거
        removed = day_pois.pop(idx)
        removed_name = removed.get("name", "")

        return RepairAction(
            action_type="remove_poi",
            day=issue.day,
            poi_index=idx,
            description=f"이동시간 과다 → {removed_name} 제거",
            before=removed_name,
            after="(제거됨)",
        )

    def _fix_meal_missing(self, itinerary: dict, issue: IssueItem) -> RepairAction | None:
        """식사 누락: 해당 Day 중간 지점 근처 식당 삽입."""
        day_pois = self._get_day_pois(itinerary, issue.day)
        if not day_pois:
            return None

        # 중간 POI 좌표 기준으로 식당 검색
        mid_idx = len(day_pois) // 2
        mid_poi = day_pois[mid_idx]
        lat = mid_poi.get("lat")
        lng = mid_poi.get("lng")
        cluster = assign_cluster(float(lat), float(lng)) if lat and lng else "other"

        all_existing = [
            p.get("name", "")
            for d in itinerary.get("days", [])
            for p in d.get("pois", [])
        ]
        dietary = self.user_state.get("dietary")

        restaurants = self.poi_db.get_by_type_and_cluster(
            "restaurant", cluster, all_existing, dietary=dietary
        )
        if not restaurants:
            restaurants = self.poi_db.get_by_type_and_cluster(
                "cafe", cluster, all_existing
            )
        if not restaurants:
            return None

        best = restaurants[0]
        insert_poi = {
            "name": best.get("poi_name", ""),
            "type": best.get("poi_type", "restaurant"),
            "address": best.get("address_en") or best.get("address_ko", ""),
            "lat": best.get("lat"),
            "lng": best.get("lng"),
            "stay_minutes": 60,
            "notes": "식사 슬롯 자동 추가",
        }

        insert_idx = mid_idx + 1
        day_pois.insert(insert_idx, insert_poi)

        return RepairAction(
            action_type="insert_slot",
            day=issue.day,
            poi_index=insert_idx,
            description=f"식사 누락 → {insert_poi['name']} 삽입 (Day {issue.day} {mid_idx+1}번째 위치)",
            before="(없음)",
            after=insert_poi["name"],
        )

    def _fix_dietary_constraint(self, itinerary: dict, issue: IssueItem) -> RepairAction | None:
        """식단 제약: vegetarian-friendly 식당으로 교체."""
        day_pois = self._get_day_pois(itinerary, issue.day)
        if issue.poi_index is None or issue.poi_index >= len(day_pois):
            return None

        target = day_pois[issue.poi_index]
        target_name = target.get("name", "")
        lat = target.get("lat")
        lng = target.get("lng")
        cluster = assign_cluster(float(lat), float(lng)) if lat and lng else "other"
        all_existing = [
            p.get("name", "")
            for d in itinerary.get("days", [])
            for p in d.get("pois", [])
        ]

        candidates = self.poi_db.get_by_type_and_cluster(
            "restaurant", cluster, all_existing,
            dietary=self.user_state.get("dietary")
        )
        if not candidates:
            return None

        best = candidates[0]
        replacement = {
            "name": best.get("poi_name", ""),
            "type": "restaurant",
            "address": best.get("address_en") or best.get("address_ko", ""),
            "lat": best.get("lat"),
            "lng": best.get("lng"),
            "stay_minutes": target.get("stay_minutes", 60),
            "notes": f"식단 제약({self.user_state.get('dietary')}) 반영 대체 장소",
        }
        day_pois[issue.poi_index] = replacement

        return RepairAction(
            action_type="replace_poi",
            day=issue.day,
            poi_index=issue.poi_index,
            description=f"식단 제약 → {target_name}을 {replacement['name']}으로 교체",
            before=target_name,
            after=replacement["name"],
        )

    def _fix_daily_excess(self, itinerary: dict, issue: IssueItem) -> RepairAction | None:
        """일정 과밀/이동 과다: 가장 멀리 떨어진 POI 제거."""
        day_pois = self._get_day_pois(itinerary, issue.day)
        if len(day_pois) <= 2:
            return None

        # 중심점에서 가장 먼 POI 제거
        lats = [p.get("lat") for p in day_pois if p.get("lat")]
        lngs = [p.get("lng") for p in day_pois if p.get("lng")]
        if not lats:
            return None

        center_lat = sum(lats) / len(lats)
        center_lng = sum(lngs) / len(lngs)

        farthest_idx = max(
            range(len(day_pois)),
            key=lambda i: haversine_km(
                float(day_pois[i].get("lat") or center_lat),
                float(day_pois[i].get("lng") or center_lng),
                center_lat, center_lng,
            )
        )
        removed = day_pois.pop(farthest_idx)
        removed_name = removed.get("name", "")

        return RepairAction(
            action_type="remove_poi",
            day=issue.day,
            poi_index=farthest_idx,
            description=f"이동 과다 → 중심에서 가장 먼 {removed_name} 제거",
            before=removed_name,
            after="(제거됨)",
        )


# ============================================================
# Critic-Repair 반복 루프
# ============================================================

def run_critic_repair_loop(
    itinerary: dict,
    poi_db: POIDatabase,
    reference_courses: list[dict],
    user_state: dict,
    max_iterations: int = 3,
) -> tuple[dict, CriticResult, list[RepairResult]]:
    """
    Generator 출력 itinerary를 받아
    Critic → Repair → Critic → ... 반복.

    반환:
        (최종 itinerary, 최종 CriticResult, 각 iteration의 RepairResult 목록)
    """
    critic = CriticAgent(poi_db, reference_courses, user_state)
    repair = RepairAgent(poi_db, user_state)

    repair_history: list[RepairResult] = []
    current = itinerary

    for iteration in range(1, max_iterations + 1):
        result = critic.evaluate(current)

        print(f"\n[Critic Iteration {iteration}]")
        print(f"  Feasibility : {result.feasibility_score:.3f}")
        print(f"  ECS         : {result.ecs_score:.3f}")
        print(f"  Foreignness : {result.foreignness_score:.3f}")
        print(f"  Total       : {result.total_score:.3f}")
        print(f"  Issues      : {len(result.issues)}개")
        for iss in result.issues:
            print(f"    [{iss.severity.upper()}] {iss.description}")

        if result.passed:
            print(f"  → 기준 통과! 반복 종료.")
            return current, result, repair_history

        if not result.issues:
            print("  → 문제 없음. 반복 종료.")
            return current, result, repair_history

        print(f"  → Repair 수행...")
        repair_result = repair.repair(current, result)
        repair_history.append(repair_result)
        current = repair_result.itinerary

        print(f"  → {len(repair_result.actions)}개 수정:")
        for action in repair_result.actions:
            print(f"    [{action.action_type}] {action.description}")

    # 마지막 평가
    final_result = critic.evaluate(current)
    print(f"\n[최종 결과]")
    print(f"  Total Score : {final_result.total_score:.3f}")
    print(f"  Passed      : {final_result.passed}")

    return current, final_result, repair_history


# ============================================================
# LangGraph 노드 함수 (graph.py에 연결용)
# ============================================================

def make_critic_repair_node(base_dir: str | Path = "."):
    """
    LangGraph에 추가할 Critic-Repair 노드 생성.

    사용법 (graph.py):
        from critic_repair import make_critic_repair_node
        builder.add_node("critic_repair", make_critic_repair_node())
    """
    from langchain_core.messages import AIMessage
    from state import TravelState

    base_dir = Path(base_dir)

    # 참조 코스 로드
    course_data_path = base_dir / "course_data.json"
    if course_data_path.exists():
        with open(course_data_path, encoding="utf-8") as f:
            reference_courses = json.load(f)
    else:
        reference_courses = []

    poi_db = POIDatabase(base_dir)

    def critic_repair_node(state: TravelState) -> TravelState:
        itinerary = state.get("itinerary")
        if not itinerary:
            return {
                **state,
                "messages": [AIMessage(content="⚠️ 평가할 일정이 없습니다.")],
            }

        user_state = {
            "budget":  state.get("budget"),
            "dietary": state.get("dietary"),
            "duration": state.get("duration"),
            "location": state.get("location"),
        }

        try:
            fixed_itinerary, critic_result, repair_history = run_critic_repair_loop(
                itinerary=itinerary,
                poi_db=poi_db,
                reference_courses=reference_courses,
                user_state=user_state,
                max_iterations=3,
            )
        except Exception as e:
            return {
                **state,
                "messages": [AIMessage(content=f"⚠️ Critic-Repair 오류: {e}")],
            }

        # 결과 메시지 구성
        score_lines = (
            f"📊 **일정 품질 평가 결과**\n"
            f"- 실현 가능성 (Feasibility): {critic_result.feasibility_score:.0%}\n"
            f"- 경험 일관성 (ECS): {critic_result.ecs_score:.0%}\n"
            f"- 외국인 친화성 (Foreignness): {critic_result.foreignness_score:.0%}\n"
            f"- 종합 점수: {critic_result.total_score:.0%}\n"
        )

        if repair_history:
            total_actions = sum(len(r.actions) for r in repair_history)
            score_lines += f"\n✅ {total_actions}개 문제 자동 수정 완료"
        else:
            score_lines += "\n✅ 수정 사항 없음"

        if critic_result.issues:
            remaining = [i for i in critic_result.issues if i.severity == "high"]
            if remaining:
                score_lines += f"\n⚠️ 잔여 주의 사항 {len(remaining)}개:"
                for iss in remaining[:3]:
                    score_lines += f"\n  - {iss.description}"

        return {
            **state,
            "itinerary": fixed_itinerary,
            "messages": [AIMessage(content=score_lines)],
            "current_step": "done",
        }

    return critic_repair_node


# ============================================================
# 단독 실행 테스트
# ============================================================

if __name__ == "__main__":
    import sys

    # 테스트용 샘플 itinerary
    sample_itinerary = {
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
                        "address": "20 Hongik-ro, Mapo-gu, Seoul",
                        "lat": 37.5563,
                        "lng": 126.9227,
                        "stay_minutes": 90,
                        "notes": "홍대 거리 탐방",
                    },
                    {
                        "name": "Gyeongbokgung Palace (경복궁)",
                        "type": "history",
                        "address": "사직로 161, 종로구, 서울",
                        "lat": 37.5796,
                        "lng": 126.9770,
                        "stay_minutes": 120,
                        "notes": "조선 왕조 궁궐",
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
                        "address": "강남구, 서울",
                        "lat": 37.4979,
                        "lng": 127.0276,
                        "stay_minutes": 90,
                        "notes": "강남 거리",
                    },
                ],
            },
        ],
        "sources": [],
    }

    base = Path(__file__).parent
    poi_db = POIDatabase(base)

    course_data_path = base / "course_data.json"
    if course_data_path.exists():
        with open(course_data_path, encoding="utf-8") as f:
            ref_courses = json.load(f)
    else:
        ref_courses = []

    user_state = {
        "budget": "$500",
        "dietary": "none",
        "duration": "2 days",
        "location": "Hongdae",
    }

    fixed, critic_result, repairs = run_critic_repair_loop(
        itinerary=sample_itinerary,
        poi_db=poi_db,
        reference_courses=ref_courses,
        user_state=user_state,
        max_iterations=3,
    )

    print("\n====== 최종 일정 ======")
    for day in fixed.get("days", []):
        print(f"\nDay {day['day']} — {day['theme']}")
        for p in day.get("pois", []):
            print(f"  {p['name']} ({p['type']}, {p['stay_minutes']}min)")