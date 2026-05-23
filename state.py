from typing import Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated


class TravelState(TypedDict, total=False):
    duration: Optional[str]         # 여행 기간
    location: Optional[str]         # 숙박 지역
    budget: Optional[str]           # 예산
    dietary: Optional[str]          # 식단 제약
    purpose: Optional[str]          # 가는 이유
    current_step: str               # 현재 수집 단계: start | collecting | confirm | retrieving | planning | done
    confirmed: bool                 # 최종 컨펌 여부
    messages: Annotated[list, add_messages]  # 대화 히스토리 (reducer 적용)

    # RAG + planning
    retrieved_courses: list[dict[str, Any]]   # FAISS 검색 결과 코스 리스트
    itinerary: Optional[dict[str, Any]]       # 최종 일정 (구조화된 JSON)
