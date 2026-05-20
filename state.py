from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated


class TravelState(TypedDict):
    duration: Optional[str]         # 여행 기간
    location: Optional[str]         # 숙박 지역
    budget: Optional[str]           # 예산
    dietary: Optional[str]          # 식단 제약
    purpose: Optional[str]          # 가는 이유
    current_step: str               # 현재 수집 단계
    confirmed: bool                 # 최종 컨펌 여부
    messages: Annotated[list, add_messages]  # 대화 히스토리 (reducer 적용)