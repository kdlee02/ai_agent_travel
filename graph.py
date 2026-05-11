import dspy
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import TravelState

# ---------------------------------------------------------------------------
# Singleton LLM setup
# ---------------------------------------------------------------------------

_lm: dspy.LM | None = None
_api_key: str | None = None

def get_lm() -> dspy.LM:
    global _lm
    if _lm is None:
        if not _api_key:
            raise ValueError("API key not set. Call build_graph(api_key) first.")
        _lm = dspy.LM(
            model="gemini/gemini-2.5-flash",
            api_key=_api_key,
            temperature=0.7,
        )
        dspy.configure(lm=_lm)
    return _lm


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------

class TripDetails(dspy.Signature):
    """Extract trip details from user input. Use 'MISSING' if not mentioned."""
    text: str = dspy.InputField()
    duration: str = dspy.OutputField(desc="Trip length, e.g. '3 days', '1 week'. 'MISSING' if not mentioned.")
    location: str = dspy.OutputField(desc="Destination or accommodation area. 'MISSING' if not mentioned.")
    budget: str = dspy.OutputField(desc="Total budget, e.g. '$500'. 'MISSING' if not mentioned.")
    dietary: str = dspy.OutputField(desc="Dietary restrictions or preferences. 'MISSING' if not mentioned.")
    purpose: str = dspy.OutputField(desc="Purpose of the trip, e.g. family trip, leisure. 'MISSING' if not mentioned.")


class ConfirmIntent(dspy.Signature):
    """Classify whether the user is confirming or editing."""
    user_message: str = dspy.InputField()
    intent: str = dspy.OutputField(
        desc="Return 'CONFIRM' or exactly one of: duration, location, budget, dietary, purpose"
    )


# DSPy predictors
_extractor: dspy.Predict | None = None
_classifier: dspy.Predict | None = None

def get_extractor() -> dspy.Predict:
    global _extractor
    if _extractor is None:
        get_lm()
        _extractor = dspy.Predict(TripDetails)
    return _extractor

def get_classifier() -> dspy.Predict:
    global _classifier
    if _classifier is None:
        get_lm()
        _classifier = dspy.Predict(ConfirmIntent)
    return _classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "duration": "📅 Trip Duration",
    "location": "📍 Destination",
    "budget": "💰 Budget",
    "dietary": "🥗 Dietary Restrictions",
    "purpose": "🎯 Travel Purpose",
}

FIELD_QUESTIONS = {
    "duration": "How long is your trip? (e.g. 3 days, 1 week)",
    "location": "Where are you planning to stay or visit? (e.g. Tokyo, Paris)",
    "budget": "What is your total budget? (e.g. $500, $1000)",
    "dietary": "Do you have any dietary restrictions? (e.g. vegetarian, none)",
    "purpose": "What is the purpose of your trip? (e.g. vacation, family trip, birthday)",
}

ALL_FIELDS = list(FIELD_QUESTIONS.keys())


def get_missing_fields(state: TravelState) -> list[str]:
    return [f for f in ALL_FIELDS if not state.get(f)]


def build_summary(state: TravelState) -> str:
    lines = "\n".join(f"{FIELD_LABELS[f]}: {state.get(f)}" for f in ALL_FIELDS)
    return (
        f"✅ All travel details collected! Please review:\n\n{lines}\n\n"
        "If everything looks good, type **'confirm'**.\n"
        "If you want to change something, just tell me (e.g. 'change budget', 'edit duration')."
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def collect_node(state: TravelState) -> TravelState:
    messages = state.get("messages", [])

    # First turn
    if state.get("current_step") == "start":
        greeting = (
            "Hi! I’ll help you plan your trip 😊\n\n"
            "Please provide the following information in one message:\n"
            "- Trip duration (e.g. 3 days)\n"
            "- Destination (e.g. Tokyo)\n"
            "- Total budget (e.g. $500)\n"
            "- Dietary restrictions (e.g. vegetarian, none)\n"
            "- Travel purpose (e.g. vacation, family trip)"
        )
        return {**state, "current_step": "collecting", "messages": [AIMessage(content=greeting)]}

    # Extract
    last_human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human:
        return state

    updates = {}
    try:
        result = get_extractor()(text=last_human)
        for field in ALL_FIELDS:
            value = getattr(result, field, "").strip()
            if value and value.upper() != "MISSING" and not state.get(field):
                updates[field] = value
    except Exception as e:
        return {**state, "messages": [AIMessage(content=f"Error extracting info. Please try again. ({e})")]}

    merged = {**state, **updates}
    missing = get_missing_fields(merged)

    if missing:
        questions = "\n".join(f"- {FIELD_QUESTIONS[f]}" for f in missing)
        return {
            **merged,
            "current_step": "collecting",
            "messages": [AIMessage(content=f"Almost done! I still need:\n\n{questions}")]
        }

    return {**merged, "current_step": "confirm"}


def confirm_node(state: TravelState) -> TravelState:
    return {**state, "current_step": "confirm", "messages": [AIMessage(content=build_summary(state))]}


def handle_confirm_node(state: TravelState) -> TravelState:
    messages = state.get("messages", [])
    last_human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human:
        return state

    try:
        result = get_classifier()(user_message=last_human)
        intent = result.intent.strip().upper()
    except Exception as e:
        return {**state, "messages": [AIMessage(content=f"Error occurred. Please try again. ({e})")]}

    if intent == "CONFIRM":
        lines = "\n".join(f"{FIELD_LABELS[f]}: {state.get(f)}" for f in ALL_FIELDS)
        return {
            **state,
            "confirmed": True,
            "messages": [AIMessage(content=f"🎉 Perfect! Your trip is confirmed.\n\n{lines}\n\nI’ll use this to plan your itinerary! ✈️")]
        }

    if intent.lower() in ALL_FIELDS:
        field = intent.lower()
        return {
            **state,
            field: None,
            "current_step": "collecting",
            "messages": [AIMessage(content=f"Got it! {FIELD_QUESTIONS[field]}")]
        }

    return {**state, "messages": [AIMessage(content="I didn’t understand. Type 'confirm' or tell me what to change.")]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_entry(state: TravelState) -> str:
    if state.get("confirmed"):
        return END
    step = state.get("current_step", "start")
    if step == "confirm":
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            return "handle_confirm"
        return "confirm"
    return "collect"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(api_key: str):
    global _lm, _extractor, _classifier, _api_key
    # Reset singletons so a new key takes effect
    _lm = None
    _extractor = None
    _classifier = None
    _api_key = api_key

    builder = StateGraph(TravelState)

    builder.add_node("collect", collect_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("handle_confirm", handle_confirm_node)

    builder.set_conditional_entry_point(route_entry, {
        "collect": "collect",
        "confirm": "confirm",
        "handle_confirm": "handle_confirm",
        END: END,
    })

    builder.add_conditional_edges(
        "collect",
        lambda s: "confirm" if s.get("current_step") == "confirm" else END,
        {"confirm": "confirm", END: END},
    )

    builder.add_edge("confirm", END)
    builder.add_edge("handle_confirm", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
