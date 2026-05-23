import dspy
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import TravelState
from planner import make_retrieve_node, plan_node
from llm import set_api_key, lm_context

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


# DSPy predictors — Predict() doesn't bind to an LM at construction time,
# so we can build them once and let lm_context() supply the LM per call.
_extractor: dspy.Predict | None = None
_classifier: dspy.Predict | None = None

def get_extractor() -> dspy.Predict:
    global _extractor
    if _extractor is None:
        _extractor = dspy.Predict(TripDetails)
    return _extractor

def get_classifier() -> dspy.Predict:
    global _classifier
    if _classifier is None:
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
        with lm_context():
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
        with lm_context():
            result = get_classifier()(user_message=last_human)
        intent = result.intent.strip().upper()
    except Exception as e:
        return {**state, "messages": [AIMessage(content=f"Error occurred. Please try again. ({e})")]}

    if intent == "CONFIRM":
        lines = "\n".join(f"{FIELD_LABELS[f]}: {state.get(f)}" for f in ALL_FIELDS)
        return {
            **state,
            "confirmed": True,
            "current_step": "retrieving",
            "messages": [AIMessage(
                content=f"🎉 Perfect! Your trip is confirmed.\n\n{lines}\n\n"
                        "🔍 Searching course catalogue and drafting your itinerary..."
            )]
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
    # Finished trip planning — nothing more to do.
    if state.get("itinerary"):
        return END

    step = state.get("current_step", "start")

    # Resume mid-pipeline if the graph was interrupted.
    if step == "retrieving":
        return "retrieve"
    if step == "planning":
        return "plan"

    if step == "confirm":
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            return "handle_confirm"
        return "confirm"

    return "collect"


def _after_collect(state: TravelState) -> str:
    return "confirm" if state.get("current_step") == "confirm" else END


def _after_handle_confirm(state: TravelState) -> str:
    # On CONFIRM, handle_confirm sets current_step="retrieving" → continue.
    if state.get("current_step") == "retrieving":
        return "retrieve"
    return END


def _after_retrieve(state: TravelState) -> str:
    return "plan" if state.get("current_step") == "planning" else END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(api_key: str):
    global _extractor, _classifier
    # Register the key with the shared LM module. set_api_key() invalidates
    # the cached dspy.LM if the key changed, so a new key takes effect.
    set_api_key(api_key)
    # Predict() instances don't bind to an LM at construction time, but we
    # still reset them so subsequent runs build fresh ones if anything
    # signature-related changed on disk.
    _extractor = None
    _classifier = None

    builder = StateGraph(TravelState)

    builder.add_node("collect", collect_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("handle_confirm", handle_confirm_node)
    builder.add_node("retrieve", make_retrieve_node(api_key))
    builder.add_node("plan", plan_node)

    builder.set_conditional_entry_point(route_entry, {
        "collect": "collect",
        "confirm": "confirm",
        "handle_confirm": "handle_confirm",
        "retrieve": "retrieve",
        "plan": "plan",
        END: END,
    })

    builder.add_conditional_edges(
        "collect",
        _after_collect,
        {"confirm": "confirm", END: END},
    )

    builder.add_edge("confirm", END)

    builder.add_conditional_edges(
        "handle_confirm",
        _after_handle_confirm,
        {"retrieve": "retrieve", END: END},
    )

    builder.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"plan": "plan", END: END},
    )

    builder.add_edge("plan", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
