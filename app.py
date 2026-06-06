import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph import build_graph
from state import TravelState

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Travel Planner 🗺️", page_icon="✈️", layout="centered")
st.title("✈️ AI Travel Planner")
st.caption("Enter your travel details and get a personalized itinerary.")

# Session initialization defaults
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "travel-session-1"
if "initialized" not in st.session_state:
    st.session_state.initialized = False

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------------------------
# Sidebar (always rendered first so the key input is always visible)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔑 Gemini API Key")
    api_key_input = st.text_input(
        "Enter your API key",
        type="password",
        value=st.session_state.api_key or "",
        help="Get your key at https://aistudio.google.com/app/apikey",
        placeholder="AIza...",
    )
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        # Reset graph and conversation so new key takes effect
        for key in ["graph", "initialized"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.subheader("📋 Collected Information")

    # Placeholder — filled in after current_state is available
    collected_placeholder = st.empty()

    st.divider()
    if st.button("🔄 Restart"):
        api_key = st.session_state.api_key
        st.session_state.clear()
        st.session_state.api_key = api_key
        st.rerun()

    if st.button("🔑 Change API Key"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Require API key before running the graph
# ---------------------------------------------------------------------------
if not st.session_state.api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to get started.")
    st.stop()

# Initialize graph once API key is available
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(st.session_state.api_key)

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def get_current_state() -> TravelState:
    snapshot = st.session_state.graph.get_state(config)
    if snapshot and snapshot.values:
        return snapshot.values
    return {
        "duration": None, "location": None, "budget": None,
        "dietary": None, "purpose": None,
        "current_step": "start", "confirmed": False, "messages": [],
        "retrieved_courses": [], "itinerary": None,
    }


def run_graph(user_input: str = None) -> TravelState:
    state = get_current_state()
    messages = list(state.get("messages", []))
    if user_input:
        messages = messages + [HumanMessage(content=user_input)]
    updated_state = {**state, "messages": messages}
    return st.session_state.graph.invoke(updated_state, config)


# Initial greeting
if not st.session_state.initialized:
    run_graph()
    st.session_state.initialized = True

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
current_state = get_current_state()
messages = current_state.get("messages", [])

for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Fill in collected info in the sidebar placeholder
with collected_placeholder.container():
    fields = {
        "📅 Trip Duration": current_state.get("duration"),
        "📍 Destination": current_state.get("location"),
        "💰 Budget": current_state.get("budget"),
        "🥗 Dietary Restrictions": current_state.get("dietary"),
        "🎯 Travel Purpose": current_state.get("purpose"),
    }
    for label, value in fields.items():
        if value:
            st.success(f"{label}: {value}")
        else:
            st.warning(f"{label}: Not provided")

# ---------------------------------------------------------------------------
# Itinerary rendering
# ---------------------------------------------------------------------------

def render_itinerary(itinerary: dict) -> None:
    st.divider()
    st.header("🗺️ Your Itinerary")

    summary = itinerary.get("summary")
    if summary:
        st.write(summary)

    for day in itinerary.get("days", []):
        day_num = day.get("day", "?")
        theme = day.get("theme", "")
        cost = day.get("estimated_cost", "")
        header = f"Day {day_num}"
        if theme:
            header += f" — {theme}"
        with st.expander(header, expanded=True):
            if cost:
                st.caption(f"💰 Estimated cost: {cost}")
            for i, poi in enumerate(day.get("pois", []), start=1):
                name = poi.get("name", "")
                ptype = poi.get("type", "")
                addr = poi.get("address", "")
                stay = poi.get("stay_minutes")
                notes = poi.get("notes", "")
                line = f"**{i}. {name}**"
                if ptype:
                    line += f"  _({ptype})_"
                st.markdown(line)
                meta_bits = []
                if addr:
                    meta_bits.append(f"📍 {addr}")
                if stay:
                    meta_bits.append(f"⏱️ {stay} min")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))
                if notes:
                    st.write(notes)

    # Sources block — URLs of the courses the planner actually drew from
    # (validated against retrieved_courses; falls back to all candidates
    # if the LLM omitted citations).
    sources = itinerary.get("sources") or []
    if sources:
        st.divider()
        st.subheader("🔗 Sources")
        st.caption("Courses referenced for this itinerary:")
        for s in sources:
            title = s.get("course_title") or s.get("course_id") or "Untitled course"
            src = s.get("source") or ""
            url = s.get("source_url")
            if url:
                label = f"- [{title}]({url})"
                if src:
                    label += f" — _{src}_"
                st.markdown(label)
            else:
                st.markdown(f"- {title}")

    st.download_button(
        "⬇️ Download itinerary JSON",
        data=json.dumps(itinerary, ensure_ascii=False, indent=2),
        file_name="itinerary.json",
        mime="application/json",
    )


itinerary = current_state.get("itinerary")
if itinerary:
    render_itinerary(itinerary)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if itinerary:
    st.success("🎉 Your travel plan is ready! Hit Restart in the sidebar to plan another trip.")
    st.chat_input("Completed.", disabled=True)
else:
    if user_msg := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.write(user_msg)
        with st.spinner("Thinking..."):
            run_graph(user_msg)
        st.rerun()
