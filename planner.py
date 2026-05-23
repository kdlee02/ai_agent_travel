"""Itinerary planning nodes for the LangGraph.

`retrieve_node` runs FAISS retrieval over course_data.json and stashes
the top courses in state. `plan_node` calls a DSPy signature that turns
those courses + the user's confirmed fields into a structured day-by-day
itinerary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import dspy
from langchain_core.messages import AIMessage

from llm import lm_context
from rag import build_query, retrieve_courses
from state import TravelState


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------

class ItineraryPlanner(dspy.Signature):
    """Generate a personalized Seoul travel itinerary.

    You are given the user's trip details and a shortlist of candidate
    courses (each with a sequence of POIs). Build a realistic day-by-day
    plan that:
      - Respects the requested trip duration (one entry per day).
      - Stays within the budget when possible (note it in estimated_cost).
      - Honors dietary restrictions when picking restaurants/cafes.
      - Groups POIs geographically so a single day doesn't crisscross the city.
      - You MAY mix POIs from different candidate courses on the same day.

    Return ONLY valid JSON, no markdown fences, matching this schema:
    {
      "summary": "<one-paragraph overview of the trip>",
      "days": [
        {
          "day": 1,
          "theme": "<short theme for the day>",
          "pois": [
            {
              "name": "<POI name>",
              "type": "<poi_type>",
              "address": "<address_en or address_ko>",
              "lat": <number>,
              "lng": <number>,
              "stay_minutes": <integer>,
              "notes": "<why this POI fits, dietary/budget notes if relevant>"
            }
          ],
          "estimated_cost": "<rough cost for the day in the user's currency>"
        }
      ],
      "sources": [
        {
          "course_id": "<course_id of a course you actually drew POIs from>",
          "course_title": "<that course's title>",
          "source": "<Visit Seoul or Visit Korea>",
          "source_url": "<that course's source_url>"
        }
      ]
    }

    Only include a course in "sources" if at least one of its POIs appears
    in your "days". Use the exact course_id and source_url given in the
    candidate_courses input.
    """
    duration: str = dspy.InputField(desc="Trip length, e.g. '3 days'.")
    location: str = dspy.InputField(desc="Destination or accommodation area.")
    budget: str = dspy.InputField(desc="Total trip budget.")
    dietary: str = dspy.InputField(desc="Dietary restrictions or preferences.")
    purpose: str = dspy.InputField(desc="Purpose of the trip.")
    candidate_courses: str = dspy.InputField(
        desc="Shortlist of candidate courses with POIs, as compact text."
    )
    itinerary_json: str = dspy.OutputField(
        desc="Itinerary as a JSON object matching the schema in the docstring."
    )


class FixJSON(dspy.Signature):
    """Repair a JSON document that failed to parse.

    Output ONLY the corrected JSON object. No prose, no markdown fences,
    no explanation. Preserve all fields and values from the broken input;
    only fix the syntax (escape quotes, remove trailing commas, replace
    smart quotes with straight quotes, etc.).
    """
    broken_json: str = dspy.InputField(desc="The malformed JSON text.")
    error_message: str = dspy.InputField(desc="The parser error reported.")
    fixed_json: str = dspy.OutputField(desc="Strictly valid JSON only.")


_planner: dspy.Predict | None = None
_fixer: dspy.Predict | None = None


def get_planner() -> dspy.Predict:
    """Lazy singleton — relies on dspy.configure() already being called."""
    global _planner
    if _planner is None:
        _planner = dspy.Predict(ItineraryPlanner)
    return _planner


def get_fixer() -> dspy.Predict:
    global _fixer
    if _fixer is None:
        _fixer = dspy.Predict(FixJSON)
    return _fixer


# ---------------------------------------------------------------------------
# Course compaction for the prompt
# ---------------------------------------------------------------------------

def _format_courses_for_prompt(courses: list[dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(courses, start=1):
        title = c.get("course_title", "")
        course_id = c.get("course_id", "")
        source = c.get("source", "")
        source_url = c.get("source_url", "")
        themes = ", ".join(c.get("theme_category", []) or [])
        poi_lines = []
        for p in c.get("sequence", []) or []:
            poi_lines.append(
                f"    - {p.get('poi_name', '')} "
                f"[{p.get('poi_type', '')}] "
                f"addr={p.get('address_en') or p.get('address_ko', '')} "
                f"lat={p.get('lat')} lng={p.get('lng')} "
                f"stay={p.get('estimated_stay_time')}min"
            )
        blocks.append(
            f"Course {i}: {title}\n"
            f"  course_id : {course_id}\n"
            f"  source    : {source}\n"
            f"  source_url: {source_url}\n"
            f"  Themes    : {themes}\n"
            f"  POIs:\n" + "\n".join(poi_lines)
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# JSON parsing (tolerates ```json fences, smart quotes, trailing commas;
# falls back to an LLM-driven repair pass if all else fails)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _isolate_json_object(text: str) -> str:
    """Strip fences and any leading/trailing prose around the outer {...}."""
    text = _FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def _simple_repair(text: str) -> str:
    """Cheap repairs for the failure modes Gemini hits most often."""
    # Smart / curly quotes -> ASCII
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
    )
    # Trailing commas before } or ]
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _parse_itinerary_json(raw: str, *, use_llm_fallback: bool = True) -> dict[str, Any]:
    """Parse the planner's JSON output, repairing it if needed."""
    isolated = _isolate_json_object(raw)

    # 1) Strict parse
    try:
        return json.loads(isolated)
    except json.JSONDecodeError as first_err:
        pass

    # 2) Cheap local repairs
    repaired = _simple_repair(isolated)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as second_err:
        pass

    # 3) Ask the LLM to repair it
    if use_llm_fallback:
        try:
            with lm_context():
                fixed = get_fixer()(
                    broken_json=isolated[:8000],   # cap to keep prompt bounded
                    error_message=str(second_err),
                ).fixed_json
            return json.loads(_isolate_json_object(fixed))
        except Exception:
            pass

    # 4) Surface the original error after dumping raw output for debugging
    _dump_debug(raw)
    raise second_err


def _dump_debug(raw: str) -> None:
    """Write the offending output next to this file for inspection."""
    try:
        dbg_path = Path(__file__).resolve().parent / "planner_last_failed.txt"
        dbg_path.write_text(raw, encoding="utf-8")
        print(f"[planner] wrote failing output to {dbg_path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sources hygiene — validate/repair the LLM's source citations against
# the actual retrieved courses, falling back to all candidates if needed.
# ---------------------------------------------------------------------------

def _normalize_sources(
    itinerary: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace any hallucinated URLs with values from retrieved_courses.

    Also fills in ``sources`` from ``retrieved_courses`` if the planner
    omitted it entirely.
    """
    by_id = {c.get("course_id"): c for c in retrieved if c.get("course_id")}
    by_url = {c.get("source_url"): c for c in retrieved if c.get("source_url")}

    raw_sources = itinerary.get("sources") or []
    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for s in raw_sources:
        if not isinstance(s, dict):
            continue
        # Prefer matching by course_id, fall back to URL match.
        match = by_id.get(s.get("course_id")) or by_url.get(s.get("source_url"))
        if not match:
            continue  # drop hallucinated entries
        cid = match.get("course_id")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        cleaned.append({
            "course_id": cid,
            "course_title": match.get("course_title", ""),
            "source": match.get("source", ""),
            "source_url": match.get("source_url", ""),
        })

    # Fallback: if nothing survived, surface all retrieved candidates so
    # the user at least sees where the data came from.
    if not cleaned and retrieved:
        cleaned = [
            {
                "course_id": c.get("course_id"),
                "course_title": c.get("course_title", ""),
                "source": c.get("source", ""),
                "source_url": c.get("source_url", ""),
            }
            for c in retrieved
            if c.get("source_url")
        ]

    itinerary["sources"] = cleaned
    return itinerary


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def make_retrieve_node(api_key: str):
    """Bind the API key into a node closure (graph nodes take only state)."""

    def retrieve_node(state: TravelState) -> TravelState:
        query = build_query(
            purpose=state.get("purpose"),
            dietary=state.get("dietary"),
            location=state.get("location"),
            duration=state.get("duration"),
        )
        try:
            courses = retrieve_courses(api_key=api_key, query=query, k=2)
        except Exception as e:
            return {
                **state,
                "current_step": "confirm",
                "messages": [AIMessage(content=f"⚠️ Failed to retrieve courses: {e}")],
            }

        return {
            **state,
            "retrieved_courses": courses,
            "current_step": "planning",
        }

    return retrieve_node


def plan_node(state: TravelState) -> TravelState:
    courses = state.get("retrieved_courses") or []
    if not courses:
        return {
            **state,
            "current_step": "done",
            "messages": [AIMessage(content="⚠️ No candidate courses found. Try different details.")],
        }

    try:
        with lm_context():
            result = get_planner()(
                duration=state.get("duration") or "",
                location=state.get("location") or "",
                budget=state.get("budget") or "",
                dietary=state.get("dietary") or "none",
                purpose=state.get("purpose") or "",
                candidate_courses=_format_courses_for_prompt(courses),
            )
        itinerary = _parse_itinerary_json(result.itinerary_json)
        itinerary = _normalize_sources(itinerary, courses)
    except json.JSONDecodeError as e:
        return {
            **state,
            "current_step": "done",
            "messages": [AIMessage(content=f"⚠️ Planner returned invalid JSON: {e}")],
        }
    except Exception as e:
        return {
            **state,
            "current_step": "done",
            "messages": [AIMessage(content=f"⚠️ Planning failed: {e}")],
        }

    summary = itinerary.get("summary", "")
    day_count = len(itinerary.get("days", []))
    ack = (
        f"✅ Your {day_count}-day itinerary is ready!\n\n"
        f"{summary}\n\n"
        "See the full plan below."
    )

    return {
        **state,
        "itinerary": itinerary,
        "current_step": "done",
        "messages": [AIMessage(content=ack)],
    }
