import os
import re
import json
import time
import hashlib
import pandas as pd
from tavily import TavilyClient
import google.generativeai as genai


# =========================
# 1. 설정값
# =========================

INPUT_DIR = "input_urls"
OUTPUT_DIR = "output"

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY",
    ""
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

GEMINI_RPM = 15
GEMINI_MIN_INTERVAL = 60.0 / GEMINI_RPM  # 4 seconds between calls

_last_gemini_call = 0.0


def gemini_generate(prompt):
    """Wraps gemini_model.generate_content with RPM throttling."""
    global _last_gemini_call
    elapsed = time.time() - _last_gemini_call
    wait = GEMINI_MIN_INTERVAL - elapsed
    if wait > 0:
        time.sleep(wait)
    response = gemini_model.generate_content(prompt)
    _last_gemini_call = time.time()
    return response


os.makedirs(OUTPUT_DIR, exist_ok=True)


SOURCE_CONFIG = {
    "visitseoul_urls.txt": {
        "source": "Visit Seoul",
        "course_prefix": "VS_WALK",
        "course_type": "walking"
    },
    "visitkorea_urls.txt": {
        "source": "Visit Korea",
        "course_prefix": "VK_THEME",
        "course_type": "theme"
    },
    "getyourguide_urls.txt": {
        "source": "GetYourGuide",
        "course_prefix": "GYG",
        "course_type": "tour"
    }
}

VALID_THEMES = [
    "K-POP", "History", "Culture", "Nature", "Shopping", "Food",
    "Nightlife", "Family", "Healing", "Couple", "Walking",
    "Traditional", "Modern Seoul"
]


# =========================
# 2. 유틸
# =========================

def clean_text(text):
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def read_url_files():
    url_items = []

    for filename, config in SOURCE_CONFIG.items():
        path = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        for idx, url in enumerate(urls, start=1):
            course_id = f"{config['course_prefix']}_{idx:03d}"
            url_items.append({
                "course_id": course_id,
                "source": config["source"],
                "source_url": url,
                "course_type": config["course_type"]
            })

    return url_items


# =========================
# 3. Tavily Extract
# =========================

def extract_page_content(url):
    response = tavily_client.extract(urls=[url])

    results = response.get("results", [])
    if not results:
        raise ValueError(f"Tavily extract returned no results for {url}")

    raw_content = results[0].get("raw_content", "")
    if not raw_content:
        raise ValueError(f"Tavily extract returned empty content for {url}")

    return raw_content


# =========================
# 4. Gemini: 파싱 + 테마 분류
# =========================

def extract_route_section(page_text):
    """
    For visitseoul walking tour pages, the Route field looks like:
        Route
        :   Stop 1
        :   Stop 2
        ...
    Extract and return just those lines as a clean list, or None if not found.
    """
    match = re.search(r"Route\s*\n((?::[ \t]+.+\n?)+)", page_text)
    if not match:
        return None
    stops = re.findall(r":[ \t]+(.+)", match.group(1))
    return [s.strip() for s in stops if s.strip()]


def parse_course_with_gemini(page_text, source_url):
    """
    Single Gemini call that extracts course structure from raw page text.

    Handles three formats:
    - Single-day / single-route: one set of POIs
    - Walking tour with explicit Route field: extracts stops from Route in order
    - Multi-day itinerary (Day 1 / Day 2 ...): returns one entry per day

    Returns a list of one or more dicts:
        [
            {
                "course_title": str,       # appended with " - Day N" for multi-day
                "day_num": int or None,
                "pois": [{"name": str, "poi_type": str, "estimated_stay_time": int or None}, ...],
                "theme_category": [str, ...]
            },
            ...
        ]
    """
    themes_str = ", ".join(VALID_THEMES)
    truncated = page_text[:8000]

    # Pre-extract Route section for walking tour pages so Gemini can't miss it
    route_stops = extract_route_section(page_text)
    route_hint = ""
    if route_stops:
        route_hint = f"\n\nNOTE: This page contains a Route field. The stops in order are:\n" + \
                     "\n".join(f"  {i+1}. {s}" for i, s in enumerate(route_stops)) + \
                     "\nFor TYPE B, use EXACTLY these stops in this order."

    prompt = f"""You are a Seoul tourism data extractor. Analyze the following webpage content from a Seoul tour/course page.

The page may be one of three types:
- TYPE A — Single-day course or themed tour: one set of POIs with no day breakdown
- TYPE B — Walking tour with an explicit "Route" field listing stops separated by ":   " (colon + spaces)
- TYPE C — Multi-day itinerary (e.g. "Day 1", "DAY2", "2일차") with different POIs per day

Detect the type and return a JSON object with exactly these top-level fields:

1. "course_title": The main title of the tour or course (string)
2. "type": One of "single", "walking", "multiday"
3. "theme_category": All relevant themes from this list ONLY: {themes_str}
   - Pick all that genuinely apply based on the course content
   - Be critical — only include themes that clearly match
   - If nothing fits, use ["general"]
4. "days": An array of day objects. Rules by type:
   - TYPE A (single): one object with {{"day": 1, "pois": [...]}}
   - TYPE B (walking): one object with {{"day": 1, "pois": [...]}}
     CRITICAL: Extract stops ONLY from the "Route" field. The Route field looks like:
       Route
       :   Stop Name 1
       :   Stop Name 2
       :   Stop Name 3
     Copy EVERY line after "Route" that starts with ":   " until a different field label appears (e.g. "Length of tour", "Meeting Place"). Do NOT use the "Main Tourist Attractions" section — that is a subset with descriptions, not the full route.
   - TYPE C (multiday): one object per day, e.g. [{{"day": 1, "pois": [...]}}, {{"day": 2, "pois": [...]}}]

Each entry in a "pois" array must be an object with:
- "name": the place name with its Korean name in parentheses (string).
  Format: "English Name (한국어 이름)" — e.g. "Gyeongbokgung Palace (경복궁)".
  If the page already shows the Korean name alongside the English name, use it exactly.
  If only an English name is present, generate the correct Korean name yourself.
  If only a Korean name is present, keep it as-is without adding English.
  Never omit the Korean name — always include it in parentheses.
- "poi_type": one of: museum, history, shopping, park, kpop_landmark, cafe, restaurant, street, tourist_spot
  Use the page content and your knowledge to determine the most accurate type.
- "estimated_stay_time": visit duration in minutes (integer) IF AND ONLY IF the page explicitly
  states a duration for this specific stop (e.g. "30 min", "about 1 hour", "1시간 소요").
  If no duration is mentioned on the page for this stop, use null.

For all pois arrays:
- Include only actual place names (landmarks, parks, palaces, markets, cafes, museums, etc.)
- Exclude descriptions, region labels, navigation text, image captions, and website boilerplate
- Preserve the original visit order within each day

Return ONLY valid JSON. No explanation, no markdown fences.

Example A (single-day):
{{"course_title": "RM's Pick: Seoul Art Tour", "type": "single", "theme_category": ["Culture", "Modern Seoul"], "days": [{{"day": 1, "pois": [{{"name": "Seoul Museum of Art (서울시립미술관)", "poi_type": "museum", "estimated_stay_time": 60}}, {{"name": "Leeum Museum (리움미술관)", "poi_type": "museum", "estimated_stay_time": null}}, {{"name": "National Museum of Korea (국립중앙박물관)", "poi_type": "museum", "estimated_stay_time": null}}]}}]}}

Example B (walking tour with Route field):
Given this Route section:
  Route
  :   Exit 3 of Hyehwa Station
  :   Daehan Hospital
  :   Hamchunwon
  :   Marronnier Park
  :   ARKO Art Center
Output: {{"course_title": "Daehak-ro Buildings Course", "type": "walking", "theme_category": ["History", "Walking"], "days": [{{"day": 1, "pois": [{{"name": "Exit 3 of Hyehwa Station (혜화역 3번 출구)", "poi_type": "tourist_spot", "estimated_stay_time": null}}, {{"name": "Daehan Hospital (대한의원)", "poi_type": "history", "estimated_stay_time": null}}, {{"name": "Hamchunwon (함춘원)", "poi_type": "park", "estimated_stay_time": null}}, {{"name": "Marronnier Park (마로니에공원)", "poi_type": "park", "estimated_stay_time": null}}, {{"name": "ARKO Art Center (아르코미술관)", "poi_type": "museum", "estimated_stay_time": null}}]}}]}}

Example C (multi-day):
{{"course_title": "3-Day Family Adventure", "type": "multiday", "theme_category": ["Family", "Nature"], "days": [{{"day": 1, "pois": [{{"name": "Seoul Robot & AI Museum (서울로봇인공지능과학관)", "poi_type": "museum", "estimated_stay_time": 90}}, {{"name": "Dream Forest (드림숲)", "poi_type": "park", "estimated_stay_time": null}}]}}, {{"day": 2, "pois": [{{"name": "Seoul Botanic Park (서울식물원)", "poi_type": "park", "estimated_stay_time": 60}}, {{"name": "Han River Ferry (한강 유람선)", "poi_type": "tourist_spot", "estimated_stay_time": null}}]}}]}}

Webpage content:
{truncated}{route_hint}"""

    try:
        response = gemini_generate(prompt)
        raw = response.text.strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        course_title = str(data.get("course_title", "")).strip()
        course_type = data.get("type", "single")
        days = data.get("days", [])
        themes = data.get("theme_category", [])

        if not course_title:
            raise ValueError("Gemini returned empty course_title")

        if not isinstance(days, list) or len(days) == 0:
            raise ValueError("Gemini returned empty days array")

        # Validate and clean themes
        if not isinstance(themes, list):
            themes = ["general"]
        valid_themes = [t for t in themes if t in VALID_THEMES]
        if not valid_themes:
            valid_themes = ["general"]

        results = []
        is_multiday = course_type == "multiday" and len(days) > 1

        for day_obj in days:
            day_num = day_obj.get("day", 1)
            pois = day_obj.get("pois", [])

            cleaned_pois = []
            for p in pois:
                name = clean_text(p.get("name", ""))
                if not name or len(name) < 2:
                    continue
                raw_stay = p.get("estimated_stay_time")
                cleaned_pois.append({
                    "name": name,
                    "poi_type": p.get("poi_type", "tourist_spot"),
                    "estimated_stay_time": int(raw_stay) if raw_stay is not None else None
                })

            if not cleaned_pois:
                continue  # skip empty days

            title = f"{course_title} - Day {day_num}" if is_multiday else course_title

            results.append({
                "course_title": title,
                "day_num": day_num if is_multiday else None,
                "pois": cleaned_pois,
                "theme_category": valid_themes
            })

        if not results:
            raise ValueError("All days were empty after cleaning POI names")

        return results

    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {e} | raw: {raw[:200]}")
    except Exception as e:
        raise ValueError(f"Gemini parse failed: {e}")


def parse_course(item):
    """Fetch page via Tavily, parse + classify via Gemini.
    Returns a list of parsed day dicts (always at least one entry)."""
    url = item["source_url"]
    page_text = extract_page_content(url)
    return parse_course_with_gemini(page_text, url)


# =========================
# 5. 메인 실행
# =========================

def build_dataset():
    url_items = read_url_files()

    course_master_rows = []
    course_sequence_rows = []
    poi_dict = {}
    failed_rows = []

    for item in url_items:
        course_id = item["course_id"]
        source = item["source"]
        url = item["source_url"]
        course_type = item["course_type"]

        print(f"[START] {course_id} | {source} | {url}")

        try:
            parsed_days = parse_course(item)

            for parsed in parsed_days:
                course_title = parsed["course_title"]
                pois = parsed["pois"]
                theme_category = parsed["theme_category"]
                theme_str = ",".join(theme_category)
                day_num = parsed["day_num"]

                # For multi-day courses, suffix the course_id with _D1, _D2, etc.
                sub_course_id = f"{course_id}_D{day_num}" if day_num else course_id

                course_master_rows.append({
                    "course_id": sub_course_id,
                    "parent_course_id": course_id if day_num else None,
                    "day_num": day_num,
                    "source": source,
                    "source_url": url,
                    "course_title": course_title,
                    "course_type": course_type,
                    "theme_category": theme_str,
                    "num_pois": len(pois),
                    "status": "success"
                })

                for order, poi in enumerate(pois, start=1):
                    poi_name = poi["name"]
                    poi_type = poi["poi_type"]
                    estimated_stay_time = poi["estimated_stay_time"]

                    # Generate a stable POI ID from the name alone
                    poi_id = "POI_" + hashlib.md5(poi_name.encode("utf-8")).hexdigest()[:10]

                    course_sequence_rows.append({
                        "course_id": sub_course_id,
                        "sequence_order": order,
                        "poi_id": poi_id,
                        "poi_name": poi_name,
                        "poi_type": poi_type,
                        "address": None,
                        "lat": None,
                        "lng": None,
                        "estimated_stay_time": estimated_stay_time,
                    })

                    if poi_id not in poi_dict:
                        poi_dict[poi_id] = {
                            "poi_id": poi_id,
                            "poi_name": poi_name,
                            "poi_type": poi_type,
                            "address": None,
                            "lat": None,
                            "lng": None,
                            "description": None,
                        }

                print(f"[SUCCESS] {sub_course_id} | '{course_title}' | themes: {theme_category} | {len(pois)} POIs")

        except Exception as e:
            print(f"[FAILED] {course_id} | {e}")

            failed_rows.append({
                "source": source,
                "url": url,
                "error_reason": str(e)
            })

            course_master_rows.append({
                "course_id": course_id,
                "parent_course_id": None,
                "day_num": None,
                "source": source,
                "source_url": url,
                "course_title": None,
                "course_type": course_type,
                "theme_category": None,
                "num_pois": 0,
                "status": "failed"
            })

    course_master_df = pd.DataFrame(course_master_rows)
    course_sequence_df = pd.DataFrame(course_sequence_rows)
    poi_master_df = pd.DataFrame(list(poi_dict.values()))
    failed_df = pd.DataFrame(failed_rows)

    course_master_df.to_csv(os.path.join(OUTPUT_DIR, "course_master.csv"), index=False, encoding="utf-8-sig")
    course_sequence_df.to_csv(os.path.join(OUTPUT_DIR, "course_sequence.csv"), index=False, encoding="utf-8-sig")
    poi_master_df.to_csv(os.path.join(OUTPUT_DIR, "poi_master.csv"), index=False, encoding="utf-8-sig")
    failed_df.to_csv(os.path.join(OUTPUT_DIR, "failed_urls.csv"), index=False, encoding="utf-8-sig")

    course_json = []

    if not course_sequence_df.empty:
        for sub_course_id, group in course_sequence_df.groupby("course_id"):
            master = course_master_df[course_master_df["course_id"] == sub_course_id].iloc[0]

            entry = {
                "course_id": sub_course_id,
                "source": master["source"],
                "source_url": master["source_url"],
                "course_title": master["course_title"],
                "theme_category": str(master["theme_category"]).split(",") if pd.notna(master["theme_category"]) else ["general"],
                "sequence": [
                    {
                        "sequence_order": int(row["sequence_order"]),
                        "poi_name": row["poi_name"],
                        "poi_type": row["poi_type"],
                        "address": None,
                        "lat": None,
                        "lng": None,
                        "estimated_stay_time": int(row["estimated_stay_time"]) if pd.notna(row["estimated_stay_time"]) else None
                    }
                    for _, row in group.sort_values("sequence_order").iterrows()
                ]
            }

            # Include parent linkage for multi-day sub-courses
            if pd.notna(master.get("parent_course_id")):
                entry["parent_course_id"] = master["parent_course_id"]
                entry["day_num"] = int(master["day_num"])

            course_json.append(entry)

    with open(os.path.join(OUTPUT_DIR, "course_data.json"), "w", encoding="utf-8") as f:
        json.dump(course_json, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {len(course_master_df)} courses | {len(course_sequence_df)} POIs | output in /{OUTPUT_DIR}")


if __name__ == "__main__":
    build_dataset()