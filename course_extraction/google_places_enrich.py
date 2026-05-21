import pandas as pd
import requests
import time
import re

GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"

# -----------------------------------
# 읽기
# -----------------------------------
df = pd.read_csv("output/poi_master_step2.csv")

# 테스트: 처음엔 head(10)으로 확인
# df = df.head(10)  # ← 테스트 완료 후 주석 처리

# -----------------------------------
# 새 컬럼 초기화
# -----------------------------------
for col in ["google_place_id", "google_phone", "google_rating",
            "google_user_ratings_total", "google_opening_hours",
            "google_website", "google_lat", "google_lng",
            "google_status"]:
    if col not in df.columns:
        df[col] = None

# -----------------------------------
# Google Places Text Search
# -----------------------------------
def google_text_search(query):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{query} 서울",
        "key": GOOGLE_API_KEY,
        "language": "ko",
        "region": "kr",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        results = r.json().get("results", [])
        if not results:
            return None
        return results[0]  # 첫 번째 결과
    except Exception as e:
        print(f"  [Search 예외] {e}")
        return None

# -----------------------------------
# Google Place Details
# -----------------------------------
def google_place_details(place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,formatted_phone_number,rating,user_ratings_total,"
                  "opening_hours,website,geometry",
        "key": GOOGLE_API_KEY,
        "language": "ko",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json().get("result", {})
    except Exception as e:
        print(f"  [Details 예외] {e}")
        return {}

def extract_korean(name):
    m = re.search(r'\(([^)]+)\)', str(name))
    if m and re.search(r'[가-힣]', m.group(1)):
        return m.group(1).strip()
    return str(name)

def format_opening_hours(hours_data):
    """Google opening_hours → 간단한 텍스트"""
    if not hours_data:
        return ""
    weekday_text = hours_data.get("weekday_text", [])
    return " | ".join(weekday_text) if weekday_text else ""

# -----------------------------------
# 메인 루프
# -----------------------------------
total = len(df)
for idx, row in df.iterrows():

    # 멱등성: 이미 처리된 건 건너뜀
    if pd.notna(row.get("google_status")):
        continue

    # 검색어 결정: vs_title 있으면 우선, 없으면 poi_name 한글명
    if pd.notna(row.get("vs_title")) and str(row["vs_title"]).strip() not in ["", "nan"]:
        query = str(row["vs_title"]).strip()
    else:
        query = extract_korean(str(row["poi_name"]))

    print(f"[{idx}/{total}] {query}", end=" ")

    # 1) Text Search → place_id
    result = google_text_search(query)
    if not result:
        df.at[idx, "google_status"] = "not_found"
        print("→ 검색 실패")
        time.sleep(0.2)
        continue

    place_id = result.get("place_id")
    if not place_id:
        df.at[idx, "google_status"] = "no_place_id"
        print("→ place_id 없음")
        time.sleep(0.2)
        continue

    # 2) Place Details
    details = google_place_details(place_id)

    # 좌표
    geometry = result.get("geometry", {}).get("location", {})
    g_lat = geometry.get("lat")
    g_lng = geometry.get("lng")

    # 저장
    df.at[idx, "google_place_id"]           = place_id
    df.at[idx, "google_phone"]              = details.get("formatted_phone_number", "")
    df.at[idx, "google_rating"]             = details.get("rating")
    df.at[idx, "google_user_ratings_total"] = details.get("user_ratings_total")
    df.at[idx, "google_opening_hours"]      = format_opening_hours(details.get("opening_hours"))
    df.at[idx, "google_website"]            = details.get("website", "")
    df.at[idx, "google_lat"]               = g_lat
    df.at[idx, "google_lng"]               = g_lng
    df.at[idx, "google_status"]            = "success"

    print(f"→ ✓ {details.get('name', '')} (rating={details.get('rating', '-')})")

    # 중간 저장 (50개마다)
    if idx % 50 == 0 and idx > 0:
        df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")
        print(f"  [중간저장] {idx}건")

    time.sleep(0.2)  # rate limit

# -----------------------------------
# 최종 저장
# -----------------------------------
df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")

print("\n===== 완료 =====")
print(f"성공: {(df['google_status'] == 'success').sum()}건")
print(f"실패: {(df['google_status'] == 'not_found').sum()}건")
print()
print("=== Google 필드 채워진 비율 ===")
for col in ["google_place_id", "google_phone", "google_rating",
            "google_opening_hours", "google_website", "google_lat"]:
    if col in df.columns:
        filled = df[col].apply(
            lambda x: str(x).strip() not in ['', 'nan', 'None']
        ).sum()
        print(f"  {col}: {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")