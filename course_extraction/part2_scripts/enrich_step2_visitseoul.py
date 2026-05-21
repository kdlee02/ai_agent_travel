import pandas as pd
import requests
import time
import re
import urllib.parse

# ← 새 키로 교체
SERVICE_KEY = "YOUR_TOUR_API_KEY"
BASE_URL = "https://apis.data.go.kr/B551011/KorService2/searchKeyword2"

# ← poi_master.csv 아닌 step2.csv 읽기 (기존 성공 데이터 유지)
df = pd.read_csv("output/poi_master_step2.csv")

for col in ["vs_title", "vs_addr", "vs_tel", "vs_lat", "vs_lng",
            "vs_content_id", "vs_content_type", "vs_firstimage",
            "step2_status", "step2_match_type"]:
    if col not in df.columns:
        df[col] = None

PALACE_FALLBACK = {
    "창덕궁": ["인정문", "인정전", "선정전", "희정당", "대조전", "낙선재", "돈화문"],
    "덕수궁": ["중화문", "중화전", "석어당", "덕홍전", "함녕전", "정관헌", "즉조당", "준명당", "석조전", "대한문"],
    "경복궁": ["근정전", "경회루", "향원정", "강녕전", "교태전", "광화문", "흥례문", "근정문", "사정전", "자경전", "동십자각"],
    "창경궁": ["명정전", "홍화문", "통명전"],
    "경희궁": ["숭정전", "흥화문"],
}

def get_palace_fallback(kor_name):
    for palace, buildings in PALACE_FALLBACK.items():
        if any(b in kor_name for b in buildings):
            return palace
    return None

def extract_korean(name):
    m = re.search(r"\(([^)]+)\)", str(name))
    if m and re.search(r"[가-힣]", m.group(1)):
        return m.group(1).strip()
    return str(name)

def extract_english(name):
    return str(name).split("(")[0].strip()

def tourapi_search(keyword):
    query_params = {
        "numOfRows": 5,
        "pageNo": 1,
        "MobileOS": "ETC",
        "MobileApp": "SeoulMate",
        "_type": "json",
        "keyword": keyword,
        "areaCode": 1,
    }
    encoded = urllib.parse.urlencode(query_params)
    url = f"{BASE_URL}?serviceKey={SERVICE_KEY}&{encoded}"
    try:
        r = requests.get(url, timeout=10)
        if not r.text or r.text.strip().startswith("<"):
            print(f"  [HTML/빈응답] {r.status_code}: {r.text[:80]}")
            return None
        if r.status_code == 429:
            print("  [한도초과] 1분 대기 후 재시도...")
            time.sleep(60)
            r = requests.get(url, timeout=10)
        data = r.json()
        body = data["response"]["body"]
        if body.get("totalCount", 0) == 0:
            return None
        items = body["items"]["item"]
        return items if isinstance(items, list) else [items]
    except Exception as e:
        print(f"  [예외] {e}")
        return None

for idx, row in df.iterrows():
    if pd.isna(row["poi_name"]):
        continue

    # ← 멱등성: success나 not_found면 건너뜀
    status = row.get("step2_status")
    if pd.notna(status) and status in ("success", "not_found"):
        continue

    raw_name = str(row["poi_name"])
    name_kor = extract_korean(raw_name)
    name_eng = extract_english(raw_name)
    matched = False
    match_type = None

    print(f"\n[{idx}] {name_kor}")

    # 시도 1: 한글명
    items = tourapi_search(name_kor)
    if items:
        match_type = "exact_kor"
        matched = True

    # 시도 2: 영문명
    if not matched and name_eng != name_kor:
        items = tourapi_search(name_eng)
        if items:
            match_type = "exact_eng"
            matched = True

    # 시도 3: 한글 앞 2글자 축약
    if not matched and len(name_kor) >= 3:
        short = name_kor[:2]
        items = tourapi_search(short)
        if items:
            filtered = [i for i in items if name_kor[:2] in i.get("title", "")]
            if filtered:
                items = filtered
                match_type = "short_kor"
                matched = True

    # 시도 4: 궁궐 건물 → 상위 궁 fallback
    if not matched:
        palace = get_palace_fallback(name_kor)
        if palace:
            items = tourapi_search(palace)
            if items:
                match_type = f"palace_fallback:{palace}"
                matched = True
                print(f"  → 궁 fallback: {palace}")

    if not matched:
        print("  → 최종 실패")
        df.at[idx, "step2_status"] = "not_found"
        time.sleep(0.3)
        continue

    item = items[0]
    df.at[idx, "vs_title"]         = item.get("title")
    df.at[idx, "vs_addr"]          = (item.get("addr1", "") + " " + item.get("addr2", "")).strip()
    df.at[idx, "vs_tel"]           = item.get("tel")
    df.at[idx, "vs_lat"]           = item.get("mapy")
    df.at[idx, "vs_lng"]           = item.get("mapx")
    df.at[idx, "vs_content_id"]    = item.get("contentid")
    df.at[idx, "vs_content_type"]  = item.get("contenttypeid")
    df.at[idx, "vs_firstimage"]    = item.get("firstimage")
    df.at[idx, "step2_status"]     = "success"
    df.at[idx, "step2_match_type"] = match_type

    print(f"  → 성공 ({match_type}): {item.get('title')}")

    # 중간 저장 (50개마다)
    if idx % 50 == 0 and idx > 0:
        df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")
        print(f"  [중간저장] {idx}건")

    time.sleep(0.3)

df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")

print("\n===== 완료 =====")
print(f"성공:   {(df['step2_status'] == 'success').sum()}건")
print(f"미발견: {(df['step2_status'] == 'not_found').sum()}건")
if "step2_match_type" in df.columns:
    print("\n--- 매칭 유형별 ---")
    print(df["step2_match_type"].value_counts().to_string())