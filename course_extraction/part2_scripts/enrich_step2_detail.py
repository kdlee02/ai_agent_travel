import pandas as pd
import requests
import time
import urllib.parse
import re
from difflib import SequenceMatcher

SERVICE_KEY = "YOUR_TOUR_API_KEY"

df = pd.read_csv("output/poi_master_step2.csv")

# -----------------------------------------------
# 유틸 함수
# -----------------------------------------------
def extract_korean(name):
    m = re.search(r'\(([^)]+)\)', str(name))
    if m and re.search(r'[가-힣]', m.group(1)):
        return m.group(1).strip()
    return str(name)

def similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def is_seoul_coord(lat, lng):
    try:
        lat, lng = float(lat), float(lng)
        return 37.4 <= lat <= 37.7 and 126.7 <= lng <= 127.2
    except:
        return False

# -----------------------------------------------
# STEP 1: False Positive 검수 및 초기화
# -----------------------------------------------
print("=" * 50)
print("STEP 1: False Positive 검수")
print("=" * 50)

reset_count = 0
success_df = df[df['step2_status'] == 'success'].copy()

for idx, row in success_df.iterrows():
    match_type = str(row.get('step2_match_type', ''))
    poi_kor = extract_korean(str(row['poi_name']))
    vs_title = str(row.get('vs_title', ''))
    vs_lat = row.get('vs_lat')
    vs_lng = row.get('vs_lng')

    should_reset = False
    reason = ""

    # palace fallback은 유지
    if 'palace_fallback' in match_type:
        continue

    # 1) short_kor 매칭 유사도 검사
    if match_type == 'short_kor':
        sim = similarity(poi_kor, vs_title)
        if sim < 0.4 and poi_kor[:2] not in vs_title:
            should_reset = True
            reason = f"short_kor 유사도 낮음 ({sim:.2f}): {poi_kor} → {vs_title}"

    # 2) 서울 외 좌표
    if not should_reset and pd.notna(vs_lat) and pd.notna(vs_lng):
        if not is_seoul_coord(vs_lat, vs_lng):
            should_reset = True
            reason = f"서울 외 좌표: ({vs_lat}, {vs_lng})"

    if should_reset:
        reset_cols = ["vs_title", "vs_addr", "vs_tel", "vs_lat", "vs_lng",
                      "vs_content_id", "vs_content_type", "vs_firstimage",
                      "step2_status", "step2_match_type"]
        for col in reset_cols:
            df.at[idx, col] = None
        df.at[idx, "step2_status"] = "not_found"
        reset_count += 1
        print(f"  [초기화] [{idx}] {reason}")

print(f"\n초기화 완료: {reset_count}건")
print(f"현재 success: {(df['step2_status'] == 'success').sum()}건")
print(f"현재 not_found: {(df['step2_status'] == 'not_found').sum()}건")

# -----------------------------------------------
# STEP 2: Detail API Enrichment
# -----------------------------------------------
print("\n" + "=" * 50)
print("STEP 2: Detail API Enrichment")
print("=" * 50)

for col in ["vs_admission_fee", "vs_use_time", "vs_holiday",
            "vs_homepage", "vs_foreign_languages", "step2_detail_status"]:
    if col not in df.columns:
        df[col] = None

def tourapi_detail(content_id):
    """detailCommon2 - 최소 파라미터로 호출"""
    url = "https://apis.data.go.kr/B551011/KorService2/detailCommon2"
    params = {
        "MobileOS": "ETC",
        "MobileApp": "SeoulMate",
        "_type": "json",
        "contentId": int(content_id),
    }
    encoded = urllib.parse.urlencode(params)
    full_url = f"{url}?serviceKey={SERVICE_KEY}&{encoded}"
    try:
        r = requests.get(full_url, timeout=10)
        if not r.text or r.text.strip().startswith("<"):
            return None
        data = r.json()
        body = data["response"]["body"]
        items = body.get("items", {})
        if not items or items == "":
            return None
        item = items.get("item", [])
        if isinstance(item, list):
            return item[0] if item else None
        return item
    except Exception as e:
        print(f"  [detail 예외] {e}")
        return None

def tourapi_intro(content_id, content_type_id):
    """detailIntro2 - 입장료, 운영시간, 휴무일"""
    url = "https://apis.data.go.kr/B551011/KorService2/detailIntro2"
    params = {
        "MobileOS": "ETC",
        "MobileApp": "SeoulMate",
        "_type": "json",
        "contentId": int(content_id),
        "contentTypeId": int(content_type_id),
    }
    encoded = urllib.parse.urlencode(params)
    full_url = f"{url}?serviceKey={SERVICE_KEY}&{encoded}"
    try:
        r = requests.get(full_url, timeout=10)
        if not r.text or r.text.strip().startswith("<"):
            return None
        data = r.json()
        body = data["response"]["body"]
        items = body.get("items", {})
        if not items or items == "":
            return None
        item = items.get("item", [])
        if isinstance(item, list):
            return item[0] if item else None
        return item
    except Exception as e:
        print(f"  [intro 예외] {e}")
        return None

def parse_foreign_languages(text):
    langs = []
    if not text:
        return ""
    if re.search(r"영어|English", text, re.I): langs.append("en")
    if re.search(r"중국어|Chinese|中文", text, re.I): langs.append("zh")
    if re.search(r"일본어|Japanese|日本語", text, re.I): langs.append("ja")
    return str(langs) if langs else ""

# 멱등성: step2_detail_status 없는 success 건만 처리
target = df[
    (df['step2_status'] == 'success') &
    df['vs_content_id'].notna() &
    df['step2_detail_status'].isna()
]
print(f"Detail API 호출 대상: {len(target)}건\n")

for i, (idx, row) in enumerate(target.iterrows()):
    content_id = row["vs_content_id"]
    content_type = row.get("vs_content_type")

    try:
        content_id_int = int(float(content_id))
    except:
        df.at[idx, "step2_detail_status"] = "invalid_id"
        continue

    poi_name = str(row['poi_name'])
    print(f"[{idx}] {extract_korean(poi_name)}", end=" ")

    # detailCommon2 호출
    detail = tourapi_detail(content_id_int)
    if detail:
        homepage = detail.get("homepage", "") or ""
        homepage = re.sub(r'<[^>]+>', '', homepage).strip()
        df.at[idx, "vs_homepage"] = homepage

        overview = detail.get("overview", "") or ""
        df.at[idx, "vs_foreign_languages"] = parse_foreign_languages(overview)

    # detailIntro2 호출
    intro = None
    if content_type and str(content_type) not in ["nan", "None", ""]:
        try:
            intro = tourapi_intro(content_id_int, int(float(content_type)))
        except:
            pass

    if intro:
        fee = (intro.get("usefee") or intro.get("entrancefee") or
               intro.get("admission") or "")
        df.at[idx, "vs_admission_fee"] = str(fee).strip() if fee else ""

        use_time = (intro.get("usetime") or intro.get("opentime") or
                    intro.get("playtime") or "")
        df.at[idx, "vs_use_time"] = str(use_time).strip() if use_time else ""

        holiday = (intro.get("restdate") or intro.get("closingday") or "")
        df.at[idx, "vs_holiday"] = str(holiday).strip() if holiday else ""

        df.at[idx, "step2_detail_status"] = "success"
        print("✓")
    else:
        df.at[idx, "step2_detail_status"] = "no_intro"
        print("- (intro 없음)")

    if (i + 1) % 50 == 0:
        df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")
        print(f"  [중간저장] {i+1}건 처리")

    time.sleep(0.3)

df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 50)
print("최종 결과 요약")
print("=" * 50)
print(f"success:    {(df['step2_status'] == 'success').sum()}건")
print(f"not_found:  {(df['step2_status'] == 'not_found').sum()}건")
print(f"detail 성공: {(df['step2_detail_status'] == 'success').sum()}건")
print(f"detail 없음: {(df['step2_detail_status'] == 'no_intro').sum()}건")
print()
print("=== 핵심 필드 채워진 비율 ===")
for col in ["vs_lat", "vs_lng", "vs_content_id",
            "vs_admission_fee", "vs_use_time", "vs_holiday", "vs_homepage"]:
    if col in df.columns:
        non_empty = df[col].apply(lambda x: str(x).strip() not in ['', 'nan', 'None']).sum()
        print(f"{col}: {non_empty}/{len(df)} ({non_empty/len(df)*100:.1f}%)")