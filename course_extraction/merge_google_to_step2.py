import pandas as pd

# -----------------------------------
# 1. 파일 읽기
# -----------------------------------
df = pd.read_csv("output/poi_master_step2.csv")
print(f"전체 POI: {len(df)}개")

def is_empty(val):
    return pd.isna(val) or str(val).strip() in ['', 'nan', 'None']

# -----------------------------------
# 2. Google 데이터로 기존 컬럼 보완
#    (TourAPI 값 있으면 유지, 없으면 Google로 채움)
# -----------------------------------

# vs_phone
filled_phone = 0
for idx, row in df.iterrows():
    if is_empty(row.get("vs_phone")) and not is_empty(row.get("google_phone")):
        df.at[idx, "vs_phone"] = row["google_phone"]
        filled_phone += 1
print(f"vs_phone 보완: {filled_phone}건")

# vs_homepage
filled_homepage = 0
for idx, row in df.iterrows():
    if is_empty(row.get("vs_homepage")) and not is_empty(row.get("google_website")):
        df.at[idx, "vs_homepage"] = row["google_website"]
        filled_homepage += 1
print(f"vs_homepage 보완: {filled_homepage}건")

# vs_use_time
filled_usetime = 0
for idx, row in df.iterrows():
    if is_empty(row.get("vs_use_time")) and not is_empty(row.get("google_opening_hours")):
        df.at[idx, "vs_use_time"] = row["google_opening_hours"]
        filled_usetime += 1
print(f"vs_use_time 보완: {filled_usetime}건")

# vs_lat / vs_lng
filled_lat = 0
for idx, row in df.iterrows():
    if is_empty(row.get("vs_lat")) and not is_empty(row.get("google_lat")):
        df.at[idx, "vs_lat"] = row["google_lat"]
        df.at[idx, "vs_lng"] = row["google_lng"]
        filled_lat += 1
print(f"vs_lat/lng 보완: {filled_lat}건")

# -----------------------------------
# 3. google_* 컬럼 전부 제거
# -----------------------------------
google_cols = [c for c in df.columns if c.startswith("google_")]
df = df.drop(columns=google_cols)
print(f"\n제거된 google_* 컬럼: {google_cols}")

# -----------------------------------
# 4. 최종 저장
# -----------------------------------
df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: poi_master_step2.csv")

# -----------------------------------
# 5. 결과 요약
# -----------------------------------
print("\n=== 최종 핵심 필드 채워진 비율 ===")
check_cols = {
    "vs_phone": "전화번호",
    "vs_homepage": "홈페이지",
    "vs_use_time": "운영시간",
    "vs_lat": "위도",
    "vs_lng": "경도",
    "vs_holiday": "휴무일",
    "vs_admission_fee_text": "입장료",
    "vs_foreign_languages": "외국어 안내",
}
for col, desc in check_cols.items():
    if col in df.columns:
        filled = df[col].apply(lambda x: not is_empty(x)).sum()
        print(f"  {desc} ({col}): {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")