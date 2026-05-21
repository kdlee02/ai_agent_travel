import pandas as pd
import re

df = pd.read_csv("output/poi_master_step2.csv")

# -----------------------------------------------
# 1. vs_matched 컬럼 추가
# -----------------------------------------------
df["vs_matched"] = df["step2_status"].apply(lambda x: True if x == "success" else False)
print(f"vs_matched 추가 완료: True={df['vs_matched'].sum()}건")

# -----------------------------------------------
# 2. vs_admission_fee 파싱
#    → vs_admission_fee_text, vs_admission_fee_value, vs_admission_fee_free
# -----------------------------------------------
def parse_admission_fee(text):
    """
    '어른 3,000원' → (3000, False)
    '무료'         → (0, True)
    ''             → (None, None)
    """
    if not text or str(text).strip() in ['', 'nan', 'None']:
        return None, None

    text = str(text).strip()

    # 무료 여부
    if re.search(r"무료|free", text, re.I):
        # 어른 요금이 따로 있는지 확인
        m = re.search(r"어른[^\d]*([\d,]+)\s*원", text)
        if m:
            val = int(m.group(1).replace(",", ""))
            return val, False
        return 0, True

    # 어른 요금
    m = re.search(r"어른[^\d]*([\d,]+)\s*원", text)
    if m:
        return int(m.group(1).replace(",", "")), False

    # 일반/성인 요금
    m = re.search(r"(?:일반|성인)[^\d]*([\d,]+)\s*원", text)
    if m:
        return int(m.group(1).replace(",", "")), False

    # 첫 번째 숫자
    m = re.search(r"([\d,]+)\s*원", text)
    if m:
        return int(m.group(1).replace(",", "")), False

    return None, False

# 컬럼 추가
df["vs_admission_fee_text"] = df["vs_admission_fee"].apply(
    lambda x: str(x).strip() if pd.notna(x) and str(x).strip() not in ['', 'nan', 'None'] else None
)

parsed = df["vs_admission_fee"].apply(parse_admission_fee)
df["vs_admission_fee_value"] = parsed.apply(lambda x: x[0])
df["vs_admission_fee_free"] = parsed.apply(lambda x: x[1])

print(f"vs_admission_fee_text: {df['vs_admission_fee_text'].notna().sum()}건")
print(f"vs_admission_fee_value: {df['vs_admission_fee_value'].notna().sum()}건")
print(f"vs_admission_fee_free: {df['vs_admission_fee_free'].notna().sum()}건")
print(f"  무료: {(df['vs_admission_fee_free'] == True).sum()}건")
print(f"  유료: {(df['vs_admission_fee_free'] == False).sum()}건")

# -----------------------------------------------
# 3. vs_foreign_languages 보완
#    overview 텍스트 외에 vs_title로도 추가 추론
# -----------------------------------------------
def enrich_foreign_languages(row):
    existing = str(row.get("vs_foreign_languages", "")).strip()

    # 이미 값 있으면 유지
    if existing and existing not in ['', 'nan', 'None', '[]']:
        return existing

    # vs_title 기반 추론 (주요 관광지는 영어 지원 기본)
    title = str(row.get("vs_title", ""))
    poi_type = str(row.get("poi_type", ""))

    major_tourist = ["경복궁", "창덕궁", "덕수궁", "창경궁", "경희궁",
                     "국립중앙박물관", "국립현대미술관", "국립민속박물관",
                     "전쟁기념관", "북촌한옥마을", "인사동", "명동",
                     "남산", "N서울타워", "롯데월드", "올림픽공원"]

    if any(k in title for k in major_tourist):
        return "['en']"

    # poi_type이 history/museum이면 영어 지원 기본
    if poi_type in ["history", "museum"]:
        return "['en']"

    return existing if existing not in ['nan', 'None'] else ""

df["vs_foreign_languages"] = df.apply(enrich_foreign_languages, axis=1)
filled_langs = df["vs_foreign_languages"].apply(
    lambda x: str(x).strip() not in ['', 'nan', 'None', '[]']
).sum()
print(f"\nvs_foreign_languages 보완 후: {filled_langs}건")

# -----------------------------------------------
# 4. vs_phone 컬럼 추가 (vs_tel 있으면 복사)
# -----------------------------------------------
if "vs_tel" in df.columns:
    df["vs_phone"] = df["vs_tel"]
else:
    df["vs_phone"] = None
print(f"vs_phone: {df['vs_phone'].notna().sum()}건")

# -----------------------------------------------
# 5. 최종 저장
# -----------------------------------------------
df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")

print("\n=== 최종 필드 채워진 비율 ===")
check_cols = [
    "vs_matched", "vs_content_id",
    "vs_admission_fee_text", "vs_admission_fee_value", "vs_admission_fee_free",
    "vs_foreign_languages", "vs_homepage", "vs_phone",
    "vs_use_time", "vs_holiday"
]
for col in check_cols:
    if col in df.columns:
        filled = df[col].apply(
            lambda x: str(x).strip() not in ['', 'nan', 'None', '[]', 'False', 'True']
            if col not in ['vs_matched', 'vs_admission_fee_free']
            else pd.notna(x)
        ).sum()
        print(f"  {col}: {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")

print("\n완료!")