import pandas as pd

df = pd.read_csv("output/poi_master_step2.csv")

print("=== 초기화 전 ===")
print(df["step2_match_type"].value_counts(dropna=False))
print(f"\nsuccess: {(df['step2_status'] == 'success').sum()}건")
print(f"not_found: {(df['step2_status'] == 'not_found').sum()}건")

# short_kor 행 인덱스 추출
short_kor_idx = df[df["step2_match_type"] == "short_kor"].index

# step2_match_type은 유지, 나머지만 초기화
reset_cols = [
    "vs_title", "vs_addr", "vs_tel", "vs_lat", "vs_lng",
    "vs_content_id", "vs_content_type", "vs_firstimage",
    "vs_admission_fee", "vs_admission_fee_text", "vs_admission_fee_value",
    "vs_admission_fee_free", "vs_use_time", "vs_holiday",
    "vs_homepage", "vs_foreign_languages", "vs_phone",
    "step2_detail_status",
    # step2_match_type은 의도적으로 제외 (디버깅용 유지)
]

for idx in short_kor_idx:
    for col in reset_cols:
        if col in df.columns:
            df.at[idx, col] = None
    df.at[idx, "step2_status"] = "not_found"
    df.at[idx, "vs_matched"] = False

print(f"\n=== short_kor 초기화: {len(short_kor_idx)}건 ===")
print("(step2_match_type 컬럼은 디버깅용으로 유지)")

print("\n=== 초기화 후 ===")
print(df["step2_match_type"].value_counts(dropna=False))
print(f"\nsuccess: {(df['step2_status'] == 'success').sum()}건")
print(f"not_found: {(df['step2_status'] == 'not_found').sum()}건")

df.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")

print("\n=== 최종 핵심 필드 채워진 비율 ===")
check_cols = ["vs_matched", "vs_content_id", "vs_admission_fee_text",
              "vs_use_time", "vs_holiday", "vs_homepage", "vs_foreign_languages"]
for col in check_cols:
    if col in df.columns:
        filled = df[col].apply(
            lambda x: str(x).strip() not in ['', 'nan', 'None', '[]', 'False']
        ).sum()
        print(f"  {col}: {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")

print("\n완료!")