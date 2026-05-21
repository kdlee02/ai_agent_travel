import pandas as pd

# -----------------------------------
# 현재 step2 읽기
# -----------------------------------
df = pd.read_csv("output/poi_master_step2.csv")

# -----------------------------------
# 목표 컬럼만 추출
# (파트 2 output 목표 스펙 기준)
# -----------------------------------
target_cols = [
    "poi_id",
    "poi_name",
    "poi_type",
    "vs_matched",
    "vs_content_id",
    "vs_admission_fee_text",
    "vs_admission_fee_value",
    "vs_admission_fee_free",
    "vs_foreign_languages",
    "vs_homepage",
    "vs_phone",
    "vs_use_time",
    "vs_holiday",
    "step2_status",
]

# 없는 컬럼은 None으로 추가
for col in target_cols:
    if col not in df.columns:
        df[col] = None

df_clean = df[target_cols].copy()

# -----------------------------------
# vs_matched 정리
# (step2_status == success면 True)
# -----------------------------------
df_clean["vs_matched"] = df_clean["step2_status"].apply(
    lambda x: True if x == "success" else False
)

# -----------------------------------
# step2_status 정리
# (not_found / success만 유지)
# -----------------------------------
def clean_status(s):
    if s == "success":
        return "success"
    return "not_found"

df_clean["step2_status"] = df_clean["step2_status"].apply(clean_status)

# -----------------------------------
# 저장
# -----------------------------------
df_clean.to_csv("output/poi_master_step2.csv", index=False, encoding="utf-8-sig")
print(f"저장 완료: {len(df_clean)}행, {len(df_clean.columns)}컬럼")
print()
print("=== 최종 컬럼 ===")
print(df_clean.columns.tolist())
print()
print("=== 채워진 비율 ===")
for col in target_cols:
    filled = df_clean[col].apply(
        lambda x: str(x).strip() not in ['', 'nan', 'None', 'False']
        if col not in ['vs_matched', 'vs_admission_fee_free']
        else pd.notna(x)
    ).sum()
    print(f"  {col}: {filled}/{len(df_clean)} ({filled/len(df_clean)*100:.1f}%)")