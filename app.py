import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="CVD 위험도 예측", page_icon="🫀")

# ── 모델 로드 ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('cvh_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle = load_model()
model          = bundle['model']
scaler         = bundle['scaler']
feature_cols   = bundle['feature_columns']
scale_cols     = bundle['scale_cols']

# ── 제목 ─────────────────────────────────────────────────────
st.title("🫀 심혈관 건강(CVH) 위험도 예측")
st.caption("국민건강영양조사 제9기 (2022~2024) · LDA 모델 · AUC 0.7227")
st.divider()

# ── 입력 ─────────────────────────────────────────────────────
st.subheader("개인 정보 입력")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("연령 (세)", min_value=19, max_value=80, value=40)

    sex = st.selectbox("성별", ["남성", "여성"])

    edu = st.selectbox(
        "학력",
        ["중졸 이하", "고졸", "전문대졸", "대졸 이상"]
    )

with col2:
    ainc = st.number_input("월평균 가구 소득 (만원)", min_value=17, max_value=1500, value=400, step=10)

    marri = st.selectbox(
        "가족 구조 유형",
        ["기혼·유자녀", "기혼·무자녀", "미혼", "기타 (이혼·사별·별거)"]
    )

    region = st.selectbox("거주지역", ["수도권", "비수도권"])

st.divider()

# ── 입력값 → 모델 형식으로 변환 ──────────────────────────────
def encode_input(age, sex, edu, marri, ainc, region):
    sex_val  = 2.0 if sex == "여성" else 1.0
    edu_val  = {"중졸 이하": 1.0, "고졸": 2.0, "전문대졸": 3.0, "대졸 이상": 4.0}[edu]
    marri_val = {
        "기혼·유자녀":          "Married_with_children",
        "기혼·무자녀":          "Married_without_children",
        "미혼":                 "Single",
        "기타 (이혼·사별·별거)": "Other",
    }[marri]
    metro = 1 if region == "수도권" else 0

    row = {col: 0 for col in feature_cols}
    row['age']          = age
    row['ainc']         = ainc
    row['region_metro'] = metro

    if sex_val == 2.0:
        row['sex_2.0'] = 1

    if edu_val == 2.0:
        row['edu_2.0'] = 1
    elif edu_val == 3.0:
        row['edu_3.0'] = 1
    elif edu_val == 4.0:
        row['edu_4.0'] = 1

    if marri_val == 'Married_without_children':
        row['marri_status_Married_without_children'] = 1
    elif marri_val == 'Single':
        row['marri_status_Single'] = 1
    elif marri_val == 'Other':
        row['marri_status_Other'] = 1

    df_input = pd.DataFrame([row])[feature_cols]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])

    return df_input

# ── 예측 버튼 ─────────────────────────────────────────────────
if st.button("예측하기", type="primary", use_container_width=True):
    X = encode_input(age, sex, edu, marri, ainc, region)

    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    prob_poor = round(proba[0] * 100, 1)
    prob_good = round(proba[1] * 100, 1)

    st.subheader("예측 결과")

    if pred == 1:
        st.success(f"✅ 정상군 (Good CVH)  —  Good CVH 확률 {prob_good}%")
    else:
        st.error(f"⚠️ 위험군 (Poor CVH)  —  Poor CVH 확률 {prob_poor}%")

    st.progress(int(prob_good), text=f"Good CVH {prob_good}% / Poor CVH {prob_poor}%")

st.divider()
st.caption("※ 본 예측은 연구 목적으로만 활용하며 실제 의료 진단을 대체하지 않습니다.")