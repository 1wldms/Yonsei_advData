import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="CVD 위험도 예측", page_icon="🫀", layout="wide")

# ── 모델 로드 ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('cvd_model_single.pkl', 'rb') as f:
        single = pickle.load(f)
    with open('cvd_model_multi.pkl', 'rb') as f:
        multi = pickle.load(f)
    return single, multi

bundle_single, bundle_multi = load_models()

# ── 제목 ─────────────────────────────────────────────────────
st.title("🫀 심혈관 질환(CVD) 위험도 예측")
st.caption("국민건강영양조사 7·8기 (2016–2021) · 50세 이상 · LDA 모델")
st.divider()

# ── 가구 유형 선택 ────────────────────────────────────────────
st.subheader("가구 유형 선택")
household = st.radio(
    "본인의 가구 유형을 선택해주세요.",
    ["1인가구", "다인가구"],
    horizontal=True,
)
bundle = bundle_single if household == "1인가구" else bundle_multi
model        = bundle['model']
scaler       = bundle['scaler']
feature_cols = bundle['feature_columns']
scale_cols   = bundle['scale_cols']

st.divider()

# ── 입력 ─────────────────────────────────────────────────────
st.subheader("📋 개인 정보 입력")

# ── 섹션 1: 인구사회학적 정보 ─────────────────────────────────
with st.expander("① 인구사회학적 정보", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("연령 (세)", min_value=50, max_value=100, value=60)
        sex = st.selectbox("성별", ["남성", "여성"])
    with c2:
        edu = st.selectbox("학력", ["중졸 이하", "고졸", "전문대졸", "대졸 이상"])
        occp = st.selectbox("직업", ["무직/기타", "화이트칼라", "기타직종"])
    with c3:
        ainc = st.number_input("월평균 가구 소득 (만원)", min_value=0, max_value=2000, value=300, step=10)
        region = st.selectbox("거주지역", ["수도권", "비수도권"])

# ── 섹션 2: 신체 계측 및 임상 수치 ──────────────────────────────
with st.expander("② 신체 계측 및 임상 수치", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        HE_HP   = st.number_input("수축기 혈압 (mmHg)", min_value=60, max_value=250, value=120)
        HE_obe  = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=23.0, step=0.1)
    with c2:
        HE_chol = st.number_input("총콜레스테롤 (mg/dL)", min_value=50, max_value=600, value=190)
        HE_glu  = st.number_input("공복혈당 (mg/dL)", min_value=50, max_value=500, value=95)
    with c3:
        st.markdown("**동반 질환 여부**")
        dyslipidemia = st.checkbox("이상지질혈증")
        diabetes     = st.checkbox("당뇨")
        arthritis    = st.checkbox("관절염")
        tuberculosis = st.checkbox("결핵")
        asthma       = st.checkbox("천식")
        thyroid      = st.checkbox("갑상선질환")
        cancer       = st.checkbox("암")

# ── 섹션 3: 생활 습관 ─────────────────────────────────────────
with st.expander("③ 생활 습관", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        pa_aerobic   = st.selectbox("유산소 신체활동 충족 여부", ["미충족", "충족"])
        smoking_cvh  = st.selectbox("흡연 상태", ["비흡연", "과거흡연", "현재흡연"])
    with c2:
        BD1_11 = st.selectbox(
            "음주 빈도",
            ["최근 1년간 전혀 마시지 않음", "월 1회 미만", "월 1회 정도",
             "월 2~4회", "주 2~3회", "주 4회 이상"]
        )
        BD2_1  = st.number_input("1회 음주량 (잔)", min_value=0, max_value=30, value=2)
    with c3:
        mh_stress    = st.selectbox("스트레스 인지 수준", ["거의 없음", "조금 있음", "많이 있음", "매우 많이 있음"])
        sleep_time   = st.number_input("수면 시간 (시간/일)", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
        sitting_time = st.number_input("좌식 시간 (시간/일)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)

# ── 섹션 4: 영양 섭취 ─────────────────────────────────────────
with st.expander("④ 영양 섭취 (1일 기준)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        N_EN   = st.number_input("에너지 섭취 (kcal)", min_value=0, max_value=6000, value=1800)
        N_FAT  = st.number_input("지방 섭취 (g)", min_value=0.0, max_value=300.0, value=40.0, step=0.5)
    with c2:
        N_PROT = st.number_input("단백질 섭취 (g)", min_value=0.0, max_value=300.0, value=60.0, step=0.5)
        N_CHO  = st.number_input("탄수화물 섭취 (g)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
    with c3:
        N_NA   = st.number_input("나트륨 섭취 (mg)", min_value=0, max_value=20000, value=3000, step=100)
        N_TDF  = st.number_input("식이섬유 섭취 (g)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

# ── 인코딩 함수 ───────────────────────────────────────────────
def encode_input():
    # 범주형 → 수치
    sex_val       = 2.0 if sex == "여성" else 1.0
    edu_val       = {"중졸 이하": 1.0, "고졸": 2.0, "전문대졸": 3.0, "대졸 이상": 4.0}[edu]
    occp_val      = occp  # "무직/기타", "화이트칼라", "기타직종"
    region_val    = 1 if region == "수도권" else 0
    pa_val        = 1 if pa_aerobic == "충족" else 0
    smoke_val     = {"비흡연": 1, "과거흡연": 2, "현재흡연": 3}[smoking_cvh]
    bd1_val       = {"최근 1년간 전혀 마시지 않음": 1, "월 1회 미만": 2, "월 1회 정도": 3,
                     "월 2~4회": 4, "주 2~3회": 5, "주 4회 이상": 6}[BD1_11]
    stress_val    = {"거의 없음": 1, "조금 있음": 2, "많이 있음": 3, "매우 많이 있음": 4}[mh_stress]

    row = {col: 0 for col in feature_cols}

    # 연속형
    row['age']          = age
    row['ainc']         = ainc
    row['region_metro'] = region_val
    row['HE_HP']        = HE_HP
    row['HE_obe']       = HE_obe
    row['HE_chol']      = HE_chol
    row['HE_glu']       = HE_glu
    row['pa_aerobic']   = pa_val
    row['smoking_cvh']  = smoke_val
    row['BD1_11']       = bd1_val
    row['BD2_1']        = BD2_1
    row['mh_stress']    = stress_val
    row['sleep_time']   = sleep_time
    row['sitting_time'] = sitting_time
    row['N_EN']         = N_EN
    row['N_FAT']        = N_FAT
    row['N_PROT']       = N_PROT
    row['N_CHO']        = N_CHO
    row['N_NA']         = N_NA
    row['N_TDF']        = N_TDF

    # 동반 질환 (binary)
    row['dyslipidemia'] = int(dyslipidemia)
    row['arthritis']    = int(arthritis)
    row['tuberculosis'] = int(tuberculosis)
    row['asthma']       = int(asthma)
    row['thyroid']      = int(thyroid)
    row['diabetes']     = int(diabetes)
    row['cancer']       = int(cancer)

    # 성별 one-hot
    if 'sex' in feature_cols:
        row['sex'] = sex_val
    elif 'sex_2.0' in feature_cols and sex_val == 2.0:
        row['sex_2.0'] = 1

    # 학력 one-hot (edu_1.0 = base)
    for lv in [2.0, 3.0, 4.0]:
        key = f'edu_{lv}'
        if key in feature_cols:
            row[key] = 1 if edu_val == lv else 0

    # 직업 one-hot (무직/기타 = base)
    if 'occp_white' in feature_cols:
        row['occp_white'] = 1 if occp_val == "화이트칼라" else 0
    if 'occp_other' in feature_cols:
        row['occp_other'] = 1 if occp_val == "기타직종" else 0

    df_input = pd.DataFrame([row])[feature_cols]

    # 스케일링
    cols_to_scale = [c for c in scale_cols if c in df_input.columns]
    if cols_to_scale:
        df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])

    return df_input

# ── 예측 버튼 ─────────────────────────────────────────────────
st.divider()
if st.button("예측하기", type="primary", use_container_width=True):
    X = encode_input()

    pred = model.predict(X)[0]

    st.subheader("예측 결과")
    hh_label = "1인가구" if household == "1인가구" else "다인가구"
    st.caption(f"적용 모델: **{hh_label} LDA 모델**")

    if pred == 1:
        st.error("⚠️ 심혈관 질환 **위험군**으로 예측되었습니다.")
    else:
        st.success("✅ 심혈관 질환 **정상군**으로 예측되었습니다.")

st.divider()
st.caption("※ 본 예측은 연구 목적으로만 활용하며 실제 의료 진단을 대체하지 않습니다.")