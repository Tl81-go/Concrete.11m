
import streamlit as st
import pandas as pd

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="콘크리트 압축강도 예측 실험실 조현서를 위한",
    page_icon="🏗️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Tl81-go/Concrete.11m/main/data/concrete.csv"
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)

    # 열 이름 앞뒤 공백 제거
    df.columns = df.columns.str.strip()

    # 원본 데이터의 압축강도 열 이름을 통일
    df = df.rename(columns={
        "Concrete compressive strength": "Strength"
    })

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("GitHub의 CSV 파일 위치와 이름을 확인해 주세요.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. 데이터 열 확인
# ==========================================

features = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]

target = "Strength"

required_columns = features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error("CSV 파일의 열 이름이 예상과 다릅니다.")
    st.write("찾지 못한 열:", missing_columns)
    st.write("현재 CSV 열 이름:", df.columns.tolist())
    st.stop()


# ==========================================
# 4. 제목 및 소개
# ==========================================

st.title("🏗️ 콘크리트 압축강도 예측 실험실")

st.subheader(
    "Concrete Compressive Strength Prediction Lab"
)

st.markdown("""
콘크리트의 압축강도는 어떤 재료를 얼마나 넣느냐에 따라 달라집니다.

이 실험실에서는 콘크리트 배합 성분 데이터를 분석하고,
머신러닝 모델을 활용하여 콘크리트의 압축강도를 예측합니다.

또한 배합 성분과 압축강도의 관계를 화학공학적 관점에서
살펴보고, 가상의 배합 실험도 진행할 수 있습니다.
""")

st.divider()


# ==========================================
# 5. 실험 데이터 소개
# ==========================================

st.header("📊 실험 데이터")

st.write(
    "UCI Machine Learning Repository의 "
    "Concrete Compressive Strength Dataset을 사용합니다."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 데이터", f"{len(df):,}개")

with col2:
    st.metric("입력 변수", f"{len(features)}개")

with col3:
    st.metric("예측 변수", "1개")

with col4:
    st.metric("전체 열", f"{len(df.columns)}개")

st.divider()


# ==========================================
# 6. 콘크리트 배합 성분
# ==========================================

st.header("🧪 콘크리트 배합 성분")

st.write(
    "다음 8가지 재료 및 조건을 입력하여 "
    "콘크리트의 압축강도를 예측합니다."
)

ingredient_names = {
    "Cement": "시멘트",
    "Blast Furnace Slag": "고로 슬래그",
    "Fly Ash": "플라이 애시",
    "Water": "물",
    "Superplasticizer": "고성능 감수제",
    "Coarse Aggregate": "굵은 골재",
    "Fine Aggregate": "잔골재",
    "Age": "양생 기간"
}

col1, col2 = st.columns(2)

for i, (eng, kor) in enumerate(ingredient_names.items()):
    if i < 4:
        with col1:
            st.markdown(f"- **{kor}** ({eng})")
    else:
        with col2:
            st.markdown(f"- **{kor}** ({eng})")

st.caption(
    "재료의 양은 kg/m³, 양생 기간은 일(day) 단위입니다."
)

st.divider()


# ==========================================
# 7. 예측 목표
# ==========================================

st.header("🎯 예측 목표")

st.markdown("""
### 콘크리트 압축강도 (Strength)

8가지 배합 성분과 양생 기간을 바탕으로
콘크리트의 압축강도를 예측합니다.

- 예측 대상: Strength
- 단위: MPa (메가파스칼)
- 분석 방법: 머신러닝 회귀 분석
""")

st.divider()


# ==========================================
# 8. 실험실 메뉴 안내
# ==========================================

st.header("🔬 실험실 이용 안내")

pages_info = [
    ("🏠", "홈", "실험실 소개 및 전체 데이터 요약"),
    ("📚", "데이터 알아보기", "데이터 구조와 변수의 의미 확인"),
    ("📊", "배합 성분 탐색", "각 재료의 분포와 성분 간 관계 분석"),
    ("⚗️", "화학공학적 분석", "물-시멘트비와 배합 성분의 영향 탐색"),
    ("🔮", "압축강도 예측", "배합 성분을 입력하여 압축강도 예측"),
    ("📈", "모델 성능 평가", "세 가지 머신러닝 모델의 성능 비교"),
    ("🧪", "가상 배합 실험", "배합 조건을 바꾸며 예측 결과 비교")
]

for icon, name, description in pages_info:
    with st.container(border=True):
        st.markdown(f"### {icon} {name}")
        st.write(description)


# ==========================================
# 9. 데이터 미리보기
# ==========================================

with st.expander("데이터 미리보기"):
    st.dataframe(df.head(10), use_container_width=True)


# ==========================================
# 10. 하단 안내
# ==========================================

st.divider()

st.caption(
    "콘크리트 압축강도 예측 실험실 | "
    "Concrete Compressive Strength Prediction Lab"
)

st.caption(
    "교육 및 데이터 분석 목적으로 제작된 실험용 프로그램입니다."
)
