
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="데이터 알아보기 | 콘크리트 압축강도 예측 실험실",
    page_icon="📚",
    layout="wide"
)

st.title("📚 데이터 알아보기")
st.subheader("Concrete Compressive Strength Dataset")

st.markdown("""
콘크리트 압축강도를 예측하기 전에 먼저 데이터를 이해해 봅시다.

이 페이지에서는 데이터의 구성, 변수의 의미,
기초통계량과 결측치 등을 살펴봅니다.
""")

st.divider()


# ==========================================
# 2. 데이터 불러오기
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Tl81-go/Concrete.11m/main/data/concrete.csv"
)


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Concrete compressive strength": "Strength"
    })

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. 변수 목록
# ==========================================

variable_info = {
    "Cement": ["시멘트", "kg/m³", "콘크리트의 결합재 역할을 하는 재료"],
    "Blast Furnace Slag": ["고로 슬래그", "kg/m³", "철강 생산 과정에서 발생하는 부산물"],
    "Fly Ash": ["플라이 애시", "kg/m³", "석탄 화력발전 과정에서 발생하는 미분말"],
    "Water": ["물", "kg/m³", "시멘트 수화 반응에 필요한 물"],
    "Superplasticizer": ["고성능 감수제", "kg/m³", "물 사용량을 줄이면서 유동성을 높이는 혼화제"],
    "Coarse Aggregate": ["굵은 골재", "kg/m³", "콘크리트의 골격을 형성하는 큰 입자의 골재"],
    "Fine Aggregate": ["잔골재", "kg/m³", "굵은 골재 사이를 채우는 작은 입자의 골재"],
    "Age": ["양생 기간", "일(day)", "콘크리트가 양생된 기간"],
    "Strength": ["압축강도", "MPa", "콘크리트가 압축력에 견디는 정도"]
}

required_columns = list(variable_info.keys())

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


# ==========================================
# 4. 데이터 기본 정보
# ==========================================

st.header("1. 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 데이터 수", f"{len(df):,}개")

with col2:
    st.metric("입력 변수", "8개")

with col3:
    st.metric("예측 변수", "1개")

with col4:
    st.metric("전체 변수", f"{len(required_columns)}개")

st.markdown("""
이 데이터는 UCI Machine Learning Repository에서 제공하는
Concrete Compressive Strength Dataset입니다.

총 1,030개의 실험 데이터로 구성되어 있으며,
8개의 독립 변수로 1개의 종속 변수인 압축강도를 예측합니다.
""")

st.divider()


# ==========================================
# 5. 변수의 의미
# ==========================================

st.header("2. 변수의 의미")

st.write(
    "표에서 각 변수의 영어 이름, 한국어 이름, 단위와 의미를 확인할 수 있습니다."
)

info_df = pd.DataFrame([
    {
        "영어 변수명": name,
        "한국어 이름": info[0],
        "단위": info[1],
        "변수의 의미": info[2],
        "구분": "종속 변수" if name == "Strength" else "독립 변수"
    }
    for name, info in variable_info.items()
])

st.dataframe(
    info_df,
    use_container_width=True,
    hide_index=True
)

st.info("""
독립 변수 8개는 압축강도를 예측하기 위한 입력값이고,
종속 변수 Strength는 모델이 예측해야 하는 결과값입니다.
""")

st.divider()


# ==========================================
# 6. 결측치 확인
# ==========================================

st.header("3. 결측치 확인")

st.write("""
결측치는 데이터에서 값이 비어 있는 상태를 의미합니다.

결측치가 많으면 분석이나 머신러닝 모델 학습에
문제가 생길 수 있으므로 미리 확인해야 합니다.
""")

missing_df = pd.DataFrame({
    "변수명": df.columns,
    "결측치 개수": df.isnull().sum().values,
    "결측치 비율(%)": (
        df.isnull().mean().values * 100
    ).round(2)
})

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 결측치",
        f"{int(df.isnull().sum().sum()):,}개"
    )

with col2:
    st.metric(
        "결측치가 있는 변수",
        f"{int((df.isnull().sum() > 0).sum())}개"
    )

st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)

if df.isnull().sum().sum() == 0:
    st.success("결측치가 없습니다. 모든 변수에 값이 존재합니다.")
else:
    st.warning("결측치가 발견되었습니다. 분석 전에 처리 방법을 결정해야 합니다.")

st.divider()


# ==========================================
# 7. 기초통계량
# ==========================================

st.header("4. 기초통계량")

st.write("""
각 변수의 평균, 표준편차, 최솟값, 최댓값 등을 확인합니다.

이 통계량을 이용하면 재료가 어느 정도의 양으로 사용되었는지,
실험 데이터가 어느 범위에 분포하는지 알 수 있습니다.
""")

stats_df = df[required_columns].describe().T

stats_df = stats_df.rename(columns={
    "count": "데이터 수",
    "mean": "평균",
    "std": "표준편차",
    "min": "최솟값",
    "25%": "1사분위수",
    "50%": "중앙값",
    "75%": "3사분위수",
    "max": "최댓값"
})

stats_df.index.name = "변수명"

st.dataframe(
    stats_df.round(2),
    use_container_width=True
)

st.caption("""
표준편차는 데이터가 평균을 중심으로 얼마나 퍼져 있는지를 나타냅니다.
표준편차가 클수록 데이터의 변동성이 큰 편입니다.
""")

st.divider()


# ==========================================
# 8. 압축강도 분포
# ==========================================

st.header("5. 압축강도 분포")

st.write("""
1,030개의 실험에서 측정된 압축강도가
어떤 범위에 많이 분포하는지 살펴봅니다.
""")

fig = px.histogram(
    df,
    x="Strength",
    nbins=30,
    title="콘크리트 압축강도 분포",
    labels={
        "Strength": "압축강도 (MPa)",
        "count": "데이터 개수"
    },
    marginal="box"
)

fig.update_layout(
    xaxis_title="압축강도 (MPa)",
    yaxis_title="데이터 개수",
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "평균 압축강도",
        f"{df['Strength'].mean():.2f} MPa"
    )

with col2:
    st.metric(
        "최저 압축강도",
        f"{df['Strength'].min():.2f} MPa"
    )

with col3:
    st.metric(
        "최고 압축강도",
        f"{df['Strength'].max():.2f} MPa"
    )

st.divider()


# ==========================================
# 9. 실제 데이터 확인
# ==========================================

st.header("6. 실제 데이터 미리보기")

st.write("""
실제 실험에서 측정된 콘크리트 배합 성분과
압축강도 데이터를 확인할 수 있습니다.
""")

show_rows = st.slider(
    "표시할 데이터 개수",
    min_value=5,
    max_value=len(df),
    value=10,
    step=5
)

st.dataframe(
    df[required_columns].head(show_rows),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"전체 {len(df):,}개 데이터 중 처음 {show_rows:,}개를 표시하고 있습니다."
)


# ==========================================
# 10. 마무리
# ==========================================

st.divider()

st.success("""
데이터 알아보기를 완료했습니다!

이제 각 배합 성분이 어떤 범위에 분포하는지 확인했고,
압축강도의 전체적인 분포도 살펴보았습니다.

다음 단계에서는 각 재료의 분포와 재료 사이의 관계를
더 자세하게 분석해 보겠습니다.
""")
