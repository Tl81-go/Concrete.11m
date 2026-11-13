
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="배합 성분 탐색",
    page_icon="📊",
    layout="wide"
)

st.title("📊 배합 성분 탐색")
st.subheader("Concrete Ingredient Exploration")

st.markdown("""
콘크리트를 구성하는 8가지 재료는 각각 얼마나 사용되었을까요?

이 페이지에서는 실제 실험 데이터를 이용해
재료별 사용량의 분포와 압축강도와의 관계를 분석합니다.
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
# 3. 변수 설정
# ==========================================

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

unit_names = {
    "Cement": "kg/m³",
    "Blast Furnace Slag": "kg/m³",
    "Fly Ash": "kg/m³",
    "Water": "kg/m³",
    "Superplasticizer": "kg/m³",
    "Coarse Aggregate": "kg/m³",
    "Fine Aggregate": "kg/m³",
    "Age": "일"
}

features = list(ingredient_names.keys())

required_columns = features + ["Strength"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 숫자형 데이터로 변환
df[required_columns] = df[required_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.dropna(subset=required_columns)


# ==========================================
# 4. 전체 데이터 요약
# ==========================================

st.header("1. 배합 성분 전체 요약")

st.write(
    "각 배합 성분의 전체적인 사용량을 살펴봅니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("분석 데이터", f"{len(df):,}개")

with col2:
    st.metric("배합 성분", "8개")

with col3:
    st.metric(
        "평균 압축강도",
        f"{df['Strength'].mean():.2f} MPa"
    )

st.divider()


# ==========================================
# 5. 재료별 사용량 분포
# ==========================================

st.header("2. 재료별 사용량 분포")

st.markdown("""
재료마다 사용되는 양은 서로 다릅니다.

히스토그램을 통해 특정 사용량 구간에
실험 데이터가 얼마나 많이 모여 있는지 확인할 수 있습니다.
""")

selected_feature = st.selectbox(
    "분석할 배합 성분을 선택하세요.",
    features,
    format_func=lambda x: ingredient_names[x]
)

selected_korean = ingredient_names[selected_feature]
selected_unit = unit_names[selected_feature]

fig_hist = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    title=f"{selected_korean} 사용량 분포",
    labels={
        selected_feature: f"{selected_korean} ({selected_unit})",
        "count": "데이터 개수"
    },
    marginal="box"
)

fig_hist.update_layout(
    xaxis_title=f"{selected_korean} ({selected_unit})",
    yaxis_title="데이터 개수",
    showlegend=False
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# 선택한 재료의 통계량
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "평균",
        f"{df[selected_feature].mean():.2f}"
    )

with col2:
    st.metric(
        "중앙값",
        f"{df[selected_feature].median():.2f}"
    )

with col3:
    st.metric(
        "최솟값",
        f"{df[selected_feature].min():.2f}"
    )

with col4:
    st.metric(
        "최댓값",
        f"{df[selected_feature].max():.2f}"
    )

st.caption(f"단위: {selected_unit}")

st.info(
    f"이 그래프는 전체 실험 중 {selected_korean}이 "
    "어느 정도의 양으로 사용되었는지 보여줍니다."
)

st.divider()


# ==========================================
# 6. 재료별 상자그림
# ==========================================

st.header("3. 재료별 사용량 비교")

st.markdown("""
상자그림(Box Plot)은 데이터의 중앙값과 사분위수,
분포 범위를 살펴보는 그래프입니다.

재료의 사용량이 얼마나 다양하게 나타나는지 확인해 봅시다.
""")

box_feature = st.selectbox(
    "상자그림을 볼 재료를 선택하세요.",
    features,
    format_func=lambda x: ingredient_names[x],
    key="box_feature"
)

fig_box = px.box(
    df,
    y=box_feature,
    points="outliers",
    title=f"{ingredient_names[box_feature]} 사용량의 상자그림",
    labels={
        box_feature: f"{ingredient_names[box_feature]} ({unit_names[box_feature]})"
    }
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)

st.info("""
상자 가운데 선은 중앙값을 의미합니다.
상자의 위아래 끝은 각각 3사분위수와 1사분위수입니다.
상자 밖의 점은 일반적인 분포에서 벗어난 관측값일 수 있습니다.
""")

st.divider()


# ==========================================
# 7. 재료와 압축강도의 관계
# ==========================================

st.header("4. 재료와 압축강도의 관계")

st.markdown("""
선택한 재료의 사용량과 압축강도가 어떤 관계를 가지는지
산점도로 확인합니다.

각 점은 하나의 콘크리트 실험 데이터를 나타냅니다.
""")

relationship_feature = st.selectbox(
    "압축강도와 비교할 재료를 선택하세요.",
    features,
    format_func=lambda x: ingredient_names[x],
    key="relationship_feature"
)

fig_scatter = px.scatter(
    df,
    x=relationship_feature,
    y="Strength",
    opacity=0.65,
    title=f"{ingredient_names[relationship_feature]}과 압축강도의 관계",
    labels={
        relationship_feature: (
            f"{ingredient_names[relationship_feature]} "
            f"({unit_names[relationship_feature]})"
        ),
        "Strength": "압축강도 (MPa)"
    },
    hover_data=features
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

correlation = df[relationship_feature].corr(df["Strength"])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

with col2:
    if correlation > 0:
        direction = "양의 상관관계"
    elif correlation < 0:
        direction = "음의 상관관계"
    else:
        direction = "선형 상관관계가 거의 없음"

    st.metric("관계 방향", direction)

st.info("""
상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다.

+1에 가까우면 양의 상관관계,
-1에 가까우면 음의 상관관계,
0에 가까우면 선형 상관관계가 약하다는 의미입니다.

상관관계가 있다고 해서 한 변수가 다른 변수의 원인이라고
단정할 수는 없습니다.
""")

st.divider()


# ==========================================
# 8. 변수 간 상관관계
# ==========================================

st.header("5. 전체 변수의 상관관계")

st.markdown("""
8가지 배합 성분과 압축강도 사이의 관계를
한꺼번에 비교해 봅니다.
""")

corr = df[required_columns].corr()

korean_names = {
    **ingredient_names,
    "Strength": "압축강도"
}

corr_display = corr.rename(
    index=korean_names,
    columns=korean_names
)

fig_heatmap = px.imshow(
    corr_display,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="배합 성분과 압축강도의 상관관계 히트맵"
)

fig_heatmap.update_layout(
    height=700,
    coloraxis_colorbar_title="상관계수"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

st.caption("""
빨간색 계열은 양의 상관관계,
파란색 계열은 음의 상관관계를 의미합니다.
색이 진할수록 상관관계의 절댓값이 큽니다.
""")

st.divider()


# ==========================================
# 9. 압축강도와 가장 관련이 높은 변수
# ==========================================

st.header("6. 압축강도와 관련된 변수 순위")

strength_corr = (
    corr["Strength"]
    .drop("Strength")
    .sort_values(key=abs, ascending=False)
)

rank_df = pd.DataFrame({
    "변수": [ingredient_names[x] for x in strength_corr.index],
    "상관계수": strength_corr.values.round(3),
    "상관관계": [
        "양의 상관관계" if x > 0
        else "음의 상관관계" if x < 0
        else "거의 없음"
        for x in strength_corr.values
    ]
})

st.dataframe(
    rank_df,
    use_container_width=True,
    hide_index=True
)

strongest = strength_corr.index[0]

st.success(
    f"이 데이터에서 압축강도와 절댓값 기준으로 "
    f"가장 높은 선형 상관관계를 보이는 변수는 "
    f"'{ingredient_names[strongest]}'입니다. "
    f"(상관계수: {strength_corr.iloc[0]:.3f})"
)

st.warning("""
이 순위는 상관관계에 따른 순위이지,
각 재료가 압축강도에 미치는 실제 인과적 영향의 순위는 아닙니다.
""")

st.divider()


# ==========================================
# 10. 마무리
# ==========================================

st.header("🔍 탐색 결과 정리")

st.markdown("""
이번 페이지에서는 다음 내용을 살펴보았습니다.

1. 각 배합 성분의 사용량 분포
2. 재료별 통계량과 상자그림
3. 각 재료와 압축강도의 관계
4. 전체 변수 사이의 상관관계
5. 압축강도와 관련된 변수의 순위

다음 페이지에서는 이 결과를 화학공학적인 관점에서
더 깊이 분석해 보겠습니다.
""")

st.success("배합 성분 탐색을 완료했습니다!")
