
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="화학공학적 분석 | 콘크리트 강도 예측 실험실",
    page_icon="🧪",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Tl81-go/Concrete.11m/main/data/concrete.csv"
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)

    # 칼럼 이름 앞뒤의 공백 제거
    df.columns = df.columns.str.strip()

    # 압축강도 칼럼 이름 통일
    if "Concrete compressive strength" in df.columns:
        df = df.rename(
            columns={"Concrete compressive strength": "Strength"}
        )

    # 숫자형 데이터로 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("오류 내용:", e)
    st.stop()


# 필요한 칼럼 확인
required_columns = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
    "Strength"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("CSV 파일에서 다음 칼럼을 찾을 수 없습니다.")
    st.write(missing_columns)
    st.stop()


# 분석에 필요한 숫자 데이터만 사용
df = df[required_columns].dropna().copy()


# --------------------------------------------------
# 3. 파생 변수 만들기
# --------------------------------------------------

# 결합재 전체 양
df["Binder"] = (
    df["Cement"]
    + df["Blast Furnace Slag"]
    + df["Fly Ash"]
)

# 물-시멘트비(W/C)
df["W_C_Ratio"] = (
    df["Water"] / df["Cement"].where(df["Cement"] > 0)
)

# 물-결합재비(W/B)
df["W_B_Ratio"] = (
    df["Water"] / df["Binder"].where(df["Binder"] > 0)
)

# 결합재 구성 비율(%)
df["Cement_Ratio"] = (
    df["Cement"] / df["Binder"].where(df["Binder"] > 0) * 100
)

df["Slag_Ratio"] = (
    df["Blast Furnace Slag"]
    / df["Binder"].where(df["Binder"] > 0) * 100
)

df["FlyAsh_Ratio"] = (
    df["Fly Ash"] / df["Binder"].where(df["Binder"] > 0) * 100
)

# 골재 전체 양
df["Total_Aggregate"] = (
    df["Coarse Aggregate"] + df["Fine Aggregate"]
)

# 굵은 골재 비율(%)
df["Coarse_Ratio"] = (
    df["Coarse Aggregate"]
    / df["Total_Aggregate"].where(df["Total_Aggregate"] > 0) * 100
)

# 잔골재 비율(%)
df["Fine_Ratio"] = (
    df["Fine Aggregate"]
    / df["Total_Aggregate"].where(df["Total_Aggregate"] > 0) * 100
)

# 고성능 감수제 사용 여부
df["Superplasticizer_Use"] = df["Superplasticizer"].apply(
    lambda x: "사용" if x > 0 else "미사용"
)

# 물-시멘트비, 물-결합재비 계산이 가능한 데이터
ratio_df = df.dropna(
    subset=["W_C_Ratio", "W_B_Ratio"]
).copy()


# --------------------------------------------------
# 4. 페이지 제목
# --------------------------------------------------

st.title("🧪 콘크리트의 화학공학적 분석")

st.write(
    """
    콘크리트의 압축강도는 단순히 시멘트의 양만으로 결정되지 않습니다.

    물과 결합재의 비율, 혼화재의 구성, 골재의 배합,
    고성능 감수제의 사용, 양생 기간 등이 서로 영향을 줍니다.

    이 페이지에서는 배합 데이터를 여러 가지 화학공학적 관점에서 분석합니다.
    """
)

st.info(
    "주의: 아래 분석은 관찰된 데이터의 관계를 살펴보는 것입니다. "
    "다른 조건이 함께 달라질 수 있으므로 그래프의 상관관계를 "
    "곧바로 원인과 결과로 해석해서는 안 됩니다."
)

st.divider()


# --------------------------------------------------
# 5. 물-시멘트비(W/C) 분석
# --------------------------------------------------

st.header("1. 물-시멘트비(W/C) 분석")

st.write(
    """
    물-시멘트비(W/C)는 물의 질량을 시멘트의 질량으로 나눈 값입니다.

    일반적으로 물-시멘트비가 높으면 시멘트가 굳은 뒤 내부에
    모세관 공극이 더 많이 남을 수 있어 압축강도가 낮아지는 경향이 있습니다.
    """
)

st.latex(r"W/C = \frac{Water}{Cement}")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        ratio_df,
        x="W_C_Ratio",
        nbins=30,
        title="물-시멘트비 분포",
        labels={"W_C_Ratio": "물-시멘트비(W/C)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        ratio_df,
        x="W_C_Ratio",
        y="Strength",
        title="물-시멘트비와 압축강도의 관계",
        labels={
            "W_C_Ratio": "물-시멘트비(W/C)",
            "Strength": "압축강도(MPa)"
        },
        hover_data=["Cement", "Water", "Age"]
    )
    st.plotly_chart(fig, use_container_width=True)

# 물-시멘트비 구간별 평균 강도
wc_bins = pd.cut(
    ratio_df["W_C_Ratio"],
    bins=8
)

wc_summary = (
    ratio_df.groupby(wc_bins, observed=False)["Strength"]
    .agg(["mean", "count"])
    .reset_index()
)

wc_summary.columns = ["물-시멘트비 구간", "평균 압축강도", "데이터 수"]
wc_summary["물-시멘트비 구간"] = wc_summary["물-시멘트비 구간"].astype(str)

fig = px.bar(
    wc_summary,
    x="물-시멘트비 구간",
    y="평균 압축강도",
    title="물-시멘트비 구간별 평균 압축강도",
    hover_data=["데이터 수"],
    labels={
        "물-시멘트비 구간": "물-시멘트비 구간",
        "평균 압축강도": "평균 압축강도(MPa)"
    }
)
st.plotly_chart(fig, use_container_width=True)

wc_corr = ratio_df["W_C_Ratio"].corr(ratio_df["Strength"])

st.metric("물-시멘트비와 압축강도의 상관계수", f"{wc_corr:.3f}")

st.markdown(
    """
    **화학공학적 해석**

    - 물은 시멘트의 수화반응에 필요합니다.
    - 하지만 반응에 필요한 양보다 물이 지나치게 많으면,
      굳은 콘크리트 내부에 공극이 남을 수 있습니다.
    - 공극이 많아지면 압축하중을 견디는 능력이 낮아질 수 있습니다.
    - 다만 실제 강도는 시멘트 종류, 혼화재, 양생 조건 등에도 영향을 받습니다.
    """
)

st.divider()


# --------------------------------------------------
# 6. 물-결합재비(W/B) 분석
# --------------------------------------------------

st.header("2. 물-결합재비(W/B) 분석")

st.write(
    """
    물-결합재비(W/B)는 물의 질량을 전체 결합재의 질량으로 나눈 값입니다.

    여기서 결합재는 시멘트, 고로슬래그, 플라이애시를 합한 양입니다.
    """
)

st.latex(
    r"W/B = \frac{Water}{Cement + Slag + Fly\ Ash}"
)

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        ratio_df,
        x="W_B_Ratio",
        y="Strength",
        title="물-결합재비와 압축강도",
        labels={
            "W_B_Ratio": "물-결합재비(W/B)",
            "Strength": "압축강도(MPa)"
        },
        hover_data=["Cement", "Blast Furnace Slag", "Fly Ash"]
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        ratio_df,
        x="W_C_Ratio",
        y="W_B_Ratio",
        title="물-시멘트비와 물-결합재비 비교",
        labels={
            "W_C_Ratio": "물-시멘트비(W/C)",
            "W_B_Ratio": "물-결합재비(W/B)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

wb_corr = ratio_df["W_B_Ratio"].corr(ratio_df["Strength"])

st.metric("물-결합재비와 압축강도의 상관계수", f"{wb_corr:.3f}")

st.markdown(
    """
    **화학공학적 해석**

    - 고로슬래그와 플라이애시도 결합재에 포함됩니다.
    - 따라서 시멘트만 기준으로 계산하는 W/C와
      모든 결합재를 기준으로 계산하는 W/B는 서로 다른 정보를 줍니다.
    - 혼화재가 많이 포함된 배합에서는 W/B가 배합 특성을
      이해하는 데 더 유용할 수 있습니다.
    - 혼화재의 반응 속도와 양생 조건도 강도에 영향을 줍니다.
    """
)

st.divider()


# --------------------------------------------------
# 7. 결합재 구성 비율 분석
# --------------------------------------------------

st.header("3. 결합재 구성 비율 분석")

st.write(
    """
    결합재는 시멘트, 고로슬래그, 플라이애시로 구성됩니다.

    전체 결합재에서 각각의 재료가 차지하는 비율을 계산하면
    배합에서 어떤 재료가 중심이 되는지 비교할 수 있습니다.
    """
)

st.latex(
    r"재료\ 비율(\%) = \frac{재료의\ 질량}{전체\ 결합재\ 질량} \times 100"
)

binder_ratio_df = df.dropna(
    subset=["Cement_Ratio", "Slag_Ratio", "FlyAsh_Ratio"]
).copy()

binder_mean = pd.DataFrame({
    "재료": ["시멘트", "고로슬래그", "플라이애시"],
    "평균 구성 비율(%)": [
        binder_ratio_df["Cement_Ratio"].mean(),
        binder_ratio_df["Slag_Ratio"].mean(),
        binder_ratio_df["FlyAsh_Ratio"].mean()
    ]
})

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        binder_mean,
        names="재료",
        values="평균 구성 비율(%)",
        title="전체 데이터의 평균 결합재 구성 비율",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        binder_ratio_df,
        x="Slag_Ratio",
        y="Strength",
        color="FlyAsh_Ratio",
        title="고로슬래그 비율과 압축강도",
        labels={
            "Slag_Ratio": "고로슬래그 비율(%)",
            "FlyAsh_Ratio": "플라이애시 비율(%)",
            "Strength": "압축강도(MPa)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **화학공학적 해석**

    - 시멘트는 물과 반응하여 콘크리트의 강도 발현에 기여합니다.
    - 고로슬래그와 플라이애시는 시멘트 일부를 대체할 수 있는 혼화재입니다.
    - 이 재료들은 수화반응 및 포졸란 반응 등에 관여할 수 있습니다.
    - 재료별 반응 속도와 양생 기간이 다르므로 구성 비율만으로
      강도가 높거나 낮다고 단정할 수는 없습니다.
    """
)

st.divider()


# --------------------------------------------------
# 8. 골재 구성 분석
# --------------------------------------------------

st.header("4. 골재 구성 분석")

st.write(
    """
    콘크리트에는 굵은 골재와 잔골재가 함께 사용됩니다.

    두 골재의 총량과 구성 비율을 살펴보면
    배합에서 골재가 차지하는 비중을 비교할 수 있습니다.
    """
)

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        df,
        x="Total_Aggregate",
        y="Strength",
        title="전체 골재량과 압축강도",
        labels={
            "Total_Aggregate": "전체 골재량(kg/m³)",
            "Strength": "압축강도(MPa)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        df,
        x="Coarse_Ratio",
        y="Strength",
        color="Fine_Ratio",
        title="굵은 골재 비율과 압축강도",
        labels={
            "Coarse_Ratio": "굵은 골재 비율(%)",
            "Fine_Ratio": "잔골재 비율(%)",
            "Strength": "압축강도(MPa)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **화학공학적 해석**

    - 골재는 콘크리트의 뼈대 역할을 합니다.
    - 굵은 골재와 잔골재의 크기 분포 및 배합 비율은
      입자 사이의 빈 공간과 충전 상태에 영향을 줄 수 있습니다.
    - 골재의 양과 비율뿐 아니라 입자 모양, 표면 상태,
      입도 분포도 중요합니다.
    - 이 데이터에는 입자 크기 분포와 표면 특성이 없으므로
      골재 비율만으로 모든 특성을 설명할 수는 없습니다.
    """
)

st.divider()


# --------------------------------------------------
# 9. 고성능 감수제 분석
# --------------------------------------------------

st.header("5. 고성능 감수제와 물의 관계")

st.write(
    """
    고성능 감수제는 시멘트 입자들이 뭉치는 현상을 줄여
    콘크리트의 유동성을 높이는 혼화제입니다.

    이를 이용하면 작업성을 유지하면서 물의 사용량을
    줄이는 배합을 설계할 수 있습니다.
    """
)

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        df,
        x="Superplasticizer",
        y="Water",
        color="Strength",
        title="고성능 감수제 사용량과 물의 양",
        labels={
            "Superplasticizer": "고성능 감수제(kg/m³)",
            "Water": "물(kg/m³)",
            "Strength": "압축강도(MPa)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df,
        x="Superplasticizer_Use",
        y="Water",
        color="Superplasticizer_Use",
        title="감수제 사용 여부에 따른 물 사용량",
        labels={
            "Superplasticizer_Use": "감수제 사용 여부",
            "Water": "물(kg/m³)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

fig = px.box(
    df,
    x="Superplasticizer_Use",
    y="Strength",
    color="Superplasticizer_Use",
    title="감수제 사용 여부에 따른 압축강도",
    labels={
        "Superplasticizer_Use": "감수제 사용 여부",
        "Strength": "압축강도(MPa)"
    }
)
st.plotly_chart(fig, use_container_width=True)

sp_summary = (
    df.groupby("Superplasticizer_Use", observed=False)
    .agg(
        평균_물_사용량=("Water", "mean"),
        평균_압축강도=("Strength", "mean"),
        배합_개수=("Strength", "count")
    )
    .reset_index()
)

sp_summary = sp_summary.rename(columns={
    "Superplasticizer_Use": "감수제 사용 여부",
    "평균_물_사용량": "평균 물 사용량(kg/m³)",
    "평균_압축강도": "평균 압축강도(MPa)",
    "배합_개수": "배합 개수"
})

st.dataframe(sp_summary, use_container_width=True, hide_index=True)

st.markdown(
    """
    **화학공학적 해석**

    - 감수제는 시멘트 입자 사이의 분산을 도와 유동성을 개선합니다.
    - 적절히 사용하면 물을 줄이면서도 작업성을 확보하는 데 도움이 됩니다.
    - 다만 감수제를 사용한 배합과 사용하지 않은 배합은
      시멘트 양이나 다른 재료의 양도 다를 수 있습니다.
    - 따라서 두 집단의 평균 차이를 감수제 하나의 효과라고
      단정해서는 안 됩니다.
    """
)

st.divider()


# --------------------------------------------------
# 10. 양생 기간과 압축강도 분석
# --------------------------------------------------

st.header("6. 양생 기간과 압축강도 분석")

st.write(
    """
    콘크리트는 물과 시멘트의 수화반응을 통해 시간이 지나면서
    강도를 발현합니다.

    일반적으로 초기에는 강도가 빠르게 증가하고,
    시간이 지날수록 강도 증가 속도가 느려지는 경향이 있습니다.
    """
)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="Age",
        nbins=30,
        title="양생 기간 분포",
        labels={"Age": "양생 기간(일)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        df,
        x="Age",
        y="Strength",
        title="양생 기간과 압축강도의 관계",
        labels={
            "Age": "양생 기간(일)",
            "Strength": "압축강도(MPa)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# 양생 기간 구간별 평균 강도
age_bins = [0, 7, 28, 90, float("inf")]
age_labels = ["1~7일", "8~28일", "29~90일", "91일 이상"]

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels
)

age_summary = (
    df.groupby("Age_Group", observed=False)["Strength"]
    .agg(["mean", "count"])
    .reset_index()
)

age_summary.columns = ["양생 기간", "평균 압축강도", "배합 개수"]

fig = px.bar(
    age_summary,
    x="양생 기간",
    y="평균 압축강도",
    title="양생 기간 구간별 평균 압축강도",
    hover_data=["배합 개수"],
    labels={"평균 압축강도": "평균 압축강도(MPa)"}
)
st.plotly_chart(fig, use_container_width=True)

age_corr = df["Age"].corr(df["Strength"])

st.metric("양생 기간과 압축강도의 상관계수", f"{age_corr:.3f}")

st.markdown(
    """
    **화학공학적 해석**

    - 시멘트의 수화반응은 시간이 지나면서 진행됩니다.
    - 수화 생성물이 형성되면서 콘크리트 내부 구조가 치밀해지고
      강도가 발현될 수 있습니다.
    - 초기와 장기 양생에서 강도 증가 양상은 다를 수 있습니다.
    - 양생 온도와 습도 등은 이 데이터에 포함되어 있지 않으므로
      양생 기간만으로 강도 발현을 완전히 설명할 수는 없습니다.
    """
)

st.divider()


# --------------------------------------------------
# 11. 주요 변수와 압축강도의 관계 종합
# --------------------------------------------------

st.header("7. 주요 변수와 압축강도의 관계 종합")

st.write(
    """
    지금까지 살펴본 배합 특성과 압축강도의 상관계수를 한곳에서 비교합니다.

    상관계수는 두 변수의 선형적인 관계를 나타냅니다.
    1에 가까우면 양의 관계, -1에 가까우면 음의 관계,
    0에 가까우면 선형 관계가 약하다는 뜻입니다.
    """
)

summary_variables = {
    "물-시멘트비(W/C)": "W_C_Ratio",
    "물-결합재비(W/B)": "W_B_Ratio",
    "시멘트 비율": "Cement_Ratio",
    "고로슬래그 비율": "Slag_Ratio",
    "플라이애시 비율": "FlyAsh_Ratio",
    "굵은 골재 비율": "Coarse_Ratio",
    "잔골재 비율": "Fine_Ratio",
    "고성능 감수제 사용량": "Superplasticizer",
    "양생 기간": "Age",
    "전체 골재량": "Total_Aggregate"
}

correlation_rows = []

for name, column in summary_variables.items():
    valid = df[[column, "Strength"]].dropna()

    correlation_rows.append({
        "변수": name,
        "상관계수": valid[column].corr(valid["Strength"]),
        "데이터 수": len(valid)
    })

correlation_df = pd.DataFrame(correlation_rows)
correlation_df = correlation_df.sort_values(
    "상관계수",
    ascending=False
)

fig = px.bar(
    correlation_df,
    x="상관계수",
    y="변수",
    orientation="h",
    color="상관계수",
    color_continuous_scale="RdBu",
    range_color=[-1, 1],
    title="주요 배합 특성과 압축강도의 상관계수 비교",
    hover_data=["데이터 수"]
)

fig.add_vline(x=0, line_dash="dash")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    correlation_df.style.format({
        "상관계수": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# 가장 큰 양의 상관관계와 음의 상관관계
positive = correlation_df.loc[
    correlation_df["상관계수"].idxmax()
]

negative = correlation_df.loc[
    correlation_df["상관계수"].idxmin()
]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "가장 큰 양의 상관관계",
        positive["변수"],
        f"{positive['상관계수']:.3f}"
    )

with col2:
    st.metric(
        "가장 큰 음의 상관관계",
        negative["변수"],
        f"{negative['상관계수']:.3f}"
    )


# --------------------------------------------------
# 12. 마무리
# --------------------------------------------------

st.divider()

st.header("📌 화학공학적 분석 정리")

st.markdown(
    """
    이번 분석에서는 콘크리트의 압축강도에 영향을 줄 수 있는
    여러 배합 특성을 살펴보았습니다.

    1. **물-시멘트비와 물-결합재비:** 물과 결합재의 비율을 통해 배합 특성을 살펴봅니다.
    2. **결합재 구성:** 시멘트, 고로슬래그, 플라이애시의 비율을 비교합니다.
    3. **골재 구성:** 굵은 골재와 잔골재의 양과 비율을 살펴봅니다.
    4. **고성능 감수제:** 감수제 사용 여부와 물 사용량 및 압축강도를 비교합니다.
    5. **양생 기간:** 시간이 지남에 따른 압축강도의 변화를 살펴봅니다.
    6. **종합 분석:** 여러 변수와 압축강도의 상관관계를 비교합니다.

    다음 단계에서는 이러한 변수들을 이용해 실제 압축강도를 예측하는
    머신러닝 모델을 만들어 볼 수 있습니다.
    """
)

st.caption(
    "참고: 상관관계는 인과관계를 의미하지 않습니다. "
    "실제 콘크리트 배합 설계에는 재료 특성, 시공성, 양생 조건 등도 고려해야 합니다."
)
