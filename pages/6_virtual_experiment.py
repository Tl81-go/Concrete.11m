
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# ==================================================
# 1. 기본 설정
# ==================================================

st.set_page_config(
    page_title="가상 배합 실험 | 콘크리트 압축강도 예측 실험실",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 가상 배합 실험")
st.write(
    "콘크리트 재료의 배합을 직접 조절하고 "
    "머신러닝 모델을 활용해 예상 압축강도의 변화를 탐구합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Tl81-go/Concrete.11m/main/data/concrete.csv"
)

FEATURES = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]

TARGET = "Strength"

KOREAN_NAMES = {
    "Cement": "시멘트",
    "Blast Furnace Slag": "고로슬래그",
    "Fly Ash": "플라이애시",
    "Water": "물",
    "Superplasticizer": "고성능 감수제",
    "Coarse Aggregate": "굵은 골재",
    "Fine Aggregate": "잔골재",
    "Age": "양생 기간"
}

UNITS = {
    "Cement": "kg/m³",
    "Blast Furnace Slag": "kg/m³",
    "Fly Ash": "kg/m³",
    "Water": "kg/m³",
    "Superplasticizer": "kg/m³",
    "Coarse Aggregate": "kg/m³",
    "Fine Aggregate": "kg/m³",
    "Age": "일"
}


# ==================================================
# 2. 데이터 불러오기
# ==================================================

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)

    df.columns = df.columns.str.strip()

    rename_candidates = [
        "Concrete compressive strength",
        "Concrete compressive strength (MPa)"
    ]

    for old_name in rename_candidates:
        if old_name in df.columns:
            df = df.rename(columns={old_name: TARGET})

    required = FEATURES + [TARGET]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"필요한 열을 찾을 수 없습니다: {missing}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    return df


# ==================================================
# 3. 랜덤 포레스트 모델 학습
# ==================================================

@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test


try:
    df = load_data()
    model, X_train, X_test, y_test = train_model(df)

except Exception as e:
    st.error("데이터를 불러오거나 모델을 학습하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# ==================================================
# 4. 기준 배합 설정
# ==================================================

base_mix = X_train.median()

data_min = X_train.min()
data_max = X_train.max()

st.divider()

st.header("1. 기준 배합 확인")

st.write(
    "학습 데이터의 각 재료별 중앙값을 기준 배합으로 설정합니다. "
    "이 기준 배합을 바탕으로 새로운 배합을 실험합니다."
)

base_table = pd.DataFrame({
    "재료": [KOREAN_NAMES[col] for col in FEATURES],
    "기준 배합": [base_mix[col] for col in FEATURES],
    "단위": [UNITS[col] for col in FEATURES]
})

st.dataframe(
    base_table.round(2),
    use_container_width=True,
    hide_index=True
)

base_prediction = float(
    model.predict(pd.DataFrame([base_mix]))[0]
)

st.metric(
    "기준 배합의 예상 압축강도",
    f"{base_prediction:.2f} MPa"
)


# ==================================================
# 5. 사용자 배합 조절
# ==================================================

st.divider()

st.header("2. 나만의 배합 만들기")

st.write(
    "슬라이더를 조절하여 재료의 양과 양생 기간을 변경해 보세요. "
    "슬라이더 범위는 학습 데이터의 최솟값과 최댓값을 기준으로 합니다."
)

custom_mix = {}

left, right = st.columns(2)

for i, col in enumerate(FEATURES):
    container = left if i < 4 else right

    with container:
        minimum = float(data_min[col])
        maximum = float(data_max[col])
        default = float(base_mix[col])

        if maximum <= minimum:
            custom_mix[col] = minimum
            st.write(
                f"**{KOREAN_NAMES[col]}:** {minimum:.2f}"
            )

        elif col == "Age":
            custom_mix[col] = st.slider(
                f"{KOREAN_NAMES[col]} ({UNITS[col]})",
                min_value=int(np.ceil(minimum)),
                max_value=int(np.floor(maximum)),
                value=int(round(default)),
                step=1,
                key=f"custom_{col}"
            )

        else:
            custom_mix[col] = st.slider(
                f"{KOREAN_NAMES[col]} ({UNITS[col]})",
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=max((maximum - minimum) / 100, 0.01),
                format="%.2f",
                key=f"custom_{col}"
            )

custom_df = pd.DataFrame([custom_mix])

custom_prediction = float(
    model.predict(custom_df)[0]
)

difference = custom_prediction - base_prediction


# ==================================================
# 6. 기준 배합과 실험 배합 비교
# ==================================================

st.divider()

st.header("3. 가상 배합 실험 결과")

col1, col2, col3 = st.columns(3)

col1.metric(
    "기준 배합",
    f"{base_prediction:.2f} MPa"
)

col2.metric(
    "실험 배합",
    f"{custom_prediction:.2f} MPa"
)

col3.metric(
    "예상 강도 변화",
    f"{difference:+.2f} MPa"
)

if difference > 0:
    st.success(
        f"실험 배합의 예상 압축강도가 기준 배합보다 "
        f"{difference:.2f} MPa 높게 예측되었습니다."
    )
elif difference < 0:
    st.info(
        f"실험 배합의 예상 압축강도가 기준 배합보다 "
        f"{abs(difference):.2f} MPa 낮게 예측되었습니다."
    )
else:
    st.info("두 배합의 예상 압축강도가 같습니다.")

comparison = pd.DataFrame({
    "재료": [KOREAN_NAMES[col] for col in FEATURES],
    "기준 배합": [base_mix[col] for col in FEATURES],
    "실험 배합": [custom_mix[col] for col in FEATURES]
})

st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True
)

comparison_chart = pd.DataFrame({
    "배합": ["기준 배합", "실험 배합"],
    "예상 압축강도": [
        base_prediction,
        custom_prediction
    ]
})

fig = px.bar(
    comparison_chart,
    x="배합",
    y="예상 압축강도",
    text="예상 압축강도",
    title="기준 배합과 실험 배합의 압축강도 비교"
)

fig.update_traces(
    texttemplate="%{text:.2f} MPa",
    textposition="outside"
)

fig.update_layout(
    yaxis_title="예상 압축강도 (MPa)",
    xaxis_title="",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)


# ==================================================
# 7. 재료 하나씩 변경하기
# ==================================================

st.divider()

st.header("4. 재료 하나씩 변경했을 때의 영향")

st.write(
    "기준 배합에서 한 가지 재료만 변경하고 나머지는 그대로 유지합니다. "
    "각 재료를 데이터 범위의 약 10%만큼 증가시키거나, "
    "증가할 수 없는 경우 감소시켜 비교합니다."
)

single_results = []

for col in FEATURES:
    changed_mix = base_mix.copy()

    change_amount = (
        float(data_max[col]) - float(data_min[col])
    ) * 0.1

    new_value = min(
        float(base_mix[col]) + change_amount,
        float(data_max[col])
    )

    if new_value == float(base_mix[col]):
        new_value = max(
            float(base_mix[col]) - change_amount,
            float(data_min[col])
        )

    changed_mix[col] = new_value

    changed_prediction = float(
        model.predict(pd.DataFrame([changed_mix]))[0]
    )

    single_results.append({
        "변경한 재료": KOREAN_NAMES[col],
        "기준값": base_mix[col],
        "변경값": new_value,
        "기준 예상 강도": base_prediction,
        "변경 후 예상 강도": changed_prediction,
        "예상 강도 변화": changed_prediction - base_prediction
    })

single_df = pd.DataFrame(single_results)

st.dataframe(
    single_df.round(2),
    use_container_width=True,
    hide_index=True
)

impact_fig = px.bar(
    single_df,
    x="변경한 재료",
    y="예상 강도 변화",
    text="예상 강도 변화",
    title="재료 하나씩 변경했을 때 예상 압축강도 변화",
    labels={
        "변경한 재료": "변경한 재료",
        "예상 강도 변화": "예상 강도 변화 (MPa)"
    }
)

impact_fig.update_traces(
    texttemplate="%{text:+.2f}",
    textposition="outside"
)

impact_fig.add_hline(y=0, line_dash="dash")

st.plotly_chart(
    impact_fig,
    use_container_width=True
)


# ==================================================
# 8. 2차원 배합 실험
# ==================================================

st.divider()

st.header("5. 두 가지 재료의 조합 탐구")

st.write(
    "이번에는 두 가지 변수를 동시에 변경해 보겠습니다. "
    "나머지 재료는 기준 배합으로 고정하고, "
    "두 변수의 조합에 따른 예상 압축강도를 색상으로 표현합니다."
)

experiment_type = st.selectbox(
    "탐구할 배합 실험을 선택하세요.",
    [
        "물-결합재비와 압축강도",
        "시멘트-플라이애시 조합",
        "시멘트-양생 기간"
    ]
)

st.caption(
    "그래프의 색상은 실제 시험 결과가 아니라 "
    "랜덤 포레스트 모델의 예상 압축강도입니다."
)


# --------------------------------------------------
# 실험 1: 물-결합재비
# --------------------------------------------------

if experiment_type == "물-결합재비와 압축강도":

    st.subheader("물과 결합재 총량의 조합")

    st.write(
        "결합재는 시멘트, 고로슬래그, 플라이애시의 합입니다. "
        "결합재 총량을 변경할 때는 기준 배합의 결합재 구성 비율을 유지합니다."
    )

    binder_data = (
        X_train["Cement"]
        + X_train["Blast Furnace Slag"]
        + X_train["Fly Ash"]
    )

    binder_min = float(binder_data.min())
    binder_max = float(binder_data.max())

    water_min = float(data_min["Water"])
    water_max = float(data_max["Water"])

    cement_base = float(base_mix["Cement"])
    slag_base = float(base_mix["Blast Furnace Slag"])
    flyash_base = float(base_mix["Fly Ash"])

    binder_base = cement_base + slag_base + flyash_base

    if binder_base <= 0:
        st.warning("기준 배합의 결합재 총량이 0이어서 실험할 수 없습니다.")
        st.stop()

    cement_fraction = cement_base / binder_base
    slag_fraction = slag_base / binder_base
    flyash_fraction = flyash_base / binder_base

    binder_values = np.linspace(binder_min, binder_max, 30)
    water_values = np.linspace(water_min, water_max, 30)

    X_grid, Y_grid = np.meshgrid(
        binder_values,
        water_values
    )

    grid_df = pd.DataFrame({
        "Cement": (X_grid.ravel() * cement_fraction),
        "Blast Furnace Slag": (X_grid.ravel() * slag_fraction),
        "Fly Ash": (X_grid.ravel() * flyash_fraction),
        "Water": Y_grid.ravel(),
        "Superplasticizer": base_mix["Superplasticizer"],
        "Coarse Aggregate": base_mix["Coarse Aggregate"],
        "Fine Aggregate": base_mix["Fine Aggregate"],
        "Age": base_mix["Age"]
    })

    predictions = model.predict(grid_df).reshape(X_grid.shape)

    heatmap = go.Figure(data=go.Heatmap(
        x=binder_values,
        y=water_values,
        z=predictions,
        colorscale="RdYlBu_r",
        colorbar=dict(
            title="예상 강도<br>(MPa)"
        ),
        hovertemplate=(
            "결합재 총량: %{x:.1f} kg/m³<br>"
            "물의 양: %{y:.1f} kg/m³<br>"
            "예상 압축강도: %{z:.2f} MPa"
            "<extra></extra>"
        )
    ))

    heatmap.update_layout(
        title="물-결합재 배합에 따른 예상 압축강도",
        xaxis_title="결합재 총량 (kg/m³)",
        yaxis_title="물의 양 (kg/m³)"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.info(
        "이 실험은 결합재 총량을 바꾸면서 시멘트·고로슬래그·"
        "플라이애시의 구성 비율은 기준 배합과 동일하게 유지합니다."
    )

    st.markdown("**탐구 의의**")
    st.write(
        "물-결합재비(W/B)는 물의 양을 결합재 총량으로 나눈 비율입니다. "
        "물과 결합재의 상대적인 양이 달라질 때 모델의 강도 예측이 "
        "어떻게 변하는지 탐구할 수 있습니다."
    )


# --------------------------------------------------
# 실험 2: 시멘트와 플라이애시
# --------------------------------------------------

elif experiment_type == "시멘트-플라이애시 조합":

    st.subheader("시멘트와 플라이애시의 조합")

    st.write(
        "시멘트와 플라이애시의 양을 동시에 바꿔 봅니다. "
        "고로슬래그와 나머지 재료는 기준 배합으로 유지합니다."
    )

    x_values = np.linspace(
        float(data_min["Cement"]),
        float(data_max["Cement"]),
        30
    )

    y_values = np.linspace(
        float(data_min["Fly Ash"]),
        float(data_max["Fly Ash"]),
        30
    )

    X_grid, Y_grid = np.meshgrid(
        x_values,
        y_values
    )

    grid_df = pd.DataFrame({
        "Cement": X_grid.ravel(),
        "Blast Furnace Slag": base_mix["Blast Furnace Slag"],
        "Fly Ash": Y_grid.ravel(),
        "Water": base_mix["Water"],
        "Superplasticizer": base_mix["Superplasticizer"],
        "Coarse Aggregate": base_mix["Coarse Aggregate"],
        "Fine Aggregate": base_mix["Fine Aggregate"],
        "Age": base_mix["Age"]
    })

    predictions = model.predict(grid_df).reshape(X_grid.shape)

    heatmap = go.Figure(data=go.Heatmap(
        x=x_values,
        y=y_values,
        z=predictions,
        colorscale="RdYlBu_r",
        colorbar=dict(
            title="예상 강도<br>(MPa)"
        ),
        hovertemplate=(
            "시멘트: %{x:.1f} kg/m³<br>"
            "플라이애시: %{y:.1f} kg/m³<br>"
            "예상 압축강도: %{z:.2f} MPa"
            "<extra></extra>"
        )
    ))

    heatmap.update_layout(
        title="시멘트와 플라이애시 조합에 따른 예상 압축강도",
        xaxis_title="시멘트 양 (kg/m³)",
        yaxis_title="플라이애시 양 (kg/m³)"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.markdown("**탐구 의의**")
    st.write(
        "플라이애시는 시멘트 일부를 대체할 수 있는 혼화재입니다. "
        "시멘트와 플라이애시의 양을 함께 변경하면서 "
        "결합재 구성의 변화가 예측 강도와 어떤 관계를 보이는지 탐구합니다."
    )


# --------------------------------------------------
# 실험 3: 시멘트와 양생 기간
# --------------------------------------------------

elif experiment_type == "시멘트-양생 기간":

    st.subheader("시멘트 양과 양생 기간의 조합")

    st.write(
        "시멘트 양과 양생 기간을 동시에 변경합니다. "
        "나머지 재료는 기준 배합으로 유지합니다."
    )

    x_values = np.linspace(
        float(data_min["Cement"]),
        float(data_max["Cement"]),
        30
    )

    y_values = np.linspace(
        float(data_min["Age"]),
        float(data_max["Age"]),
        30
    )

    X_grid, Y_grid = np.meshgrid(
        x_values,
        y_values
    )

    grid_df = pd.DataFrame({
        "Cement": X_grid.ravel(),
        "Blast Furnace Slag": base_mix["Blast Furnace Slag"],
        "Fly Ash": base_mix["Fly Ash"],
        "Water": base_mix["Water"],
        "Superplasticizer": base_mix["Superplasticizer"],
        "Coarse Aggregate": base_mix["Coarse Aggregate"],
        "Fine Aggregate": base_mix["Fine Aggregate"],
        "Age": Y_grid.ravel()
    })

    grid_df["Age"] = grid_df["Age"].round().astype(int)

    predictions = model.predict(grid_df).reshape(X_grid.shape)

    heatmap = go.Figure(data=go.Heatmap(
        x=x_values,
        y=y_values,
        z=predictions,
        colorscale="RdYlBu_r",
        colorbar=dict(
            title="예상 강도<br>(MPa)"
        ),
        hovertemplate=(
            "시멘트: %{x:.1f} kg/m³<br>"
            "양생 기간: %{y:.1f}일<br>"
            "예상 압축강도: %{z:.2f} MPa"
            "<extra></extra>"
        )
    ))

    heatmap.update_layout(
        title="시멘트와 양생 기간에 따른 예상 압축강도",
        xaxis_title="시멘트 양 (kg/m³)",
        yaxis_title="양생 기간 (일)"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.markdown("**탐구 의의**")
    st.write(
        "시멘트의 양과 양생 기간이 함께 달라질 때 "
        "모델이 예측하는 압축강도의 변화를 탐구합니다. "
        "양생 기간에 따른 강도 발현 양상을 살펴볼 수 있습니다."
    )


# ==================================================
# 9. 마무리 및 주의사항
# ==================================================

st.divider()

st.header("6. 실험 결과 정리")

st.write(
    """
    이번 가상 배합 실험에서는 다음과 같은 내용을 탐구했습니다.

    - 기준 배합과 실험 배합의 예상 압축강도 비교
    - 한 가지 재료를 변경했을 때의 예측 변화
    - 두 가지 재료의 조합에 따른 예상 강도 분포
    - 물-결합재비, 혼화재 구성, 양생 기간의 관계
    """
)

st.warning(
    """
    **실험 결과 해석 시 주의사항**

    이 페이지의 결과는 실제 콘크리트를 제조하고 압축 시험을 진행한 결과가 아닙니다.
    기존 데이터로 학습한 랜덤 포레스트 모델의 예측값입니다.

    두 변수를 변경하는 그래프에서는 나머지 재료를 기준 배합으로 고정합니다.
    따라서 실제 배합에서 발생할 수 있는 모든 재료 간 상호작용을 반영하지는 못합니다.

    또한 모델은 학습 데이터의 패턴을 바탕으로 예측하므로,
    색상 차이만으로 특정 재료가 강도 변화의 원인이라고 단정할 수 없습니다.
    실제 구조물의 안전성이나 품질을 판단하는 데 사용해서는 안 됩니다.
    """
)

st.caption(
    "가상 배합 실험 | 콘크리트 압축강도 예측 실험실"
)
