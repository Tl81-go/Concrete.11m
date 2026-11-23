
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="가상 배합 실험 | 콘크리트 압축강도 예측 실험실",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 가상 배합 실험")
st.caption("콘크리트 배합을 직접 조절하고 예상 압축강도를 비교해 봅니다.")

DATA_URL = "https://raw.githubusercontent.com/Tl81-go/Concrete.11m/main/data/concrete.csv"

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

TARGET = "Strength"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)

    df.columns = df.columns.str.strip()

    # 압축강도 열 이름 통일
    if "Concrete compressive strength" in df.columns:
        df = df.rename(
            columns={"Concrete compressive strength": "Strength"}
        )

    if "Concrete compressive strength (MPa)" in df.columns:
        df = df.rename(
            columns={"Concrete compressive strength (MPa)": "Strength"}
        )

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"데이터에서 필요한 열을 찾을 수 없습니다: {missing_columns}"
        )

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    return df


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

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
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test


# --------------------------------------------------
# 데이터 및 모델 준비
# --------------------------------------------------

try:
    df = load_data()
    model, X_train, X_test, y_test = train_model(df)

except Exception as e:
    st.error("데이터를 불러오거나 모델을 학습하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 기준 배합 설정
# --------------------------------------------------

# 학습 데이터의 중앙값을 기준 배합으로 사용
base_mix = X_train.median()

# 각 재료의 데이터 범위
data_min = X_train.min()
data_max = X_train.max()


st.markdown("---")

st.subheader("1. 기준 배합 확인")

st.write(
    "먼저 학습 데이터의 각 재료별 중앙값을 기준 배합으로 설정합니다. "
    "아래 배합을 기준으로 새로운 배합의 예측 결과를 비교합니다."
)

base_table = pd.DataFrame({
    "재료": [KOREAN_NAMES[col] for col in FEATURES],
    "기준 배합": [base_mix[col] for col in FEATURES],
    "단위": [UNITS[col] for col in FEATURES]
})

st.dataframe(
    base_table,
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


# --------------------------------------------------
# 가상 배합 조절
# --------------------------------------------------

st.markdown("---")
st.subheader("2. 나만의 배합 만들기")

st.write(
    "슬라이더를 움직여 재료의 양과 양생 기간을 바꿔 보세요. "
    "각 슬라이더의 범위는 실제 데이터에 포함된 최솟값과 최댓값을 기준으로 합니다."
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
            st.write(f"**{KOREAN_NAMES[col]}:** {minimum}")
        elif col == "Age":
            custom_mix[col] = st.slider(
                f"{KOREAN_NAMES[col]} ({UNITS[col]})",
                min_value=int(np.ceil(minimum)),
                max_value=int(np.floor(maximum)),
                value=int(round(default)),
                step=1,
                key=f"mix_{col}"
            )
        else:
            custom_mix[col] = st.slider(
                f"{KOREAN_NAMES[col]} ({UNITS[col]})",
                min_value=minimum,
                max_value=maximum,
                value=float(default),
                step=max((maximum - minimum) / 100, 0.01),
                format="%.2f",
                key=f"mix_{col}"
            )


custom_df = pd.DataFrame([custom_mix])

custom_prediction = float(
    model.predict(custom_df)[0]
)

difference = custom_prediction - base_prediction


# --------------------------------------------------
# 예측 결과
# --------------------------------------------------

st.markdown("---")
st.subheader("3. 가상 배합 실험 결과")

result1, result2, result3 = st.columns(3)

with result1:
    st.metric(
        "기준 배합",
        f"{base_prediction:.2f} MPa"
    )

with result2:
    st.metric(
        "실험 배합",
        f"{custom_prediction:.2f} MPa"
    )

with result3:
    st.metric(
        "예상 강도 변화",
        f"{difference:+.2f} MPa",
        delta=f"{difference:+.2f} MPa"
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
    st.info("기준 배합과 실험 배합의 예상 압축강도가 같습니다.")


# --------------------------------------------------
# 배합 비교
# --------------------------------------------------

st.markdown("---")
st.subheader("4. 기준 배합과 실험 배합 비교")

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

chart_data = pd.DataFrame({
    "배합": ["기준 배합", "실험 배합"],
    "예상 압축강도": [base_prediction, custom_prediction]
})

fig = px.bar(
    chart_data,
    x="배합",
    y="예상 압축강도",
    text="예상 압축강도",
    title="기준 배합과 실험 배합의 예상 압축강도 비교",
    labels={
        "배합": "배합 종류",
        "예상 압축강도": "예상 압축강도 (MPa)"
    }
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


# --------------------------------------------------
# 재료별 영향 실험
# --------------------------------------------------

st.markdown("---")
st.subheader("5. 재료 하나씩 바꿔 보기")

st.write(
    "이번에는 기준 배합에서 재료를 하나씩만 변경해 보겠습니다. "
    "다른 재료는 기준 배합으로 유지하고, 선택한 재료만 실제 데이터 범위의 "
    "약 10%만큼 증가시켜 예상 압축강도의 변화를 확인합니다."
)

single_change_results = []

for col in FEATURES:
    changed_mix = base_mix.copy()

    if data_max[col] > data_min[col]:
        change_amount = (data_max[col] - data_min[col]) * 0.1

        new_value = min(
            float(base_mix[col] + change_amount),
            float(data_max[col])
        )

        # 기준값이 최댓값이라 증가할 수 없는 경우 감소시켜 비교
        if new_value == base_mix[col]:
            new_value = max(
                float(base_mix[col] - change_amount),
                float(data_min[col])
            )

        changed_mix[col] = new_value

    changed_prediction = float(
        model.predict(pd.DataFrame([changed_mix]))[0]
    )

    single_change_results.append({
        "변경한 재료": KOREAN_NAMES[col],
        "기준값": base_mix[col],
        "변경값": changed_mix[col],
        "기준 예상 강도": base_prediction,
        "변경 후 예상 강도": changed_prediction,
        "예상 강도 변화": changed_prediction - base_prediction
    })


single_df = pd.DataFrame(single_change_results)

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
    title="재료를 하나씩 변경했을 때 예상 압축강도 변화",
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

impact_fig.update_layout(
    xaxis_title="변경한 재료",
    yaxis_title="예상 강도 변화 (MPa)"
)

st.plotly_chart(impact_fig, use_container_width=True)


# --------------------------------------------------
# 결과 해석
# --------------------------------------------------

st.markdown("---")
st.subheader("6. 실험 결과 해석")

st.markdown(
    """
    이번 실험에서는 기준 배합과 사용자가 설정한 실험 배합의
    예상 압축강도를 비교했습니다.

    - 예상 압축강도가 증가했다면 모델이 해당 배합에서 더 높은 강도를 예측한 것입니다.
    - 예상 압축강도가 감소했다면 모델이 해당 배합에서 더 낮은 강도를 예측한 것입니다.
    - 재료 하나씩 바꾸기 실험은 다른 재료를 고정했을 때의 예측 변화만 보여 줍니다.
    """
)

st.warning(
    """
    **주의사항**

    이 페이지의 결과는 실제 콘크리트를 제조하고 압축 시험을 진행한 결과가 아닙니다.

    기존 데이터로 학습한 랜덤 포레스트 모델의 예측값입니다.
    재료 사이의 상호작용, 실제 제조 과정, 양생 환경 등 모든 조건을
    완벽하게 반영하지 못할 수 있습니다.

    따라서 이 결과를 실제 건축물의 구조 안전성이나 콘크리트 품질을
    판단하는 데 사용해서는 안 됩니다.
    """
)

st.caption(
    "가상 배합 실험 | 콘크리트 압축강도 예측 실험실"
)
