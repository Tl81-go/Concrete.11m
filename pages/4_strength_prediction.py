
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="압축강도 예측 | 콘크리트 강도 예측 실험실",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ 콘크리트 압축강도 예측")
st.write(
    """
    콘크리트 배합 재료의 양과 양생 기간을 입력하면
    머신러닝 모델이 예상 압축강도를 계산합니다.

    실제 데이터로 모델을 학습한 뒤, 새로운 배합 조건을 입력해 보세요.
    """
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

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


@st.cache_data(ttl=3600)
def load_data():
    data = pd.read_csv(DATA_URL)

    data.columns = data.columns.str.strip()

    if "Concrete compressive strength" in data.columns:
        data = data.rename(
            columns={"Concrete compressive strength": "Strength"}
        )

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("오류 내용:", e)
    st.stop()


required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("CSV 파일에 필요한 칼럼이 없습니다.")
    st.write(missing_columns)
    st.stop()


# --------------------------------------------------
# 3. 모델 학습
# --------------------------------------------------

@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    # 학습용 데이터와 평가용 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 결측값 처리 + 랜덤 포레스트 모델
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1
            )
        )
    ])

    # 학습
    model.fit(X_train, y_train)

    # 평가용 데이터 예측
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, mae, rmse, r2


# 모델 학습에 사용할 데이터
model_data = df[required_columns].copy()

for col in required_columns:
    model_data[col] = pd.to_numeric(
        model_data[col],
        errors="coerce"
    )

# 타깃인 압축강도가 없는 행은 제외
model_data = model_data.dropna(subset=[TARGET])

try:
    (
        model,
        X_test,
        y_test,
        y_pred,
        mae,
        rmse,
        r2
    ) = train_model(model_data)

except Exception as e:
    st.error("모델을 학습하는 중 오류가 발생했습니다.")
    st.write("오류 내용:", e)
    st.stop()


# --------------------------------------------------
# 4. 학습 결과 안내
# --------------------------------------------------

st.divider()

st.header("1. 머신러닝 모델 학습 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("학습에 사용한 데이터", f"{len(model_data):,}개")

with col2:
    st.metric("학습용 데이터", f"{len(X_test) * 4:,}개")

with col3:
    st.metric("평가용 데이터", f"{len(X_test):,}개")

st.write(
    """
    전체 데이터를 학습용 데이터와 평가용 데이터로 나누었습니다.

    - 학습용 데이터: 모델이 패턴을 배우는 데 사용합니다.
    - 평가용 데이터: 학습에 사용하지 않고 모델의 예측 성능을 확인합니다.
    """
)

st.info(
    "현재 모델은 랜덤 포레스트 회귀 모델입니다. "
    "여러 결정 트리의 예측을 종합하여 압축강도를 추정합니다."
)


# --------------------------------------------------
# 5. 예측 입력 화면
# --------------------------------------------------

st.divider()

st.header("2. 콘크리트 배합 조건 입력")

st.write(
    """
    아래 입력창에 콘크리트 재료의 양과 양생 기간을 입력하세요.

    재료의 양은 kg/m³, 양생 기간은 일(day) 단위입니다.
    """
)

st.caption(
    "입력 범위는 원본 데이터에서 관찰된 최솟값과 최댓값을 기준으로 설정했습니다."
)

# 데이터의 최솟값과 최댓값
input_ranges = {}

for feature in FEATURES:
    values = model_data[feature].dropna()

    if len(values) == 0:
        st.error(f"{KOREAN_NAMES[feature]} 데이터가 없습니다.")
        st.stop()

    input_ranges[feature] = {
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median())
    }


with st.form("prediction_form"):

    st.subheader("배합 재료 입력")

    col1, col2 = st.columns(2)

    input_values = {}

    for i, feature in enumerate(FEATURES):

        info = input_ranges[feature]

        # 입력 기본값은 해당 재료의 중앙값
        default_value = info["median"]

        # 양생 기간은 정수로 입력
        if feature == "Age":
            with col2:
                input_values[feature] = st.number_input(
                    "양생 기간 (일)",
                    min_value=max(1, int(np.floor(info["min"]))),
                    max_value=int(np.ceil(info["max"])),
                    value=int(round(default_value)),
                    step=1
                )

        # 나머지 재료는 소수점 한 자리까지 입력
        elif i % 2 == 0:
            with col1:
                input_values[feature] = st.number_input(
                    f"{KOREAN_NAMES[feature]} (kg/m³)",
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(default_value),
                    step=1.0
                )

        else:
            with col2:
                input_values[feature] = st.number_input(
                    f"{KOREAN_NAMES[feature]} (kg/m³)",
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(default_value),
                    step=1.0
                )

    submitted = st.form_submit_button(
        "압축강도 예측하기",
        use_container_width=True
    )


# --------------------------------------------------
# 6. 예측 결과
# --------------------------------------------------

if submitted:

    # 입력값을 데이터프레임으로 변환
    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURES
    )

    # 예측 실행
    predicted_strength = float(model.predict(input_df)[0])

    st.divider()

    st.header("3. 압축강도 예측 결과")

    st.success("예측이 완료되었습니다!")

    st.metric(
        label="예상 압축강도",
        value=f"{predicted_strength:.2f} MPa"
    )

    st.write(
        f"입력한 배합 조건의 예상 압축강도는 "
        f"**{predicted_strength:.2f} MPa**입니다."
    )

    # 입력한 배합 조건 표
    st.subheader("입력한 배합 조건")

    input_table = pd.DataFrame({
        "재료": [KOREAN_NAMES[col] for col in FEATURES],
        "입력값": [input_values[col] for col in FEATURES],
        "단위": [
            "일" if col == "Age" else "kg/m³"
            for col in FEATURES
        ]
    })

    st.dataframe(
        input_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "이 결과는 학습 데이터에 기반한 모델의 추정값이며, "
        "실제 실험으로 측정한 압축강도와 다를 수 있습니다."
    )


# --------------------------------------------------
# 7. 모델 성능 확인
# --------------------------------------------------

st.divider()

st.header("4. 모델의 예측 성능")

st.write(
    """
    모델이 처음 보는 평가용 데이터에 대해 얼마나 정확하게
    예측했는지 확인합니다.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "평균 절대 오차(MAE)",
        f"{mae:.2f} MPa"
    )

with col2:
    st.metric(
        "평균 제곱근 오차(RMSE)",
        f"{rmse:.2f} MPa"
    )

with col3:
    st.metric(
        "결정계수(R²)",
        f"{r2:.3f}"
    )

st.markdown(
    """
    **평가 지표 이해하기**

    - **MAE:** 실제값과 예측값이 평균적으로 얼마나 차이 나는지 나타냅니다. 낮을수록 좋습니다.
    - **RMSE:** 큰 오차에 더 큰 영향을 주는 지표입니다. 낮을수록 좋습니다.
    - **R²:** 모델이 데이터의 변화를 얼마나 설명하는지 나타냅니다. 일반적으로 1에 가까울수록 좋습니다.
    """
)


# 실제값과 예측값 비교
comparison_df = pd.DataFrame({
    "실제 압축강도": y_test.values,
    "예측 압축강도": y_pred
})

fig = px.scatter(
    comparison_df,
    x="실제 압축강도",
    y="예측 압축강도",
    title="실제 압축강도와 모델 예측값 비교",
    labels={
        "실제 압축강도": "실제 압축강도(MPa)",
        "예측 압축강도": "예측 압축강도(MPa)"
    },
    opacity=0.7
)

# 실제값과 예측값이 같은 위치를 나타내는 기준선
min_value = min(
    comparison_df["실제 압축강도"].min(),
    comparison_df["예측 압축강도"].min()
)

max_value = max(
    comparison_df["실제 압축강도"].max(),
    comparison_df["예측 압축강도"].max()
)

fig.add_shape(
    type="line",
    x0=min_value,
    y0=min_value,
    x1=max_value,
    y1=max_value,
    line=dict(dash="dash")
)

st.plotly_chart(fig, use_container_width=True)

st.write(
    """
    점들이 대각선 기준선에 가까울수록 실제 압축강도와
    모델의 예측값이 비슷하다는 의미입니다.
    """
)


# --------------------------------------------------
# 8. 변수 중요도
# --------------------------------------------------

st.divider()

st.header("5. 압축강도 예측에 중요한 재료는?")

st.write(
    """
    랜덤 포레스트는 예측 과정에서 각 입력 변수가 얼마나
    유용하게 사용되었는지를 변수 중요도로 계산할 수 있습니다.
    """
)

regressor = model.named_steps["regressor"]

importance_df = pd.DataFrame({
    "재료": [KOREAN_NAMES[col] for col in FEATURES],
    "중요도": regressor.feature_importances_
})

importance_df = importance_df.sort_values(
    "중요도",
    ascending=True
)

fig = px.bar(
    importance_df,
    x="중요도",
    y="재료",
    orientation="h",
    title="랜덤 포레스트 변수 중요도",
    labels={
        "중요도": "변수 중요도",
        "재료": "입력 변수"
    }
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "변수 중요도는 모델의 예측에 사용된 상대적인 기여도를 나타냅니다. "
    "재료 자체가 압축강도를 직접적으로 결정하는 정도나 인과관계를 뜻하지는 않습니다."
)


# --------------------------------------------------
# 9. 마무리
# --------------------------------------------------

st.divider()

st.header("📌 이번 페이지 정리")

st.markdown(
    """
    이번 페이지에서는 다음 과정을 수행했습니다.

    1. 콘크리트 배합 데이터를 불러왔습니다.
    2. 학습용 데이터와 평가용 데이터로 나누었습니다.
    3. 랜덤 포레스트 회귀 모델을 학습했습니다.
    4. 새로운 배합 조건을 입력받아 압축강도를 예측했습니다.
    5. MAE, RMSE, R²로 예측 성능을 평가했습니다.
    6. 변수 중요도를 통해 모델이 중요하게 활용한 입력 변수를 확인했습니다.

    다음 단계에서는 다른 회귀 모델들과 성능을 비교하고,
    어떤 모델이 콘크리트 압축강도 예측에 더 적합한지 살펴볼 수 있습니다.
    """
)
