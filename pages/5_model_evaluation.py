
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="모델 평가 | 콘크리트 강도 예측 실험실",
    page_icon="📊",
    layout="wide"
)

st.title("📊 머신러닝 모델 평가 및 비교")

st.write(
    """
    하나의 데이터에 여러 가지 머신러닝 회귀 모델을 적용하고,
    각 모델의 압축강도 예측 성능을 비교합니다.

    어떤 모델이 콘크리트 압축강도를 가장 정확하게 예측하는지 알아봅시다.
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
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("오류 내용:", e)
    st.stop()


# 필요한 칼럼 확인
required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("CSV 파일에 필요한 칼럼이 없습니다.")
    st.write(missing_columns)
    st.stop()


# 분석에 필요한 데이터 준비
model_data = df[required_columns].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

# 정답인 압축강도가 없는 행은 제외
model_data = model_data.dropna(
    subset=[TARGET]
)

if len(model_data) < 10:
    st.error("모델 학습에 사용할 데이터가 너무 적습니다.")
    st.stop()


# --------------------------------------------------
# 3. 여러 모델 학습 및 평가
# --------------------------------------------------

@st.cache_resource
def evaluate_models(data):

    X = data[FEATURES]
    y = data[TARGET]

    # 모든 모델이 동일한 데이터로 학습하고 평가하도록 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 비교할 회귀 모델
    models = {
        "선형 회귀": LinearRegression(),

        "의사결정트리": DecisionTreeRegressor(
            random_state=42,
            max_depth=8,
            min_samples_leaf=3
        ),

        "랜덤 포레스트": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "그래디언트 부스팅": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    trained_models = {}
    predictions = {}
    result_rows = []

    for name, regressor in models.items():

        # 결측값 처리 후 모델 학습
        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "regressor",
                regressor
            )
        ])

        pipeline.fit(X_train, y_train)

        # 평가용 데이터 예측
        y_pred = pipeline.predict(X_test)

        # 성능 지표 계산
        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        result_rows.append({
            "모델": name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

        trained_models[name] = pipeline
        predictions[name] = y_pred

    result_df = pd.DataFrame(result_rows)

    # RMSE가 작은 순서로 정렬
    result_df = result_df.sort_values(
        "RMSE",
        ascending=True
    ).reset_index(drop=True)

    return (
        result_df,
        trained_models,
        predictions,
        X_train,
        X_test,
        y_train,
        y_test
    )


try:
    (
        result_df,
        trained_models,
        predictions,
        X_train,
        X_test,
        y_train,
        y_test
    ) = evaluate_models(model_data)

except Exception as e:
    st.error("모델 평가 중 오류가 발생했습니다.")
    st.write("오류 내용:", e)
    st.stop()


# --------------------------------------------------
# 4. 데이터 분할 결과
# --------------------------------------------------

st.divider()

st.header("1. 데이터 분할 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{len(model_data):,}개"
    )

with col2:
    st.metric(
        "학습용 데이터",
        f"{len(X_train):,}개"
    )

with col3:
    st.metric(
        "평가용 데이터",
        f"{len(X_test):,}개"
    )

st.write(
    """
    모든 모델은 동일한 학습용 데이터와 평가용 데이터를 사용합니다.

    학습용 데이터로 모델을 학습하고,
    평가용 데이터로 처음 보는 데이터에 대한 예측 성능을 확인합니다.
    """
)

st.info(
    "데이터 분할은 80:20이며, random_state=42로 고정해 "
    "모델 간 비교가 같은 조건에서 이루어지도록 했습니다."
)


# --------------------------------------------------
# 5. 모델별 성능 비교
# --------------------------------------------------

st.divider()

st.header("2. 모델별 성능 비교")

st.write(
    """
    세 가지 평가 지표를 이용해 네 가지 회귀 모델을 비교합니다.
    """
)

# 가장 좋은 모델
best_model = result_df.iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "가장 낮은 RMSE 모델",
        best_model["모델"]
    )

with col2:
    st.metric(
        "해당 모델의 RMSE",
        f"{best_model['RMSE']:.3f} MPa"
    )

with col3:
    st.metric(
        "해당 모델의 R²",
        f"{best_model['R²']:.3f}"
    )

st.subheader("모델 성능 평가표")

display_result = result_df.copy()

st.dataframe(
    display_result.style.format({
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "RMSE가 가장 낮은 모델을 기준으로 가장 좋은 모델을 표시했습니다."
)


# --------------------------------------------------
# 6. 평가 지표별 그래프
# --------------------------------------------------

st.subheader("평가 지표별 비교 그래프")

metric_choice = st.selectbox(
    "비교할 평가 지표를 선택하세요.",
    ["MAE", "RMSE", "R²"]
)

if metric_choice == "R²":
    chart_df = result_df.sort_values(
        "R²",
        ascending=True
    )

    chart_title = "모델별 결정계수(R²) 비교"

else:
    chart_df = result_df.sort_values(
        metric_choice,
        ascending=False
    )

    chart_title = f"모델별 {metric_choice} 비교"

fig = px.bar(
    chart_df,
    x=metric_choice,
    y="모델",
    orientation="h",
    title=chart_title,
    text=metric_choice,
    color=metric_choice,
    color_continuous_scale="Blues"
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown(
    """
    **평가 지표 해석**

    - MAE: 실제값과 예측값의 평균적인 차이입니다. 낮을수록 좋습니다.
    - RMSE: 큰 예측 오차에 더 큰 영향을 주는 지표입니다. 낮을수록 좋습니다.
    - R²: 모델이 데이터의 변화를 얼마나 설명하는지 나타냅니다. 일반적으로 1에 가까울수록 좋습니다.
    """
)


# --------------------------------------------------
# 7. 실제 압축강도와 예측 압축강도 비교
# --------------------------------------------------

st.divider()

st.header("3. 실제값과 예측값 비교")

selected_model = st.selectbox(
    "비교할 모델을 선택하세요.",
    result_df["모델"].tolist()
)

selected_prediction = predictions[selected_model]

comparison_df = pd.DataFrame({
    "실제 압축강도": y_test.values,
    "예측 압축강도": selected_prediction
})

fig = px.scatter(
    comparison_df,
    x="실제 압축강도",
    y="예측 압축강도",
    title=f"{selected_model}: 실제값과 예측값",
    labels={
        "실제 압축강도": "실제 압축강도(MPa)",
        "예측 압축강도": "예측 압축강도(MPa)"
    },
    opacity=0.7
)

# 실제값과 예측값이 같은 경우의 기준선
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
    line=dict(
        dash="dash",
        width=2
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write(
    """
    점들이 대각선 기준선에 가까울수록 예측값이 실제값과 비슷합니다.

    기준선 위쪽의 점은 실제보다 높게 예측한 것이고,
    아래쪽의 점은 실제보다 낮게 예측한 것입니다.
    """
)


# --------------------------------------------------
# 8. 예측 오차 분석
# --------------------------------------------------

st.divider()

st.header("4. 모델별 예측 오차 분석")

st.write(
    """
    예측 오차는 실제 압축강도에서 예측 압축강도를 뺀 값입니다.

    오차가 0에 가까울수록 예측이 정확하다는 뜻입니다.
    """
)

comparison_df["예측 오차"] = (
    comparison_df["실제 압축강도"]
    - comparison_df["예측 압축강도"]
)

fig = px.histogram(
    comparison_df,
    x="예측 오차",
    nbins=30,
    title=f"{selected_model}의 예측 오차 분포",
    labels={
        "예측 오차": "실제값 - 예측값 (MPa)"
    }
)

fig.add_vline(
    x=0,
    line_dash="dash",
    line_width=2
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write(
    """
    - 오차가 0보다 크면 실제값보다 낮게 예측한 것입니다.
    - 오차가 0보다 작으면 실제값보다 높게 예측한 것입니다.
    - 오차가 0 주변에 모여 있을수록 예측이 실제값에 가까운 편입니다.
    """
)


# --------------------------------------------------
# 9. 모델별 변수 중요도
# --------------------------------------------------

st.divider()

st.header("5. 모델이 중요하게 활용한 변수")

st.write(
    """
    트리 기반 모델은 변수 중요도를 계산할 수 있습니다.

    이 값을 통해 모델이 압축강도를 예측할 때
    어떤 입력 변수를 상대적으로 중요하게 활용했는지 살펴봅니다.
    """
)

tree_models = [
    "의사결정트리",
    "랜덤 포레스트",
    "그래디언트 부스팅"
]

selected_tree_model = st.selectbox(
    "변수 중요도를 확인할 모델을 선택하세요.",
    tree_models
)

selected_pipeline = trained_models[selected_tree_model]

importance_values = (
    selected_pipeline
    .named_steps["regressor"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "변수": [
        KOREAN_NAMES[col] for col in FEATURES
    ],
    "중요도": importance_values
})

importance_df = importance_df.sort_values(
    "중요도",
    ascending=True
)

fig = px.bar(
    importance_df,
    x="중요도",
    y="변수",
    orientation="h",
    title=f"{selected_tree_model}의 변수 중요도",
    labels={
        "중요도": "변수 중요도",
        "변수": "입력 변수"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "변수 중요도는 모델 내부에서의 상대적 활용도를 나타냅니다. "
    "재료의 물리적·화학적 인과관계를 증명하는 값은 아닙니다."
)


# --------------------------------------------------
# 10. 모델 비교 결과 정리
# --------------------------------------------------

st.divider()

st.header("📌 모델 비교 결과 정리")

st.write(
    f"""
    현재 평가용 데이터에서 RMSE가 가장 낮은 모델은
    **{best_model['모델']}**입니다.

    해당 모델의 평가 결과는 다음과 같습니다.

    - MAE: {best_model['MAE']:.3f} MPa
    - RMSE: {best_model['RMSE']:.3f} MPa
    - R²: {best_model['R²']:.3f}

    RMSE가 낮다는 것은 큰 예측 오차가 상대적으로 적다는 의미입니다.
    """
)

st.warning(
    "이번 결과는 한 번의 학습용·평가용 데이터 분할에 따른 비교입니다. "
    "다른 데이터 분할이나 실제 실험 환경에서도 같은 모델이 "
    "항상 가장 우수하다고 보장할 수는 없습니다."
)

st.markdown(
    """
    **이번 페이지에서 배운 내용**

    1. 여러 회귀 모델을 같은 데이터로 비교할 수 있습니다.
    2. MAE, RMSE, R²를 이용해 예측 성능을 평가할 수 있습니다.
    3. 실제값과 예측값의 차이를 그래프로 확인할 수 있습니다.
    4. 트리 기반 모델의 변수 중요도를 살펴볼 수 있습니다.

    모델의 성능을 평가하는 과정은 새로운 데이터에서도
    믿을 만한 예측을 할 수 있는지 확인하는 중요한 단계입니다.
    """
)

st.caption(
    "본 페이지는 머신러닝 학습을 위한 교육용 실습입니다. "
    "예측 결과가 실제 콘크리트의 품질이나 안전성을 보증하지는 않습니다."
)
