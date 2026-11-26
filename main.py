import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="KBO 구단 투수력 분석",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ KBO 구단 투수력 분석")
st.subheader("팀 평균자책점(ERA)과 승수(W)의 상관관계 분석")
st.write(
    "2024~2026년 KBO 10개 구단의 정규시즌 기록 30개를 이용해 "
    "팀 평균자책점과 승수 사이의 관계를 알아봅니다."
)

# code/main.py에서 실행해도 data/kbo_team.csv를 찾도록 경로 설정
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "kbo_team.csv"

st.divider()
st.header("1. 데이터 불러오기")

uploaded_file = st.file_uploader(
    "KBO 데이터 CSV 파일 업로드 (선택)",
    type=["csv"]
)

@st.cache_data
def read_csv_from_path(path_string):
    path = Path(path_string)
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 확인해 주세요.")

def read_uploaded_csv(file):
    file.seek(0)
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("업로드한 CSV의 인코딩을 확인해 주세요.")

try:
    if uploaded_file is not None:
        raw = read_uploaded_csv(uploaded_file)
        st.success("업로드한 CSV 파일을 불러왔습니다.")
    elif CSV_PATH.exists():
        raw = read_csv_from_path(str(CSV_PATH))
        st.success("GitHub의 data/kbo_team.csv 파일을 불러왔습니다.")
    else:
        st.error(
            "CSV 파일을 찾을 수 없습니다. GitHub에 data/kbo_team.csv를 올리거나 "
            "위에서 CSV 파일을 직접 업로드해 주세요."
        )
        st.stop()
except Exception as error:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()

# 열 이름 앞뒤 공백 및 BOM 제거
raw.columns = (
    raw.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

# 흔히 쓰이는 다른 열 이름도 자동으로 변환
rename_map = {
    "승": "승수", "W": "승수", "승리": "승수",
    "무": "무승부", "D": "무승부",
    "패": "패수", "L": "패수",
    "ERA": "팀ERA", "평균자책점": "팀ERA",
    "팀 평균자책점": "팀ERA",
    "팀": "팀명", "구단": "팀명", "구단명": "팀명",
    "시즌": "연도", "년도": "연도"
}
raw = raw.rename(columns=rename_map)

required_columns = ["승수", "무승부", "패수", "팀ERA", "팀명", "연도"]
missing_columns = [
    col for col in required_columns if col not in raw.columns
]

if missing_columns:
    st.error("CSV에 다음 열이 없습니다: " + ", ".join(missing_columns))
    st.write("현재 CSV 열 이름:", list(raw.columns))
    st.info("필요한 열: 승수, 무승부, 패수, 팀ERA, 팀명, 연도")
    st.stop()

data = raw[required_columns].copy()

# 숫자형으로 변환하고 잘못된 행 제거
for col in ["승수", "무승부", "패수", "팀ERA", "연도"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["팀명"] = data["팀명"].astype("string").str.strip()
data = data.dropna(subset=required_columns)

data = data[
    (data["승수"] >= 0)
    & (data["무승부"] >= 0)
    & (data["패수"] >= 0)
    & (data["팀ERA"] > 0)
    & (data["연도"].between(2024, 2026))
].copy()

data["연도"] = data["연도"].astype(int)
data = data[
    ~data["팀명"].str.contains(
        "합계|전체|평균|nan|none",
        case=False,
        na=False
    )
].copy()

data = data.drop_duplicates(subset=["연도", "팀명"])
data = data.sort_values(["연도", "팀명"]).reset_index(drop=True)

if data.empty:
    st.error("분석할 데이터가 없습니다. CSV 내용을 확인해 주세요.")
    st.stop()

st.success(f"총 {len(data)}행의 데이터를 불러왔습니다.")
st.caption("CSV 형식: 6열 / 승수, 무승부, 패수, 팀ERA, 팀명, 연도")

with st.expander("전체 CSV 데이터 확인하기"):
    st.dataframe(data, use_container_width=True, hide_index=True)

st.divider()
st.header("2. 분석할 시즌 선택")

year_options = ["전체"] + sorted(data["연도"].unique().tolist())
selected_year = st.selectbox("분석 범위", year_options)

if selected_year == "전체":
    analysis_data = data.copy()
else:
    analysis_data = data[data["연도"] == selected_year].copy()

if len(analysis_data) < 3:
    st.warning("상관관계를 분석하려면 데이터가 최소 3개 필요합니다.")
    st.stop()

if analysis_data["팀ERA"].nunique() < 2 or analysis_data["승수"].nunique() < 2:
    st.warning("선택한 데이터의 값이 모두 같아 상관관계를 계산할 수 없습니다.")
    st.stop()

st.header("3. 기초 통계")

col1, col2, col3 = st.columns(3)
col1.metric("분석 데이터", f"{len(analysis_data)}개")
col2.metric("평균 팀 ERA", f"{analysis_data['팀ERA'].mean():.2f}")
col3.metric("평균 승수", f"{analysis_data['승수'].mean():.1f}승")

st.dataframe(
    analysis_data[["팀ERA", "승수", "무승부", "패수"]]
    .describe()
    .round(2),
    use_container_width=True
)

st.divider()
st.header("4. 상관관계 분석")

pearson_corr = analysis_data["팀ERA"].corr(
    analysis_data["승수"],
    method="pearson"
)
spearman_corr = analysis_data["팀ERA"].corr(
    analysis_data["승수"],
    method="spearman"
)

corr_col1, corr_col2 = st.columns(2)
corr_col1.metric("피어슨 상관계수", f"{pearson_corr:.3f}")
corr_col2.metric("스피어만 상관계수", f"{spearman_corr:.3f}")

absolute_corr = abs(pearson_corr)
if absolute_corr < 0.2:
    strength_text = "거의 상관관계가 없는 수준"
elif absolute_corr < 0.4:
    strength_text = "약한 상관관계"
elif absolute_corr < 0.6:
    strength_text = "중간 정도의 상관관계"
elif absolute_corr < 0.8:
    strength_text = "비교적 강한 상관관계"
else:
    strength_text = "강한 상관관계"

direction_text = "음의 상관관계" if pearson_corr < 0 else "양의 상관관계"
st.info(
    f"현재 선택한 데이터에서는 {strength_text}이며, "
    f"방향은 {direction_text}입니다."
)
st.caption(
    "상관계수는 두 변수의 관련성을 나타냅니다. "
    "상관관계만으로 한 변수가 다른 변수의 원인이라고 단정할 수는 없습니다."
)

st.divider()
st.header("5. 산점도 분석")
st.write(
    "X축은 팀 평균자책점(ERA), Y축은 승수(W)입니다. "
    "점 하나는 한 시즌의 한 구단을 나타냅니다."
)

scatter_fig = px.scatter(
    analysis_data,
    x="팀ERA",
    y="승수",
    color="연도",
    hover_name="팀명",
    hover_data={
        "팀ERA": ":.2f",
        "승수": True,
        "무승부": True,
        "패수": True,
        "연도": True
    },
    labels={
        "팀ERA": "팀 평균자책점(ERA)",
        "승수": "승수(W)",
        "연도": "시즌"
    },
    title="KBO 구단 팀 평균자책점과 승수의 관계"
)

# 산점도에 선형 회귀선 추가
x_values = analysis_data["팀ERA"].to_numpy(dtype=float)
y_values = analysis_data["승수"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x_values, y_values, 1)

line_x = np.linspace(x_values.min(), x_values.max(), 100)
line_y = slope * line_x + intercept

scatter_fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형 회귀선",
        line=dict(width=3, dash="dash")
    )
)

scatter_fig.update_layout(
    template="plotly_white",
    height=550,
    xaxis_title="팀 평균자책점(ERA)",
    yaxis_title="승수(W)",
    legend_title="시즌"
)
st.plotly_chart(scatter_fig, use_container_width=True)

st.write(
    f"**회귀식:** 승수(W) = {slope:.3f} × 팀ERA + ({intercept:.3f})"
)
st.write(
    f"회귀선의 기울기는 **{slope:.3f}**입니다. "
    "기울기가 음수이면 ERA가 높아질수록 승수가 감소하는 방향을 뜻합니다."
)

st.divider()
st.header("6. 시즌별 기록 비교")

era_chart = px.bar(
    analysis_data.sort_values("팀ERA"),
    x="팀명",
    y="팀ERA",
    color="연도",
    barmode="group",
    text_auto=".2f",
    labels={
        "팀명": "구단",
        "팀ERA": "팀 평균자책점(ERA)",
        "연도": "시즌"
    },
    title="구단별 팀 평균자책점"
)
era_chart.update_layout(template="plotly_white", height=430)
st.plotly_chart(era_chart, use_container_width=True)

win_chart = px.bar(
    analysis_data.sort_values("승수", ascending=False),
    x="팀명",
    y="승수",
    color="연도",
    barmode="group",
    text_auto=True,
    labels={
        "팀명": "구단",
        "승수": "승수(W)",
        "연도": "시즌"
    },
    title="구단별 승수"
)
win_chart.update_layout(template="plotly_white", height=430)
st.plotly_chart(win_chart, use_container_width=True)

st.divider()
st.header("7. 선형 회귀모델 평가")

if len(analysis_data) >= 10:
    X = analysis_data[["팀ERA"]]
    y = analysis_data["승수"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_predictions = np.full(
        len(y_test),
        y_train.mean()
    )

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )
    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    model_r2 = r2_score(y_test, predictions)

    st.write(
        f"학습 데이터 {len(X_train)}개 / 테스트 데이터 {len(X_test)}개"
    )
    st.write(
        f"학습 데이터로 만든 회귀식: "
        f"**승수 = {model.coef_[0]:.3f} × ERA + ({model.intercept_:.3f})**"
    )

    evaluation = pd.DataFrame({
        "평가 지표": ["MAE", "RMSE"],
        "기준선 모델": [baseline_mae, baseline_rmse],
        "선형 회귀모델": [model_mae, model_rmse]
    })
    st.dataframe(
        evaluation.round(3),
        use_container_width=True,
        hide_index=True
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("회귀모델 MAE", f"{model_mae:.2f}승")
    metric2.metric("회귀모델 RMSE", f"{model_rmse:.2f}승")
    metric3.metric("회귀모델 R²", f"{model_r2:.3f}")

    if model_mae < baseline_mae:
        st.success("테스트 데이터에서 회귀모델의 MAE가 기준선 모델보다 작습니다.")
    else:
        st.warning("테스트 데이터에서 회귀모델의 MAE가 기준선 모델보다 작지 않습니다.")

    test_result = pd.DataFrame({
        "실제 승수": y_test.to_numpy(),
        "예측 승수": np.round(predictions, 2),
        "오차(실제-예측)": np.round(y_test.to_numpy() - predictions, 2)
    })
    st.subheader("테스트 데이터 예측 결과")
    st.dataframe(
        test_result.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "회귀모델 평가를 위해서는 선택한 데이터가 10개 이상이어야 합니다. "
        "전체 시즌을 선택하면 30개 기록으로 평가할 수 있습니다."
    )

st.divider()
st.header("8. 분석의 한계")
st.markdown("""
- 분석 자료는 2024~2026년 30개 구단-시즌 기록이므로 표본 수가 많지 않습니다.
- 승수는 팀 ERA뿐 아니라 타선의 득점 지원, 수비, 경기 운영 등 여러 요인의 영향을 받습니다.
- 상관관계가 있다고 해서 ERA가 승수 변화의 유일한 원인이라는 뜻은 아닙니다.
- 80:20 데이터 분할에서는 테스트 데이터가 적어 평가 수치가 달라질 수 있습니다.
""")

st.caption(
    "자료 출처: KBO 공식 기록실 - 역대 구단성적. "
    "각 시즌 정규시즌의 구단별 승·무·패·평균자책점 기록을 사용했습니다."
)
