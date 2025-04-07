import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# -----------------🎨 Page Settings -----------------
st.set_page_config(page_title="Student Suicide Analysis Dashboard", layout="centered")

st.markdown(
    """
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding: 2rem 1rem 1rem 1rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------📌 Title -----------------
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>🎓 Student Suicide Analysis Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px; color: gray;'>"
    "Forecasting suicide trends using <b>ARIMA model</b>"
    "</p>",
    unsafe_allow_html=True
)

# -----------------📂 Load Data -----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    return df

data = load_data("Cleaned_Suicides_Data_2001_2012.csv")

# -----------------🎛 Sidebar Filters -----------------
st.sidebar.header("🔍 Filter Options")
state = st.sidebar.selectbox("Select State", sorted(data['state'].unique()))
gender = st.sidebar.selectbox("Select Gender", sorted(data['gender'].unique()))
age_group = st.sidebar.selectbox("Select Age Group", sorted(data['age_group'].unique()))
forecast_years = st.sidebar.slider("📆 Forecast Years", min_value=1, max_value=20, value=10)

# -----------------📑 Filtered Data -----------------
filtered_data = data[
    (data['state'] == state) &
    (data['gender'] == gender) &
    (data['age_group'] == age_group)
]
grouped = filtered_data.groupby('year')['total'].sum().reset_index()
grouped.set_index('year', inplace=True)

# -----------------📈 Forecast Function -----------------
def forecast_arima(series, years=10):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=years)

    forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(years=1), periods=years, freq='YS')
    predicted = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return predicted, conf_int, forecast_index

# -----------------📊 Perform Forecast -----------------
predicted, conf_int, forecast_index = forecast_arima(grouped['total'], forecast_years)

# -----------------📉 Plotting -----------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(grouped.index, grouped['total'], label="Actual", color="#1f77b4", linewidth=2.5)
ax.plot(forecast_index, predicted, label="Forecast", color="#ff7f0e", linestyle='--', linewidth=2)
ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2)
ax.set_title(f"Suicide Forecast for {state} ({gender}, {age_group})", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Suicides")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -----------------📥 Download Forecast -----------------
with st.expander("📄 Download Forecast Data"):
    forecast_df = pd.DataFrame({
        'Year': forecast_index.year,
        'Predicted Suicides': predicted.round().astype(int)
    })
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{state}_{gender}_{age_group}_Forecast.csv",
        mime='text/csv'
    )

# -----------------🔻 Footer -----------------
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit & ARIMA</center>", unsafe_allow_html=True)
