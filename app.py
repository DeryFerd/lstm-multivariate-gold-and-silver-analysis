import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import timedelta
import plotly.graph_objects as go

# ======================
# 1. ARSITEKTUR MODEL
# ======================
class GoldLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ======================
# 2. LOAD SEMUA ASSET
# ======================
@st.cache_resource
def load_assets():
    model = GoldLSTM()
    checkpoint = torch.load('gold_lstm_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_X = joblib.load('scaler_X.pkl')
    df = pd.read_csv('sample_data.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    return model, scaler_X, df

model, scaler_X, full_df = load_assets()
lookback = 14
feature_cols = ['GOLD_PRICE', 'SILVER_PRICE', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']

# ======================
# 3. FUNGSI PREDIKSI
# ======================
def predict_from_date(df, end_date, model, scaler_X, lookback, feature_cols):
    df_until = df[df['DATE'] <= end_date].copy()
    if len(df_until) < lookback:
        return None, None, None, None
    
    recent = df_until[feature_cols].tail(lookback)
    recent_scaled = scaler_X.transform(recent)
    X_input = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        log_ret = model(X_input).item()
    
    last_price = df_until['GOLD_PRICE'].iloc[-1]
    pred_price = last_price * np.exp(log_ret)
    pred_change = (pred_price - last_price) / last_price * 100
    
    return last_price, pred_price, pred_change, log_ret

# ======================
# 4. UI FINAL - SUPER RAMAI!
# ======================
st.set_page_config(page_title="💰 Gold Forecaster", page_icon="💰", layout="wide")
st.markdown("""
<h1 style='text-align: center; color: #FFD700; text-shadow: 0 0 10px rgba(255,215,0,0.5);'>
    💰 Gold Price Forecaster
</h1>
<div style='text-align: center; color: #888; margin-bottom: 30px; font-size: 1.1em;'>
    <b>Deep Learning LSTM + Geopolitical Risk Intelligence</b><br>
    Prediksi harga emas 1 hari ke depan berdasarkan data historis & risiko geopolitik global
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Gold_coin_geometric_shine.png/220px-Gold_coin_geometric_shine.png", width=150)
    st.title("ℹ️ About")
    st.markdown("""
    - **Model**: Custom LSTM Deep Learning
    - **Input**: Harga perak + 3 indeks risiko geopolitik
    - **Data**: 1985–2024 (40 tahun histori)
    - **Akurasi**: ±1.9% error harian
    """)
    st.info("💡 Geser slider untuk lihat prediksi di tanggal berbeda!")

# Slider tanggal
min_date = full_df['DATE'].min().date()
max_date = full_df['DATE'].max().date() - timedelta(days=1)
selected_date = st.slider(
    "Pilih tanggal historis untuk prediksi **hari berikutnya**:",
    min_value=min_date,
    max_value=max_date,
    value=max_date,
    format="YYYY-MM-DD"
)

# Prediksi
last_price, pred_price, pred_change, log_ret = predict_from_date(
    full_df, pd.Timestamp(selected_date), model, scaler_X, lookback, feature_cols
)

if last_price is not None:
    next_date = selected_date + timedelta(days=1)
    last_date = selected_date
    
    # === PERBANDINGAN HISTORIS vs PREDIKSI ===
    st.subheader("📊 Perbandingan Harga Emas")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        label=f"Harga Historis\n({last_date.strftime('%Y-%m-%d')})",
        value=f"${last_price:.2f}"
    )
    
    col2.metric(
        label=f"Prediksi\n({next_date.strftime('%Y-%m-%d')})",
        value=f"${pred_price:.2f}",
        delta=f"{pred_change:.2f}%",
        delta_color="inverse" if pred_change < 0 else "normal"
    )
    
    abs_change = pred_price - last_price
    col3.metric(
        label="Perubahan (USD)",
        value=f"${abs_change:.2f}"
    )
    
    # === GPRD INSIGHT ===
    gprd_val = full_df[full_df['DATE'] <= pd.Timestamp(selected_date)]['GPRD'].iloc[-1]
    st.markdown("---")
    if gprd_val > 150:
        st.warning(f"⚠️ **Peringatan**: Risiko geopolitik sangat tinggi ({gprd_val:.1f}) — volatilitas emas mungkin meningkat!")
    elif gprd_val < 50:
        st.success(f"✅ **Stabil**: Risiko geopolitik rendah ({gprd_val:.1f}) — pasar cenderung tenang.")
    else:
        st.info(f"ℹ️ Risiko geopolitik sedang ({gprd_val:.1f})")
    
    # === GRAFIK KONTEKS 60 HARI ===
    st.markdown("---")
    st.subheader("📈 Konteks Historis (60 Hari Terakhir)")
    
    plot_days = 60
    start_plot = pd.Timestamp(selected_date) - timedelta(days=plot_days)
    plot_df = full_df[(full_df['DATE'] >= start_plot) & (full_df['DATE'] <= pd.Timestamp(selected_date))].copy()
    
    # Tambahkan titik prediksi
    pred_row = pd.DataFrame({
        'DATE': [pd.Timestamp(next_date)],
        'GOLD_PRICE': [pred_price]
    })
    plot_df = pd.concat([plot_df, pred_row], ignore_index=True)
    
    fig = go.Figure()
    # Historis
    fig.add_trace(go.Scatter(
        x=plot_df[:-1]['DATE'],
        y=plot_df[:-1]['GOLD_PRICE'],
        mode='lines+markers',
        name='Harga Historis',
        line=dict(color='#FFD700', width=3),
        marker=dict(size=4)
    ))
    # Prediksi
    fig.add_trace(go.Scatter(
        x=plot_df[-1:]['DATE'],
        y=plot_df[-1:]['GOLD_PRICE'],
        mode='markers+text',
        name='Prediksi',
        marker=dict(color='red', size=12, symbol='star'),
        text=[f"${pred_price:.0f}"],
        textposition="top center"
    ))
    
    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Harga Emas (USD)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Tidak cukup data historis untuk prediksi di tanggal ini!")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; color: #888; font-size: 0.9em;'>
    Dibuat dengan ❤️ menggunakan PyTorch & Streamlit | Data: Kaggle Geopolitical Risk Index
</div>
""", unsafe_allow_html=True)
