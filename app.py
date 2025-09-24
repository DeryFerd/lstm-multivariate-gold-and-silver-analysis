import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timedelta

# ======================
# 1. ARSITEKTUR MODEL (SAMA DENGAN DI COLAB)
# ======================
class GoldLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=1, output_horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4
        )
        self.fc = nn.Linear(hidden_dim, output_horizon)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ======================
# 2. LOAD MODEL & SCALER
# ======================
@st.cache_resource
def load_model():
    # Init model
    model = GoldLSTM()
    # Load weights
    checkpoint = torch.load('gold_lstm_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Load scaler
    scaler_X = joblib.load('scaler_X.pkl')
    lookback = checkpoint['lookback']
    feature_cols = checkpoint['feature_cols']
    return model, scaler_X, lookback, feature_cols

model, scaler_X, lookback, feature_cols = load_model()

# ======================
# 3. FUNGSI PREDIKSI
# ======================
def predict_next_gold_price(df, model, scaler_X, lookback, feature_cols):
    """
    Prediksi harga emas 1 hari ke depan
    Input: df harus punya kolom DATE, GOLD_PRICE, SILVER_PRICE, GPRD, GPRD_ACT, GPRD_THREAT
    """
    # Pastikan urut
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Ambil data terakhir sebanyak lookback
    recent_data = df[feature_cols].tail(lookback)
    
    # Normalisasi
    recent_scaled = scaler_X.transform(recent_data)
    
    # Bentuk input untuk model: (1, lookback, 5)
    X_input = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)
    
    # Prediksi log return
    with torch.no_grad():
        log_return_pred = model(X_input).item()
    
    # Konversi ke harga
    last_gold_price = df['GOLD_PRICE'].iloc[-1]
    predicted_price = last_gold_price * np.exp(log_return_pred)
    predicted_change_pct = (predicted_price - last_gold_price) / last_gold_price * 100
    
    return predicted_price, predicted_change_pct, log_return_pred

# ======================
# 4. UI STREAMLIT
# ======================
st.set_page_config(page_title="Gold Price Forecaster", page_icon="💰")
st.title("💰 Gold Price Forecaster (LSTM + Geopolitical Risk)")
st.markdown("""
Prediksi harga emas 1 hari ke depan menggunakan **LSTM Deep Learning** dengan faktor:
- Harga perak
- Indeks risiko geopolitik (GPRD, GPRD_ACT, GPRD_THREAT)
""")

# Upload data
uploaded_file = st.file_uploader("Upload historical data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Validasi kolom
    required_cols = ['DATE', 'GOLD_PRICE', 'SILVER_PRICE', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']
    if not all(col in df.columns for col in required_cols):
        st.error(f"File harus punya kolom: {required_cols}")
        st.stop()
    
    # Pastikan cukup data
    if len(df) < lookback:
        st.error(f"Data minimal {lookback} hari diperlukan!")
        st.stop()
    
    # Prediksi
    try:
        pred_price, pred_change, log_ret = predict_next_gold_price(df, model, scaler_X, lookback, feature_cols)
        
        # Tampilkan hasil
        last_date = df['DATE'].max()
        next_date = last_date + timedelta(days=1)
        
        st.subheader(f"Prediksi untuk {next_date.strftime('%Y-%m-%d')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Harga Emas", f"${pred_price:.2f}")
        col2.metric("Perubahan", f"{pred_change:.2f}%", 
                   delta_color="normal" if abs(pred_change) < 0.5 else ("inverse" if pred_change < 0 else "normal"))
        col3.metric("Log Return", f"{log_ret:.4f}")
        
        # Plot histori + prediksi
        st.subheader("Harga Emas Historis (30 Hari Terakhir)")
        plot_df = df[['DATE', 'GOLD_PRICE']].tail(30).copy()
        plot_df.loc[len(plot_df)] = [next_date, pred_price]  # tambahkan prediksi
        
        st.line_chart(plot_df.set_index('DATE'))
        
    except Exception as e:
        st.error(f"Error saat prediksi: {str(e)}")
else:
    st.info("Upload file CSV dengan kolom: DATE, GOLD_PRICE, SILVER_PRICE, GPRD, GPRD_ACT, GPRD_THREAT")
