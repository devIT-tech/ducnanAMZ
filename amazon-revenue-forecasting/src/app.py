import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SalesLSTM

# ===== Cấu hình đường dẫn tuyệt đối (an toàn) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc
DATA_DIR = os.path.join(BASE_DIR, "data", "sales_history")
MODEL_DIR = os.path.join(BASE_DIR, "best_models")  


# ===== Hàm tải mô hình và dự báo =====
def predict_sales(asin_id, input_seq):
    model = SalesLSTM()
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{asin_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # shape: [1, 30, 1]
    with torch.no_grad():
        prediction = model(input_tensor).squeeze().item()
    return prediction

# ===== Danh sách sản phẩm có sẵn =====
def get_available_products():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")
    files = os.listdir(DATA_DIR)
    return [f.replace("_sales_history.csv", "") for f in files if f.endswith("_sales_history.csv")]

# ===== Giao diện Streamlit =====
st.set_page_config(page_title="Amazon Sales Forecast", layout="wide")
st.title("📦 Amazon Sales Forecasting")
st.markdown("Chọn một mã sản phẩm để xem lịch sử doanh thu và dự báo cho ngày tiếp theo.")

# ===== Chọn sản phẩm =====
try:
    available_asins = get_available_products()
    asin_id = st.selectbox("🔍 Chọn mã sản phẩm (ASIN):", available_asins)

    # ===== Đọc dữ liệu lịch sử =====
    csv_path = os.path.join(DATA_DIR, f"{asin_id}_sales_history.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # ===== Hiển thị thông tin sản phẩm =====
    with st.expander("🛒 Thông tin sản phẩm"):
        st.write(f"**Tên sản phẩm:** {df['title'].iloc[0]}")
        st.write(f"**Giá:** {df['price'].iloc[0]}")
        st.write(f"**Xếp hạng trung bình:** {df['average_rating'].iloc[0]}")
        st.write(f"**Tổng số lượt đánh giá:** {df['num_reviews'].iloc[0]}")

    # ===== Biểu đồ doanh thu theo ngày =====
    st.subheader("📈 Doanh thu theo ngày")
    st.line_chart(df.set_index("date")["daily_revenue"])

    # ===== Dự báo doanh thu =====
    if len(df) >= 30:
        recent_sales = df["daily_revenue"].values.reshape(-1, 1).astype(np.float32)
        recent_sales = (recent_sales - recent_sales.min()) / (df["daily_revenue"].max() - df["daily_revenue"].min() + 1e-6)
        recent_sales = [[x[0]] for x in recent_sales[-30:]]

        try:
            prediction = predict_sales(asin_id, recent_sales)
            prediction_scaled = prediction * (df["daily_revenue"].max() - df["daily_revenue"].min()) + df["daily_revenue"].min()
            prediction_scaled = abs(prediction_scaled)
            st.success(f"📊 Dự báo doanh thu ngày tiếp theo: **${prediction_scaled:.2f}**")
        except Exception as e:
            st.error(f"❌ Không thể dự báo: {str(e)}")
    else:
        st.warning("⚠️ Cần ít nhất 30 ngày doanh thu để dự báo.")

except FileNotFoundError as e:
    st.error(f"❌ Lỗi: {str(e)} – Vui lòng kiểm tra lại thư mục `data/sales_history`.")

