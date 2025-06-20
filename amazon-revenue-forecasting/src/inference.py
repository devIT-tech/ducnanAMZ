import torch
import numpy as np
import os
from model import SalesLSTM

# Thư mục chứa các mô hình đã huấn luyện
MODEL_DIR = "best_model"

def load_model(asin_id: str) -> SalesLSTM:
   
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{asin_id}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")

    model = SalesLSTM()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_sales(asin_id: str, input_sequence: list) -> float:
    """
    Dự báo doanh thu cho ngày tiếp theo dựa trên chuỗi đầu vào gần nhất.

    Parameters:
        asin_id (str): Mã sản phẩm.
        input_sequence (list): Dữ liệu doanh thu 30 ngày gần nhất, dạng [[x], [x], ..., [x]].

    Returns:
        float: Doanh thu dự báo cho ngày tiếp theo.
    """
    model = load_model(asin_id)

    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # [1, 30, 1]

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().item()

    return prediction
