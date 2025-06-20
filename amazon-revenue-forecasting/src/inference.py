import torch
import numpy as np
from model import SalesLSTM
import os

MODEL_DIR = r"best_model"  # Thư mục chứa mô hình đã lưu

# Hàm load mô hình
def load_model(asin_id):
    model = SalesLSTM()
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{asin_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for ASIN {asin_id} not found")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Hàm dự báo doanh thu cho 1 mã sản phẩm
def predict_sales(asin_id, input_sequence):
    model = load_model(asin_id)
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 30, 1)
    with torch.no_grad():
        output = model(input_tensor).squeeze().item()  # Dự báo cho ngày tiếp theo
    return output có phải lỗi ở đây ko
