import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

print("Đang nạp dữ liệu...")
train_df = pd.read_csv("data/processed/train_cleaned.csv")
test_df = pd.read_csv("data/processed/test_cleaned.csv")

# 1. Xử lý các cột dữ liệu dạng chuỗi 
categorical_cols = ['Payment_currency', 'Received_currency', 
                    'Sender_bank_location', 'Receiver_bank_location', 'Payment_type']

for col in categorical_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

# 2. Loại bỏ cột timestamp (không phù hợp để đưa trực tiếp vào XGBoost)
if 'timestamp' in train_df.columns:
    train_df = train_df.drop(columns=['timestamp'])
    test_df = test_df.drop(columns=['timestamp'])

# 3. Tách X, y 
X_train = train_df.drop(columns=['Is_laundering'])
y_train = train_df['Is_laundering']

X_test = test_df.drop(columns=['Is_laundering'])
y_test = test_df['Is_laundering']

print("Bắt đầu huấn luyện XGBoost Baseline...")
# 4. Thêm tham số enable_categorical=True
model = xgb.XGBClassifier(
    scale_pos_weight=100, 
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    enable_categorical=True,  # Bật tính năng đọc dữ liệu danh mục
    tree_method='hist'        # Tham số  khi dùng enable_categorical
)
model.fit(X_train, y_train)

print("Đang đánh giá trên tập Test...")
y_pred = model.predict(X_test)

print("\n=== MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ===")
print(classification_report(y_test, y_pred))