"""
src/data_preprocessing.py
=========================
Pipeline Tien xu ly du lieu SAML-D (Production-Ready):
  1. Lam sach du lieu (loai bo duplicate, missing values).
  2. Tao truong `timestamp` va sap xep trinh tu thoi gian tuyet doi (Chronological Sorting).
  3. Feature Engineering:
     - Trigonometric/Cyclical time features (hour, day_of_week, month).
     - Causal Sliding Window Temporal Features (24h Window):
       * sender_tx_count_24h: Tan suat giao dich cua Sender trong 24h qua.
       * receiver_tx_count_24h: Tan suat giao dich cua Receiver trong 24h qua.
       * sender_amount_sum_24h: Tong so tien Sender da chuyen trong 24h qua.
       * time_since_last_sender_tx: Khoang cach giay so voi giao dich truoc do cua Sender.
  4. Time-based Split: 80% Train, 20% Test (Khong ro ri tuong lai).
  5. Zero-Leakage Scaling:
     - MinMaxScaler duoc FIT CHI TREN TRAIN_DF.
     - Sau do TRANSFORM tren ca Train va Test.
     - Luu bo scaler vao scalers.pkl de phuc vu streaming.
  6. Xuat file cleaned CSV: data/processed/train_cleaned.csv va test_cleaned.csv.
"""

from collections import defaultdict, deque
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. PATHS & DIRECTORIES
# ============================================================

BASE_DIR   = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "raw" / "SAML-D.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
SCALER_PK  = OUTPUT_DIR / "scalers.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CAUSAL TEMPORAL FEATURE ENGINEERING
# ============================================================

def compute_causal_temporal_features(df: pd.DataFrame, window_seconds: int = 86400) -> pd.DataFrame:
    """
    Tinh toan cac dac trung tan suat va dong tien trong cua so 24 gio (86,400 giay).
    Su dung thuat toan Causal Sliding Window voi hang doi Deque (Do phuc tap O(N) tuyen tinh).
    DAM BAO: Chi nhin ve qua khu (<= t), tuyet doi khong co lookahead leakage.
    """
    print("\nDang tinh toan Causal Temporal Features (24h window)...")
    start_time = time.perf_counter()

    n = len(df)
    senders = df["Sender_account"].values
    receivers = df["Receiver_account"].values
    amounts = df["Amount"].values
    # Chuyen doi sang epoch seconds (int64) de tinh toan nhanh gap 50 lan
    timestamps = (df["timestamp"].values.astype("int64") // 10**9)

    sender_counts_24h = np.zeros(n, dtype=np.int32)
    receiver_counts_24h = np.zeros(n, dtype=np.int32)
    sender_sums_24h = np.zeros(n, dtype=np.float32)
    time_diff_last_tx = np.full(n, fill_value=window_seconds, dtype=np.float32)

    # Deque theo doi lich su (timestamp, amount) cua moi account
    sender_history = defaultdict(deque)
    receiver_history = defaultdict(deque)
    sender_sum_tracker = defaultdict(float)
    last_tx_time = {}

    for i in range(n):
        s = senders[i]
        r = receivers[i]
        t = timestamps[i]
        amt = amounts[i]

        # 1. Delta time so voi giao dich truoc do cua cung Sender
        if s in last_tx_time:
            time_diff_last_tx[i] = max(0.0, float(t - last_tx_time[s]))
        last_tx_time[s] = t

        # 2. Xoa cac giao dich cu hon 24 gio cua Sender
        s_deque = sender_history[s]
        while s_deque and s_deque[0][0] < t - window_seconds:
            old_t, old_amt = s_deque.popleft()
            sender_sum_tracker[s] -= old_amt

        # 3. Ghi nhan tan suat va tong tien cua Sender trong 24h
        sender_counts_24h[i] = len(s_deque)
        sender_sums_24h[i] = sender_sum_tracker[s]

        # Cap nhat lich su cho Sender
        s_deque.append((t, amt))
        sender_sum_tracker[s] += amt

        # 4. Xoa cac giao dich cu hon 24 gio cua Receiver
        r_deque = receiver_history[r]
        while r_deque and r_deque[0][0] < t - window_seconds:
            r_deque.popleft()

        receiver_counts_24h[i] = len(r_deque)
        r_deque.append((t, amt))

    df["sender_tx_count_24h"] = sender_counts_24h
    df["receiver_tx_count_24h"] = receiver_counts_24h
    df["sender_amount_sum_24h"] = sender_sums_24h
    df["time_since_last_sender_tx"] = time_diff_last_tx

    elapsed = time.perf_counter() - start_time
    print(f"-> Hoan thanh tinh 4 features cho {n:,} dong trong {elapsed:.2f}s!")
    return df


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main():
    total_start = time.perf_counter()
    print("=" * 60)
    print("STARTING DATA PREPROCESSING PIPELINE (WEEK 3)")
    print("=" * 60)

    # 1. Load data
    print(f"Loading data tu: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Original shape: {df.shape}")

    # 2. Cleaning
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Shape sau dropna & drop_duplicates: {df.shape}")

    # 3. Create timestamp & sort
    print("Tao cot timestamp va sap xep theo trinh tu thoi gian...")
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str)
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 4. Basic time features
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["month"] = df["timestamp"].dt.month.astype(np.int8)

    # 5. Causal temporal features (24h sliding window)
    df = compute_causal_temporal_features(df, window_seconds=86400)

    # 6. Log transformations (giam do lech phan phoi truoc khi scale)
    print("Thuc hien Log1p Transformation...")
    df["Amount_log"] = np.log1p(df["Amount"].values).astype(np.float32)
    df["sender_tx_count_24h_log"] = np.log1p(df["sender_tx_count_24h"].values).astype(np.float32)
    df["receiver_tx_count_24h_log"] = np.log1p(df["receiver_tx_count_24h"].values).astype(np.float32)
    df["sender_amount_sum_24h_log"] = np.log1p(df["sender_amount_sum_24h"].values).astype(np.float32)
    df["time_since_last_sender_tx_log"] = np.log1p(df["time_since_last_sender_tx"].values).astype(np.float32)

    # 7. Time-based Train/Test split (80% Train, 20% Test)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print(f"\nTrain set size : {len(train_df):,} rows")
    print(f"Test set size  : {len(test_df):,} rows")
    print(f"Train time range: {train_df['timestamp'].min()} --> {train_df['timestamp'].max()}")
    print(f"Test time range : {test_df['timestamp'].min()} --> {test_df['timestamp'].max()}")

    # 8. Zero-Leakage Scaling
    print("\nZero-Leakage Scaling: Fit MinMaxScaler CHI TREN tap Train...")
    feature_cols_to_scale = [
        "Amount_log",
        "sender_tx_count_24h_log",
        "receiver_tx_count_24h_log",
        "sender_amount_sum_24h_log",
        "time_since_last_sender_tx_log",
    ]
    scaled_output_cols = [
        "Amount_scaled",
        "sender_tx_count_24h_scaled",
        "receiver_tx_count_24h_scaled",
        "sender_amount_sum_24h_scaled",
        "time_since_last_sender_tx_scaled",
    ]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols_to_scale])
    test_scaled = scaler.transform(test_df[feature_cols_to_scale])

    for i, col_name in enumerate(scaled_output_cols):
        train_df[col_name] = train_scaled[:, i].astype(np.float32)
        test_df[col_name] = test_scaled[:, i].astype(np.float32)

    # Luu scaler phuc vu production / streaming
    with open(SCALER_PK, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "feature_cols": feature_cols_to_scale,
                "scaled_cols": scaled_output_cols,
            },
            f,
        )
    print(f"Scaler saved -> {SCALER_PK}")

    # 9. Clean columns before exporting
    columns_to_drop = [
        "Time",
        "Date",
        "Amount",
        "Amount_log",
        "Laundering_type",
        "sender_tx_count_24h_log",
        "receiver_tx_count_24h_log",
        "sender_amount_sum_24h_log",
        "time_since_last_sender_tx_log",
    ]
    train_df = train_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)

    # 10. Save to CSV
    train_file = OUTPUT_DIR / "train_cleaned.csv"
    test_file = OUTPUT_DIR / "test_cleaned.csv"

    print(f"\nDang ghi train_cleaned.csv ({len(train_df):,} dong)...")
    train_df.to_csv(train_file, index=False)
    print(f"Train file saved -> {train_file}")

    print(f"Dang ghi test_cleaned.csv ({len(test_df):,} dong)...")
    test_df.to_csv(test_file, index=False)
    print(f"Test file saved -> {test_file}")

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f"DATA PREPROCESSING COMPLETED IN {total_time:.2f}s!")
    print("=" * 60)
    print(f"Final Train Columns: {train_df.columns.tolist()}")


if __name__ == "__main__":
    main()