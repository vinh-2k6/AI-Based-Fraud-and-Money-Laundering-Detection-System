import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


# =========================
# 1. Path
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "raw" / "SAML-D.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Load data
# =========================

print("Loading SAML-D...")

df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")


# =========================
# 3. Basic cleaning
# =========================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print(f"Shape after cleaning: {df.shape}")


# =========================
# 4. Create timestamp
# =========================

df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str)
)

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)


# =========================
# 5. Feature engineering
# =========================

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month


# =========================
# 6. Transform Amount
# =========================

# Reduce the effect of extremely large transactions
df["Amount_log"] = np.log1p(df["Amount"])


# =========================
# 7. Time-based Train/Test split
# =========================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

print("Train time range:")
print(train_df["timestamp"].min())
print(train_df["timestamp"].max())

print("Test time range:")
print(test_df["timestamp"].min())
print(test_df["timestamp"].max())


# =========================
# 8. Scale Amount
# =========================

scaler = MinMaxScaler()

# IMPORTANT:
# Fit scaler ONLY on training data
train_df["Amount_scaled"] = scaler.fit_transform(
    train_df[["Amount_log"]]
)

# Use the same scaler for test data
test_df["Amount_scaled"] = scaler.transform(
    test_df[["Amount_log"]]
)


# =========================
# 9. Remove columns
# =========================

columns_to_drop = [
    "Time",
    "Date",
    "Amount",
    "Amount_log",
    "timestamp",
    "Laundering_type"
]

train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)


# =========================
# 10. Save
# =========================

train_file = OUTPUT_DIR / "train_cleaned.csv"
test_file = OUTPUT_DIR / "test_cleaned.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print("\nPreprocessing completed.")

print(f"Train file: {train_file}")
print(f"Test file: {test_file}")