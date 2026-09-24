import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Load dataset
# =========================

file_path = "data/raw/paysim.csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("1. DATASET INFORMATION")
print("=" * 50)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())


# =========================
# 2. Data types
# =========================

print("\n" + "=" * 50)
print("2. DATA TYPES")
print("=" * 50)

print(df.dtypes)


# =========================
# 3. Missing values
# =========================

print("\n" + "=" * 50)
print("3. MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# =========================
# 4. Duplicate rows
# =========================

print("\n" + "=" * 50)
print("4. DUPLICATES")
print("=" * 50)

print("Number of duplicate rows:", df.duplicated().sum())


# =========================
# 5. Transaction types
# =========================

print("\n" + "=" * 50)
print("5. TRANSACTION TYPES")
print("=" * 50)

print(df["type"].value_counts())


# =========================
# 6. Fraud distribution
# =========================

print("\n" + "=" * 50)
print("6. FRAUD DISTRIBUTION")
print("=" * 50)

fraud_counts = df["isFraud"].value_counts()

print(fraud_counts)

print("\nFraud rate:")
print(df["isFraud"].mean() * 100, "%")


# =========================
# 7. Amount statistics
# =========================

print("\n" + "=" * 50)
print("7. AMOUNT STATISTICS")
print("=" * 50)

print(df["amount"].describe())


# =========================
# 8. Balance statistics
# =========================

print("\n" + "=" * 50)
print("8. BALANCE STATISTICS")
print("=" * 50)

print("Origin balance:")
print(df["oldbalanceOrg"].describe())

print("\nDestination balance:")
print(df["oldbalanceDest"].describe())


# =========================
# 9. Fraud by transaction type
# =========================

print("\n" + "=" * 50)
print("9. FRAUD BY TRANSACTION TYPE")
print("=" * 50)

fraud_by_type = df.groupby("type")["isFraud"].agg(
    total_transactions="count",
    fraud_transactions="sum",
    fraud_rate="mean"
)

fraud_by_type["fraud_rate"] *= 100

print(fraud_by_type)


# =========================
# 10. Flagged fraud
# =========================

print("\n" + "=" * 50)
print("10. FLAGGED FRAUD")
print("=" * 50)

print(df["isFlaggedFraud"].value_counts())


# =========================
# 11. Account statistics
# =========================

print("\n" + "=" * 50)
print("11. ACCOUNT STATISTICS")
print("=" * 50)

print("Unique origin accounts:", df["nameOrig"].nunique())
print("Unique destination accounts:", df["nameDest"].nunique())

print(
    "Unique accounts involved:",
    len(set(df["nameOrig"]) | set(df["nameDest"]))
)

# =========================
# 12. CREATE FIGURES
# =========================

# Create folder for figures
figure_path = Path("data/processed/figures")
figure_path.mkdir(parents=True, exist_ok=True)


# =========================
# 12.1 Fraud Distribution
# =========================

fraud_counts = df["isFraud"].value_counts().sort_index()

plt.figure(figsize=(6, 5))

plt.bar(
    ["Normal", "Fraud"],
    [
        fraud_counts.get(0, 0),
        fraud_counts.get(1, 0)
    ]
)

plt.title("Fraud Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    figure_path / "fraud_distribution.png",
    dpi=300
)

plt.close()


# =========================
# 12.2 Transaction Type Distribution
# =========================

transaction_counts = df["type"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    transaction_counts.index,
    transaction_counts.values
)

plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    figure_path / "transaction_types.png",
    dpi=300
)

plt.close()


# =========================
# 12.3 Fraud Rate by Transaction Type
# =========================

fraud_rate_by_type = (
    df.groupby("type")["isFraud"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    fraud_rate_by_type.index,
    fraud_rate_by_type.values
)

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    figure_path / "fraud_by_type.png",
    dpi=300
)

plt.close()


# =========================
# 12.4 Transaction Amount Distribution
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    df["amount"],
    bins=100
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    figure_path / "amount_distribution.png",
    dpi=300
)

plt.close()


print("\n" + "=" * 50)
print("12. FIGURES")
print("=" * 50)

print("Figures saved to:", figure_path)