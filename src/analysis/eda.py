import pandas as pd
from pathlib import Path


# ============================================================
# 1. LOAD DATA
# ============================================================

# Lấy thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đường dẫn đến file SAML-D
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SAML-D.csv"

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. GENERAL INFORMATION
# ============================================================

print("=" * 70)
print("1. GENERAL INFORMATION")
print("=" * 70)

print(f"Dataset path: {DATA_PATH}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# Kiểu dữ liệu
print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percent
})

print(missing_table)


# ============================================================
# 4. DUPLICATED DATA
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATED DATA")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicated rows: {duplicate_count:,}")


# ============================================================
# 5. TARGET VARIABLE - IS_LAUNDERING
# ============================================================

print("\n" + "=" * 70)
print("4. CLASS DISTRIBUTION - IS_LAUNDERING")
print("=" * 70)

class_count = df["Is_laundering"].value_counts().sort_index()
class_percent = df["Is_laundering"].value_counts(
    normalize=True
).sort_index() * 100

class_distribution = pd.DataFrame({
    "Count": class_count,
    "Percentage": class_percent
})

print(class_distribution)

print("\nMeaning:")
print("0 = Normal transaction")
print("1 = Money laundering transaction")


# ============================================================
# 6. NUMERICAL FEATURE - AMOUNT
# ============================================================

print("\n" + "=" * 70)
print("5. AMOUNT STATISTICS")
print("=" * 70)

print(df["Amount"].describe())


# So sánh Amount giữa giao dịch bình thường và laundering
print("\nAmount by Is_laundering:")
print(
    df.groupby("Is_laundering")["Amount"]
    .agg(["count", "mean", "median", "min", "max"])
)


# ============================================================
# 7. CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Laundering_type"
]

for column in categorical_columns:

    print("\n" + "=" * 70)
    print(f"6. VALUE DISTRIBUTION - {column}")
    print("=" * 70)

    print(df[column].value_counts(dropna=False))


# ============================================================
# 8. LAUNDERING TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. LAUNDERING TYPE ANALYSIS")
print("=" * 70)

laundering_analysis = (
    df.groupby("Laundering_type")
    .agg(
        transaction_count=("Laundering_type", "size"),
        laundering_count=("Is_laundering", "sum")
    )
)

laundering_analysis["laundering_percentage"] = (
    laundering_analysis["laundering_count"]
    / laundering_analysis["transaction_count"]
    * 100
)

print(laundering_analysis.sort_values(
    by="laundering_count",
    ascending=False
))


# ============================================================
# 9. PAYMENT TYPE VS LAUNDERING
# ============================================================

print("\n" + "=" * 70)
print("8. PAYMENT TYPE VS LAUNDERING")
print("=" * 70)

payment_laundering = pd.crosstab(
    df["Payment_type"],
    df["Is_laundering"],
    normalize="index"
) * 100

print(payment_laundering)


# ============================================================
# 10. CURRENCY VS LAUNDERING
# ============================================================

print("\n" + "=" * 70)
print("9. PAYMENT CURRENCY VS LAUNDERING")
print("=" * 70)

currency_laundering = pd.crosstab(
    df["Payment_currency"],
    df["Is_laundering"],
    normalize="index"
) * 100

print(currency_laundering)


# ============================================================
# 11. BANK LOCATION VS LAUNDERING
# ============================================================

print("\n" + "=" * 70)
print("10. SENDER BANK LOCATION VS LAUNDERING")
print("=" * 70)

sender_location_laundering = pd.crosstab(
    df["Sender_bank_location"],
    df["Is_laundering"],
    normalize="index"
) * 100

print(sender_location_laundering)


print("\n" + "=" * 70)
print("11. RECEIVER BANK LOCATION VS LAUNDERING")
print("=" * 70)

receiver_location_laundering = pd.crosstab(
    df["Receiver_bank_location"],
    df["Is_laundering"],
    normalize="index"
) * 100

print(receiver_location_laundering)


# ============================================================
# 12. ACCOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("12. ACCOUNT ANALYSIS")
print("=" * 70)

sender_unique = df["Sender_account"].nunique()
receiver_unique = df["Receiver_account"].nunique()

all_accounts = pd.concat([
    df["Sender_account"],
    df["Receiver_account"]
]).nunique()

print(f"Unique sender accounts: {sender_unique:,}")
print(f"Unique receiver accounts: {receiver_unique:,}")
print(f"Unique accounts overall: {all_accounts:,}")


# Số giao dịch của mỗi sender
sender_transaction_count = (
    df["Sender_account"]
    .value_counts()
)

print("\nTop 10 sender accounts by transaction count:")
print(sender_transaction_count.head(10))


# Số giao dịch của mỗi receiver
receiver_transaction_count = (
    df["Receiver_account"]
    .value_counts()
)

print("\nTop 10 receiver accounts by transaction count:")
print(receiver_transaction_count.head(10))


# ============================================================
# 13. TRANSACTION EDGE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("13. TRANSACTION EDGE ANALYSIS")
print("=" * 70)

unique_edges = df[
    ["Sender_account", "Receiver_account"]
].drop_duplicates()

print(f"Unique Sender -> Receiver relationships: {len(unique_edges):,}")


# Những cặp tài khoản giao dịch nhiều nhất
edge_frequency = (
    df.groupby(
        ["Sender_account", "Receiver_account"]
    )
    .size()
    .reset_index(name="transaction_count")
    .sort_values(
        "transaction_count",
        ascending=False
    )
)

print("\nTop 10 Sender -> Receiver relationships:")
print(edge_frequency.head(10))


# ============================================================
# 14. TIME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("14. TIME ANALYSIS")
print("=" * 70)

print("Time statistics:")
print(df["Time"].describe())

print("\nDate range:")

# Chuyển Date sang datetime để kiểm tra
date_column = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

print(f"Minimum date: {date_column.min()}")
print(f"Maximum date: {date_column.max()}")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum():,}")
print(f"Duplicated rows: {df.duplicated().sum():,}")
print(f"Unique accounts: {all_accounts:,}")
print(f"Laundering transactions: {class_count.get(1, 0):,}")
print(f"Normal transactions: {class_count.get(0, 0):,}")

print("\nEDA analysis finished successfully.")

# ============================================================
# GRAPH PATTERN ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("GRAPH PATTERN ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. FAN-OUT
# Một sender gửi tiền đến nhiều receiver khác nhau
# ------------------------------------------------------------

print("\n--- FAN-OUT ANALYSIS ---")

sender_degree = (
    df.groupby("Sender_account")["Receiver_account"]
    .nunique()
    .sort_values(ascending=False)
)

print("Top 20 accounts with highest number of receivers:")
print(sender_degree.head(20))

# Fan-out >= 5
fan_out_accounts = sender_degree[sender_degree >= 5]

print(f"\nAccounts with fan-out >= 5: {len(fan_out_accounts)}")

# So sánh laundering transactions
laundering_df = df[df["Is_laundering"] == 1]

laundering_sender_degree = (
    laundering_df.groupby("Sender_account")["Receiver_account"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTop laundering senders by number of receivers:")
print(laundering_sender_degree.head(20))


# ------------------------------------------------------------
# 2. FAN-IN
# Nhiều sender gửi tiền vào cùng một receiver
# ------------------------------------------------------------

print("\n--- FAN-IN ANALYSIS ---")

receiver_degree = (
    df.groupby("Receiver_account")["Sender_account"]
    .nunique()
    .sort_values(ascending=False)
)

print("Top 20 accounts with highest number of senders:")
print(receiver_degree.head(20))

# Fan-in >= 5
fan_in_accounts = receiver_degree[receiver_degree >= 5]

print(f"\nAccounts with fan-in >= 5: {len(fan_in_accounts)}")

laundering_receiver_degree = (
    laundering_df.groupby("Receiver_account")["Sender_account"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTop laundering receivers by number of senders:")
print(laundering_receiver_degree.head(20))


# ------------------------------------------------------------
# 3. LAUNDERING GRAPH PATTERNS
# Phân tích các loại pattern được gắn nhãn trong SAML-D
# ------------------------------------------------------------

print("\n--- LAUNDERING GRAPH PATTERNS ---")

graph_pattern_keywords = [
    "Layered",
    "Cycle",
    "Fan_In",
    "Fan_Out",
    "Scatter-Gather",
    "Gather-Scatter",
    "Bipartite",
    "Stacked",
    "Smurfing",
    "Structuring"
]

pattern_df = laundering_df[
    laundering_df["Laundering_type"]
    .str.contains("|".join(graph_pattern_keywords), case=False, na=False)
]

pattern_counts = (
    pattern_df["Laundering_type"]
    .value_counts()
)

print("Potential graph-related laundering patterns:")
print(pattern_counts)


# ------------------------------------------------------------
# 4. CYCLE ANALYSIS
# Kiểm tra A -> B và B -> A
# ------------------------------------------------------------

print("\n--- CYCLE ANALYSIS ---")

# Chỉ lấy unique edges
edges = df[["Sender_account", "Receiver_account"]].drop_duplicates()

# Đổi tên để self-join
reverse_edges = edges.rename(
    columns={
        "Sender_account": "Receiver_account",
        "Receiver_account": "Sender_account"
    }
)

# Tìm A -> B và B -> A
two_cycles = edges.merge(
    reverse_edges,
    on=["Sender_account", "Receiver_account"],
    how="inner"
)

# Loại bỏ self-loop A -> A
two_cycles = two_cycles[
    two_cycles["Sender_account"] != two_cycles["Receiver_account"]
]

# Mỗi cycle A <-> B xuất hiện 2 lần
two_cycles_unique = two_cycles[
    two_cycles["Sender_account"]
    < two_cycles["Receiver_account"]
]

print(f"Number of 2-node cycles (A -> B -> A): {len(two_cycles_unique)}")

if len(two_cycles_unique) > 0:
    print("\nExamples:")
    print(two_cycles_unique.head(20))


# ------------------------------------------------------------
# 5. CYCLE PATTERN TRONG LAUNDERING TRANSACTIONS
# ------------------------------------------------------------

print("\n--- LAUNDERING CYCLE PATTERN ---")

cycle_laundering = laundering_df[
    laundering_df["Laundering_type"]
    .str.contains("Cycle", case=False, na=False)
]

print(
    f"Transactions labeled as Cycle: "
    f"{len(cycle_laundering)}"
)

if len(cycle_laundering) > 0:
    print("\nCycle transactions:")
    print(
        cycle_laundering[
            [
                "Sender_account",
                "Receiver_account",
                "Amount",
                "Laundering_type"
            ]
        ].head(20)
    )


# ------------------------------------------------------------
# 6. FAN-IN / FAN-OUT CỦA CÁC GIAO DỊCH LAUNDERING
# ------------------------------------------------------------

print("\n--- LAUNDERING FAN-IN / FAN-OUT ---")

print(
    "Laundering transactions:",
    len(laundering_df)
)

print(
    "Unique laundering senders:",
    laundering_df["Sender_account"].nunique()
)

print(
    "Unique laundering receivers:",
    laundering_df["Receiver_account"].nunique()
)

print(
    "Unique laundering edges:",
    laundering_df[
        ["Sender_account", "Receiver_account"]
    ].drop_duplicates().shape[0]
)