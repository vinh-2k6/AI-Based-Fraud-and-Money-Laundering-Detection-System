"""
graph_builder.py
================
Chuyen doi du lieu giao dich sang dinh dang do thi PyTorch Geometric (Production-Ready).

Kien truc:
  1. [Inductive Graph]    : Mapping ID dong ho tro cac nut moi chua tung xuat hien.
  2. [Cyclical Encoding]  : Ma hoa chu ky luong giac (Sine/Cosine) cho thoi gian.
  3. [One-Hot Encoding]   : OHE cho bien phan loai tren canh, xu ly theo chunk de tranh OOM.
  4. [Cold-start Handling]: Them mask `is_new_node` chong nhieu ty le gian lan mac dinh.
  5. [No Data Leakage]    : Node structural stats & fraud history deu chi tinh tu train_df.
  6. [NaN-safe Indexing]  : Kiem tra notna() truoc moi phep gan index de chong crash.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import pickle


# ============================================================
# 0. PATHS & CONFIG
# ============================================================

BASE_DIR   = Path(__file__).resolve().parents[1]
PROC_DIR   = BASE_DIR / "data" / "processed"

TRAIN_CSV  = PROC_DIR / "train_cleaned.csv"
TEST_CSV   = PROC_DIR / "test_cleaned.csv"

TRAIN_PT   = PROC_DIR / "data_train.pt"
TEST_PT    = PROC_DIR / "data_test.pt"
ENCODER_PK = PROC_DIR / "edge_encoders.pkl"
MAPPING_PK = PROC_DIR / "account_mapping.pkl"

# Xu ly OHE theo tung batch de tranh tao mang numpy qua lon (>2GB)
# 500_000 rows x 74 OHE cols x 4 bytes ~ 148 MB/chunk - an toan tren moi may
OHE_CHUNK_SIZE = 500_000

CATEGORICAL_COLS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"Train shape : {train_df.shape}")
print(f"Test  shape : {test_df.shape}")


# ============================================================
# 2. INDUCTIVE ID MAPPING (DO THI MO)
# ============================================================

def build_account_mapping(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict, set]:
    """
    Xay dung tu dien anh xa account_id -> node_index theo chien luoc inductive:
      - Buoc 1: Dang ky tat ca tai khoan tu tap Train -> Base Graph.
      - Buoc 2: Mo rong cho cac tai khoan moi xuat hien trong Test.

    Returns:
        account_to_idx      : dict[int, int] - anh xa toan cuc
        known_train_accounts: set[int]       - tap tai khoan da biet tu Train
    """
    account_to_idx: dict[int, int] = {}
    known_train_accounts: set[int] = set()

    # Buoc 1 - Base Graph tu Train
    train_accounts = pd.concat([
        train_df["Sender_account"],
        train_df["Receiver_account"],
    ]).unique()

    for acc in train_accounts:
        acc_id = int(acc)
        account_to_idx[acc_id] = len(account_to_idx)
        known_train_accounts.add(acc_id)

    train_node_count = len(account_to_idx)

    # Buoc 2 - Mo rong cho Test (mo phong moi truong streaming)
    test_accounts = pd.concat([
        test_df["Sender_account"],
        test_df["Receiver_account"],
    ]).unique()

    new_nodes_in_test = 0
    for acc in test_accounts:
        acc_id = int(acc)
        if acc_id not in account_to_idx:
            account_to_idx[acc_id] = len(account_to_idx)
            new_nodes_in_test += 1

    print(f"Known Train accounts (Base Graph)  : {train_node_count:,}")
    print(f"Unseen accounts added from Test     : {new_nodes_in_test:,}")
    print(f"Total unique accounts (Global)      : {len(account_to_idx):,}")

    return account_to_idx, known_train_accounts


# ============================================================
# 3. OHE ENCODER - Fit chi tren Train
# ============================================================

def fit_ohe_encoder(train_df: pd.DataFrame) -> OneHotEncoder:
    """Fit OneHotEncoder chi tren tap Train de tranh data leakage."""
    ohe = OneHotEncoder(
        handle_unknown="ignore",   # Gia tri chua gap trong Test -> vector zero
        sparse_output=False,
        dtype=np.float32,
    )
    ohe.fit(train_df[CATEGORICAL_COLS])
    total_cats = sum(len(c) for c in ohe.categories_)
    print(f"One-Hot Encoder fitted: {len(CATEGORICAL_COLS)} cols, {total_cats} total categories.")
    return ohe


# ============================================================
# 4. EDGE FEATURE ENGINEERING (CHUNKED de tranh OOM)
# ============================================================

def process_edge_features(
    df: pd.DataFrame,
    ohe: OneHotEncoder,
    chunk_size: int = OHE_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Xu ly dac trung canh theo tung chunk de tranh OOM voi dataset lon.

    Feature layout:
        [0]       Amount_scaled                (1 chieu)
        [1..6]    sin/cos cua hour, dow, month (6 chieu)
        [7..]     One-Hot categorical           (tong so category)

    Args:
        df         : DataFrame chua cac cot can thiet.
        ohe        : OneHotEncoder da fit tren train.
        chunk_size : So hang xu ly moi lan de kiem soat RAM.

    Returns:
        edge_attr  : torch.Tensor [E, F_edge]
    """
    chunk_tensors = []

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]

        # a. Amount (lien tuc, da scale)
        amount_np = chunk[["Amount_scaled"]].values.astype(np.float32)

        # b. Cyclical time encoding
        #    sin/cos dam bao hour=23 va hour=0 gan nhau ve khoang cach so hoc
        time_parts = []
        for col, max_val in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
            vals = chunk[col].values.astype(np.float32)
            time_parts.append(np.sin(2.0 * np.pi * vals / max_val))
            time_parts.append(np.cos(2.0 * np.pi * vals / max_val))
        time_np = np.column_stack(time_parts)  # [chunk, 6]

        # c. One-Hot categorical (handle_unknown='ignore' -> zero cho gia tri la)
        cat_np = ohe.transform(chunk[CATEGORICAL_COLS])  # [chunk, total_cats]

        # Gop lai va chuyen sang tensor ngay -> giai phong numpy ngay sau do
        chunk_np = np.hstack([amount_np, time_np, cat_np])
        chunk_tensors.append(torch.tensor(chunk_np, dtype=torch.float))

        # Giai phong bo nho numpy cua chunk nay truoc khi xu ly chunk tiep
        del amount_np, time_np, cat_np, chunk_np

    return torch.cat(chunk_tensors, dim=0)


# ============================================================
# 5. NODE FEATURES (CHONG LEAKAGE & COLD-START)
# ============================================================

def build_node_features(
    structural_ref_df: pd.DataFrame,
    fraud_ref_df: pd.DataFrame,
    acc_to_idx: dict,
    known_train_accounts: set,
) -> torch.Tensor:
    """
    Tinh node features cho moi tai khoan (node) trong do thi.

    De dam bao tinh nhat quan va tranh leakage, CA HAI nguon tham chieu
    (`structural_ref_df` va `fraud_ref_df`) phai la `train_df`:
      - Structural stats: Neu dung test_df, distribution cua node features
        train va test se khac nhau -> model bi lech.
      - Fraud history   : Neu dung test_df, ty le gian lan 'tuong lai' bi ro ri.

    Feature layout [N, 8]:
        [0] out_degree       - So lan la sender trong structural_ref_df
        [1] in_degree        - So lan la receiver trong structural_ref_df
        [2] total_degree     - out + in
        [3] avg_amount_sent  - Amount_scaled trung binh khi gui
        [4] avg_amount_recv  - Amount_scaled trung binh khi nhan
        [5] fraud_out_ratio  - Ti le giao dich gui bi gan nhan laundering (tu fraud_ref_df)
        [6] fraud_in_ratio   - Ti le giao dich nhan bi gan nhan laundering (tu fraud_ref_df)
        [7] is_new_node      - 1.0 neu khong co trong tap Train goc

    Args:
        structural_ref_df    : DataFrame de tinh degree/amount (nen la train_df).
        fraud_ref_df         : DataFrame de tinh fraud ratio (bat buoc la train_df).
        acc_to_idx           : Tu dien account_id -> node index.
        known_train_accounts : Tap tai khoan da biet tu Train.

    Returns:
        torch.Tensor [N, 8]
    """
    n = len(acc_to_idx)
    features = np.zeros((n, 8), dtype=np.float32)

    # --- Helper: gan gia tri an toan, bo qua acc khong co trong acc_to_idx ---
    def safe_assign(account_series: pd.Series, value_series: pd.Series, col_idx: int) -> None:
        mapped = account_series.map(acc_to_idx)
        valid  = mapped.notna()                       # Loai bo acc khong tim thay
        idxs   = mapped[valid].astype(int).values
        vals   = value_series[valid].values
        features[idxs, col_idx] = vals

    # --- Structural stats (tu structural_ref_df = train_df) ---
    sender_stats = (
        structural_ref_df
        .groupby("Sender_account")
        .agg(out_cnt=("Amount_scaled", "size"), out_amt=("Amount_scaled", "mean"))
        .reset_index()
    )
    safe_assign(sender_stats["Sender_account"], sender_stats["out_cnt"], 0)
    safe_assign(sender_stats["Sender_account"], sender_stats["out_amt"], 3)

    receiver_stats = (
        structural_ref_df
        .groupby("Receiver_account")
        .agg(in_cnt=("Amount_scaled", "size"), in_amt=("Amount_scaled", "mean"))
        .reset_index()
    )
    safe_assign(receiver_stats["Receiver_account"], receiver_stats["in_cnt"], 1)
    safe_assign(receiver_stats["Receiver_account"], receiver_stats["in_amt"], 4)

    features[:, 2] = features[:, 0] + features[:, 1]  # total_degree

    # --- Fraud history (tu fraud_ref_df = train_df, KHONG DUOC DUNG test_df) ---
    sender_fraud = (
        fraud_ref_df
        .groupby("Sender_account")
        .agg(out_fraud=("Is_laundering", "mean"))
        .reset_index()
    )
    safe_assign(sender_fraud["Sender_account"], sender_fraud["out_fraud"], 5)

    receiver_fraud = (
        fraud_ref_df
        .groupby("Receiver_account")
        .agg(in_fraud=("Is_laundering", "mean"))
        .reset_index()
    )
    safe_assign(receiver_fraud["Receiver_account"], receiver_fraud["in_fraud"], 6)

    # --- Cold-start mask: vectorized O(new_nodes) thay vi O(N) ---
    new_node_indices = np.fromiter(
        (idx for acc_id, idx in acc_to_idx.items() if acc_id not in known_train_accounts),
        dtype=np.int64,
    )
    if len(new_node_indices) > 0:
        features[new_node_indices, 7] = 1.0

    return torch.tensor(features, dtype=torch.float)


# ============================================================
# 6. BUILD GRAPH
# ============================================================

def build_graph(
    df: pd.DataFrame,
    structural_ref_df: pd.DataFrame,
    fraud_ref_df: pd.DataFrame,
    acc_to_idx: dict,
    known_train_accounts: set,
    ohe: OneHotEncoder,
    split_name: str,
) -> Data:
    """
    Tao mot doi tuong Data cua PyTorch Geometric tu DataFrame.

    Args:
        df                   : DataFrame cua split can build (train hoac test).
        structural_ref_df    : Nguon tinh structural node stats (luon la train_df).
        fraud_ref_df         : Nguon tinh fraud history (luon la train_df).
        acc_to_idx           : Tu dien account_id -> node index.
        known_train_accounts : Tap tai khoan da biet tu Train.
        ohe                  : OneHotEncoder da fit.
        split_name           : Ten split de log.

    Returns:
        Data(x, edge_index, edge_attr, y, num_nodes)
    """
    print(f"\n[{split_name.upper()}] Building graph components...")

    # 1. Node features [N, 8]
    x = build_node_features(
        structural_ref_df=structural_ref_df,
        fraud_ref_df=fraud_ref_df,
        acc_to_idx=acc_to_idx,
        known_train_accounts=known_train_accounts,
    )

    # 2. Edge index [2, E]
    src = df["Sender_account"].map(acc_to_idx).values.astype(np.int64)
    dst = df["Receiver_account"].map(acc_to_idx).values.astype(np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # 3. Edge features [E, F_edge] - xu ly theo chunk
    edge_attr = process_edge_features(df, ohe)

    # 4. Labels [E]
    y = torch.tensor(df["Is_laundering"].values, dtype=torch.long)

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = y,
        num_nodes  = len(acc_to_idx),
    )

    n_fraud = int(y.sum())
    n_total = len(y)
    print(f"  Nodes      : {data.num_nodes:,}  (feature dim : {data.x.shape[1]})")
    print(f"  Edges      : {n_total:,}  (feature dim : {data.edge_attr.shape[1]})")
    print(f"  Fraud edges: {n_fraud:,} / {n_total:,}  ({n_fraud / n_total * 100:.2f}%)")

    return data


# ============================================================
# 7. EXECUTION
# ============================================================

print("\n" + "=" * 60)
print("Building dynamic account ID mapping...")
print("=" * 60)

account_to_idx, known_train_accounts = build_account_mapping(train_df, test_df)

# Luu mapping va known set de dung lai khi inference (tuan 5)
with open(MAPPING_PK, "wb") as f:
    pickle.dump(
        {
            "account_to_idx"      : account_to_idx,
            "known_train_accounts": known_train_accounts,
        },
        f,
    )
print(f"Account mapping saved -> {MAPPING_PK}")

# ---

print("\n" + "=" * 60)
print("Fitting One-Hot Encoder (train only)...")
print("=" * 60)

ohe = fit_ohe_encoder(train_df)

with open(ENCODER_PK, "wb") as f:
    pickle.dump(ohe, f)
print(f"OHE encoder saved -> {ENCODER_PK}")

# ---

print("\n" + "=" * 60)
print("Constructing PyTorch Geometric Datasets...")
print("=" * 60)

# TRAIN: structural stats & fraud history deu tu train_df
data_train = build_graph(
    df                   = train_df,
    structural_ref_df    = train_df,   # node features reflect train-time behavior
    fraud_ref_df         = train_df,   # khong co leakage
    acc_to_idx           = account_to_idx,
    known_train_accounts = known_train_accounts,
    ohe                  = ohe,
    split_name           = "train",
)

# TEST: van dung train_df lam nguon tham chieu cho node features
#   - structural_ref_df = train_df: dam bao distribution nhat quan voi luc train
#   - fraud_ref_df      = train_df: khong ro ri nhan test
data_test = build_graph(
    df                   = test_df,
    structural_ref_df    = train_df,   # nhat quan voi tap train
    fraud_ref_df         = train_df,   # khong co leakage
    acc_to_idx           = account_to_idx,
    known_train_accounts = known_train_accounts,
    ohe                  = ohe,
    split_name           = "test",
)

torch.save(data_train, TRAIN_PT)
torch.save(data_test,  TEST_PT)

print("\n" + "=" * 60)
print("Graph building successfully completed & ready for Production!")
print("=" * 60)
print(f"  data_train.pt -> {TRAIN_PT}")
print(f"  data_test.pt  -> {TEST_PT}")
print()
print("Summary:")
print(f"  Nodes (global)   : {data_train.num_nodes:,}")
print(f"  Node feature dim : {data_train.x.shape[1]}")
print(f"  Edge feature dim : {data_train.edge_attr.shape[1]}")
print(f"  Train edges      : {data_train.edge_index.shape[1]:,}")
print(f"  Test  edges      : {data_test.edge_index.shape[1]:,}")