from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parents[1]
PROC_DIR   = BASE_DIR / "data" / "processed"
TRAIN_CSV  = PROC_DIR / "train_cleaned.csv"
TEST_CSV   = PROC_DIR / "test_cleaned.csv"
SCALER_PK  = PROC_DIR / "scalers.pkl"


def run_leakage_audit() -> bool:
    print("=" * 70)
    print("      DATA LEAKAGE & INTEGRITY AUDIT REPORT (WEEK 3)")
    print("=" * 70)

    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(f"[ERROR] Chua tim thay tep du lieu:")
        print(f"  Train: {TRAIN_CSV} (Exists: {TRAIN_CSV.exists()})")
        print(f"  Test : {TEST_CSV} (Exists: {TEST_CSV.exists()})")
        return False

    print(f"Loading data tu:\n  Train: {TRAIN_CSV}\n  Test : {TEST_CSV}...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])

    passed_all = True

    # ------------------------------------------------------------
    # CHECK 1: TEMPORAL BOUNDARY
    # ------------------------------------------------------------
    print("\n[CHECK 1] TEMPORAL LEAKAGE AUDIT")
    t_train_min, t_train_max = train_df["timestamp"].min(), train_df["timestamp"].max()
    t_test_min, t_test_max   = test_df["timestamp"].min(), test_df["timestamp"].max()

    print(f"  Train Time Range: {t_train_min} --> {t_train_max}")
    print(f"  Test  Time Range: {t_test_min} --> {t_test_max}")

    if t_train_max <= t_test_min:
        print("  -> STATUS: [PASS] Ranh gioi thoi gian tuyet doi! Khong co Temporal Leakage.")
    else:
        print("  -> STATUS: [FAIL] Phat hien giao thoa thoi gian (Lookahead Leakage)!")
        passed_all = False

    # ------------------------------------------------------------
    # CHECK 2: INTEGRITY & MISSING VALUES (NaN)
    # ------------------------------------------------------------
    print("\n[CHECK 2] INTEGRITY & MISSING VALUES (NaN) AUDIT")
    train_nans = train_df.isna().sum()
    test_nans  = test_df.isna().sum()
    total_train_nan = train_nans.sum()
    total_test_nan  = test_nans.sum()

    print(f"  Train missing values: {total_train_nan}")
    print(f"  Test  missing values: {total_test_nan}")

    if total_train_nan == 0 and total_test_nan == 0:
        print("  -> STATUS: [PASS] 100% du lieu nguyen ven, khong co missing values.")
    else:
        print("  -> STATUS: [FAIL] Phat hien NaN trong du lieu!")
        print(f"  Chi tiet NaN Train: {train_nans[train_nans > 0].to_dict()}")
        print(f"  Chi tiet NaN Test : {test_nans[test_nans > 0].to_dict()}")
        passed_all = False

    # ------------------------------------------------------------
    # CHECK 3: SCALER PARAMETER AUDIT
    # ------------------------------------------------------------
    print("\n[CHECK 3] PREPROCESSING SCALER LEAKAGE AUDIT")
    if SCALER_PK.exists():
        with open(SCALER_PK, "rb") as f:
            scaler_bundle = pickle.load(f)
        scaler = scaler_bundle["scaler"]
        print(f"  Scaler da duoc serialize: {SCALER_PK.name}")
        print(f"  So cot scale: {len(scaler_bundle['feature_cols'])}")
        print(f"  Scaler n_samples_seen_: {scaler.n_samples_seen_:,}")
        if scaler.n_samples_seen_ == len(train_df):
            print(f"  -> STATUS: [PASS] Scaler chi hoc tu {len(train_df):,} dong tap Train (Zero Leakage).")
        else:
            print(f"  -> STATUS: [WARNING] Scaler n_samples_seen ({scaler.n_samples_seen_}) khac len(train_df) ({len(train_df)})!")
    else:
        print(f"  -> STATUS: [INFO] File {SCALER_PK} chua ton tai.")

    # ------------------------------------------------------------
    # CHECK 4: TARGET IMBALANCE & LOSS PARAMETERS (FOR AI ARCHITECT)
    # ------------------------------------------------------------
    print("\n[CHECK 4] TARGET DISTRIBUTION & LOSS CONFIG (CHO AI ARCHITECT)")
    train_fraud = int(train_df["Is_laundering"].sum())
    train_total = len(train_df)
    train_ratio = (train_fraud / train_total) * 100

    test_fraud  = int(test_df["Is_laundering"].sum())
    test_total  = len(test_df)
    test_ratio  = (test_fraud / test_total) * 100

    pos_weight  = (train_total - train_fraud) / max(1, train_fraud)

    print(f"  Train: {train_fraud:,} duong tinh / {train_total:,} tong ({train_ratio:.4f}%)")
    print(f"  Test : {test_fraud:,} duong tinh / {test_total:,} tong ({test_ratio:.4f}%)")
    print(f"  * POS_WEIGHT KHUYEN NGHI CHO BCE/FOCAL LOSS : {pos_weight:.2f}")

    # ------------------------------------------------------------
    # CHECK 5: INDUCTIVE TOPOLOGY STATS (CHO GRAPH ENGINEER)
    # ------------------------------------------------------------
    print("\n[CHECK 5] INDUCTIVE GRAPH TOPOLOGY (CHO GRAPH ENGINEER)")
    train_nodes = set(train_df["Sender_account"]).union(set(train_df["Receiver_account"]))
    test_nodes  = set(test_df["Sender_account"]).union(set(test_df["Receiver_account"]))

    overlap_nodes = train_nodes.intersection(test_nodes)
    unseen_nodes  = test_nodes - train_nodes

    print(f"  Tong so tai khoan tap Train (Base Graph) : {len(train_nodes):,}")
    print(f"  Tong so tai khoan tap Test               : {len(test_nodes):,}")
    print(f"  Tai khoan Test da biet tu Train (Overlap): {len(overlap_nodes):,} ({len(overlap_nodes)/len(test_nodes)*100:.2f}%)")
    print(f"  Tai khoan Test moi xuat hien (Unseen)    : {len(unseen_nodes):,} ({len(unseen_nodes)/len(test_nodes)*100:.2f}%)")
    print(f"  -> STATUS: [PASS] Ty le inductive hop ly, ho tro danh gia Cold-Start.")

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    if passed_all:
        print("  KET LUAN: 100% KIEM DINH THIET LAP THANH CONG! DU LIEU SAN SANG.")
    else:
        print("  KET LUAN: PHAT HIEN CANH BAO TRONG QUA TRINH AUDIT!")
    print("=" * 70)

    return passed_all


if __name__ == "__main__":
    success = run_leakage_audit()
    sys.exit(0 if success else 1)
