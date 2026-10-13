"""
scripts/extract_fraud_subgraphs.py
===================================
Script thực thi trích xuất đồ thị con (Subgraphs) từ tập Test phục vụ Explainable AI.
(Tuần 4: Tích hợp GNNExplainer - Thành viên 1: Data Engineer).

Cách sử dụng mẫu:
  1. Trích xuất mặc định 100 giao dịch gian lận (isFraud = 1):
     python scripts/extract_fraud_subgraphs.py --limit 100

  2. Trích xuất toàn bộ giao dịch gian lận trong tập Test:
     python scripts/extract_fraud_subgraphs.py --limit all

  3. Trích xuất với mô hình GraphSAGE đã huấn luyện:
     python scripts/extract_fraud_subgraphs.py --checkpoint data/processed/graphsage_model.pt --mode model_pred --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import torch

# Đảm bảo import được các module từ src/
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.subgraph_extractor import FraudSubgraphExtractor


# Thiết lập mã hóa UTF-8 cho Windows console nếu cần
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script trích xuất đồ thị con (Subgraphs) xung quanh các giao dịch gian lận (isFraud=1) phục vụ XAI."
    )
    parser.add_argument(
        "--test-graph",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "data_test.pt",
        help="Đường dẫn file đồ thị Test (mặc định: data/processed/data_test.pt)",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "account_mapping.pkl",
        help="Đường dẫn file account_mapping.pkl để tra cứu ID tài khoản gốc",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "graphsage_model.pt",
        help="Đường dẫn checkpoint mô hình GraphSAGE (tùy chọn)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "subgraphs",
        help="Thư mục xuất kết quả đồ thị con (mặc định: data/processed/subgraphs)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ground_truth", "model_pred", "both"],
        default="ground_truth",
        help="Chế độ lọc giao dịch gian lận: 'ground_truth' (isFraud=1), 'model_pred', 'both'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng phân loại xác suất gian lận nếu dùng mô hình (mặc định: 0.5)",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=2,
        help="Bán kính k-hop láng giềng xung quanh giao dịch mục tiêu (mặc định: 2)",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default="100",
        help="Số lượng đồ thị con tối đa cần trích xuất (ví dụ '100', '50', hoặc 'all')",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Chỉ mở rộng láng giềng theo chiều dòng tiền một chiều (mặc định False: 2 chiều để bắt trọn vòng lặp AML)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Thiết bị tính toán ('cpu', 'cuda', hoặc 'auto')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 75)
    print("      HỆ THỐNG TRÍCH XUẤT ĐỒ THỊ CON GIAN LẬN (AML SUBGRAPH EXTRACTOR)     ")
    print("           TUẦN 4: TÍCH HỢP GNNEXPLAINER - THÀNH VIÊN 1 (DATA ENGINEER)    ")
    print("=" * 75)

    # 1. Kiểm tra thiết bị
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[*] Thiết bị tính toán : {device}")
    print(f"[*] Đường dẫn đồ thị   : {args.test_graph}")
    print(f"[*] Bán kính k-hop     : {args.num_hops}")
    print(f"[*] Chế độ lọc         : {args.mode}")

    # 2. Xử lý tham số limit
    limit_val: int | None = None
    if args.limit.lower() != "all":
        try:
            limit_val = int(args.limit)
        except ValueError:
            raise ValueError(f"Tham số --limit không hợp lệ: {args.limit}. Nhập số nguyên hoặc 'all'.")

    # 3. Khởi tạo Subgraph Extractor
    checkpoint_p = args.checkpoint if args.checkpoint.exists() else None
    extractor = FraudSubgraphExtractor(
        data_test=args.test_graph,
        mapping_path=args.mapping_path,
        checkpoint_path=checkpoint_p,
        device=device,
    )

    # 4. Lọc các giao dịch gian lận mục tiêu
    print(f"\n[*] Đang tìm kiếm các cạnh giao dịch bị đánh cờ gian lận (chế độ: {args.mode})...")
    target_edges = extractor.find_fraud_edges(
        mode=args.mode,
        threshold=args.threshold,
        limit=limit_val,
    )
    print(f"[+] Tìm thấy {len(target_edges):,} giao dịch gian lận mục tiêu cần trích xuất!")

    if len(target_edges) == 0:
        print("[!] Không tìm thấy giao dịch gian lận nào thoả mãn điều kiện! Dừng chương trình.")
        return

    # 5. Thực hiện trích xuất và lưu trữ hàng loạt
    manifest_df = extractor.batch_extract_and_save(
        target_edge_indices=target_edges,
        output_dir=args.output_dir,
        num_hops=args.num_hops,
        directed_expansion=args.directed,
    )

    # 6. Báo cáo thống kê cấu trúc đồ thị con phục vụ AML
    print("\n" + "=" * 75)
    print("              BÁO CÁO THỐNG KÊ CẤU TRÚC ĐỒ THỊ CON GIAN LẬN                 ")
    print("=" * 75)
    num_complex = int((manifest_df["num_fraud_edges"] > 1).sum())
    pct_complex = (num_complex / len(manifest_df)) * 100

    print(f"  - Tổng số đồ thị con đã xuất     : {len(manifest_df):,}")
    print(f"  - Số tài khoản trung bình (Nodes) : {manifest_df['num_nodes'].mean():.1f} (min: {manifest_df['num_nodes'].min()}, max: {manifest_df['num_nodes'].max()})")
    print(f"  - Số giao dịch trung bình (Edges) : {manifest_df['num_edges'].mean():.1f} (min: {manifest_df['num_edges'].min()}, max: {manifest_df['num_edges'].max()})")
    print(f"  - Số cạnh gian lận TB trong cụm   : {manifest_df['num_fraud_edges'].mean():.2f}")
    print(f"  - Cụm cấu trúc rửa tiền phức tạp   : {num_complex}/{len(manifest_df)} cụm ({pct_complex:.1f}%)")
    print(f"    (Có từ 2 giao dịch gian lận trở lên trong phạm vi {args.num_hops}-hop)")
    print("=" * 75)

    print("\n[*] GỢI Ý BÀN GIAO TIẾP THEO CHO CÁC THÀNH VIÊN TRONG NHÓM:")
    print("  -> Thành viên 3 (AI Architect)   : Đọc subgraphs/*.pt, nạp `sub_data.x`, `sub_data.edge_index`,")
    print("                                     và `sub_data.target_edge_label_index` vào GNNExplainer.")
    print("  -> Thành viên 2 (Graph Engineer) : Dùng `sub_data.orig_account_ids` và `sub_data.edge_index`")
    print("                                     để ánh xạ ngược và trực quan hóa qua NetworkX.")
    print("  -> Thành viên 4 (MLOps & Reporter): Tham chiếu `subgraphs_manifest.csv` để lập bảng Reason Codes.")
    print("\n[OK] Hoàn tất nhiệm vụ Tuần 4 của Thành viên 1 (Data Engineer)!\n")


if __name__ == "__main__":
    main()
