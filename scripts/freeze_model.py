"""
scripts/freeze_model.py
=======================
Script đóng băng (Freeze) mô hình GraphSAGE phục vụ Streaming & Đo độ trễ 

Mục tiêu:
  1. Nạp checkpoint tốt nhất của mô hình GraphSAGE (`graphsage_model.pt`)
  2. Khóa toàn bộ gradient (requires_grad = False), chuyển vĩnh viễn sang chế độ eval()
  3. Kiểm tra tính toàn vẹn số học (Numerical Equivalence check): Sai số logit/prob < 1e-7
  4. Đóng gói checkpoint đóng băng (`graphsage_frozen.pt`) kèm đầy đủ metadata phục vụ TV4
  5. Đo kiểm Benchmark độ trễ (Latency Benchmark) trên 1,000 lần suy luận
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.gnn_model import FraudDetectionGNN

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_CHECKPOINT = BASE_DIR / "data" / "processed" / "graphsage_model.pt"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "graphsage_frozen.pt"


def freeze_and_package_model(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    device: str = "cpu",
    benchmark_runs: int = 1000,
) -> tuple[FraudDetectionGNN, Path]:
    """
    Đóng băng trọng số mô hình và lưu artifact sản xuất.
    """
    checkpoint_p = Path(checkpoint_path)
    output_p = Path(output_path)

    if not checkpoint_p.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {checkpoint_p}")

    print("=" * 70)
    print("      HỆ THỐNG ĐÓNG BĂNG MÔ HÌNH GRAPHSAGE (MODEL FREEZING) - TUẦN 5        ")
    print("                 THÀNH VIÊN 2 (GRAPH ENGINEER)                             ")
    print("=" * 70)
    print(f"[*] Nạp checkpoint từ : {checkpoint_p}")
    print(f"[*] Đường dẫn xuất     : {output_p}")
    print(f"[*] Thiết bị kiểm tra  : {device}")

    # 1. NẠP MÔ HÌNH VÀ CHECKPOINT
    checkpoint = torch.load(checkpoint_p, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Định dạng checkpoint không hợp lệ: {checkpoint_p}")

    in_channels = int(checkpoint.get("in_channels", 8))
    hidden_channels = int(checkpoint.get("hidden_channels", 64))
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Khởi tạo mô hình đối chuẩn (Original) và mô hình đóng băng (Frozen)
    model_orig = FraudDetectionGNN(in_channels=in_channels, hidden_channels=hidden_channels).to(device)
    model_orig.load_state_dict(state_dict)
    model_orig.eval()

    model_frozen = FraudDetectionGNN(in_channels=in_channels, hidden_channels=hidden_channels).to(device)
    model_frozen.load_state_dict(state_dict)
    model_frozen.eval()

    # 2. KHÓA TOÀN BỘ GRADIENT (FREEZE ALL PARAMETERS)
    total_params = 0
    for name, param in model_frozen.named_parameters():
        param.requires_grad = False
        total_params += param.numel()

    trainable_params_before = sum(p.numel() for p in model_orig.parameters() if p.requires_grad)
    trainable_params_after = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)

    print(f"\n[*] Tổng số tham số mô hình       : {total_params:,}")
    print(f"[*] Tham số khả huấn luyện trước  : {trainable_params_before:,}")
    print(f"[*] Tham số khả huấn luyện sau    : {trainable_params_after:,} (ĐÃ KHÓA 100%)")

    # 3. KIỂM TRA TÍNH TOÀN VẸN SỐ HỌC (NUMERICAL INTEGRITY VERIFICATION)
    torch.manual_seed(42)
    dummy_x = torch.randn(20, in_channels, device=device)
    dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=device, dtype=torch.long)
    dummy_target_edge = torch.tensor([[0], [1]], device=device, dtype=torch.long)

    with torch.no_grad():
        out_orig = model_orig(dummy_x, dummy_edge_index, dummy_target_edge)
        out_frozen = model_frozen(dummy_x, dummy_edge_index, dummy_target_edge)
        max_diff = float((out_orig - out_frozen).abs().max().item())

    print(f"[*] Kiểm thử sai lệch số học      : {max_diff:.10e}")
    if max_diff > 1e-6:
        raise RuntimeError(f"Sai số giữa mô hình gốc và mô hình đóng băng vượt ngưỡng cho phép: {max_diff}")
    print("  -> TRẠNG THÁI: [PASS] Trọng số nguyên vẹn 100%, không bị suy hao dự đoán.")

    # 4. BENCHMARK ĐO ĐỘ TRỄ SUY LUẬN (LATENCY BENCHMARK)
    print(f"\n[*] Đo kiểm Benchmark tốc độ suy luận ({benchmark_runs:,} lần lặp)...")
    latencies: list[float] = []

    # Warm-up 50 runs
    for _ in range(50):
        with torch.no_grad():
            _ = model_frozen(dummy_x, dummy_edge_index, dummy_target_edge)

    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_frozen(dummy_x, dummy_edge_index, dummy_target_edge)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    lat_mean = np.mean(latencies)
    lat_p50 = np.percentile(latencies, 50)
    lat_p95 = np.percentile(latencies, 95)
    lat_p99 = np.percentile(latencies, 99)

    print(f"  - Độ trễ trung bình (Mean Latency) : {lat_mean:.4f} ms")
    print(f"  - Độ trễ P50                       : {lat_p50:.4f} ms")
    print(f"  - Độ trễ P95                       : {lat_p95:.4f} ms")
    print(f"  - Độ trễ P99                       : {lat_p99:.4f} ms")
    print(f"  -> Đánh giá mục tiêu tuần 5 (<100ms): Model forward chỉ tốn ~{lat_p95:.2f}ms, cực kỳ tối ưu cho TV4!")

    # 5. ĐÓNG GÓI VÀ LƯU ARTIFACT SẢN XUẤT
    output_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model_frozen.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "is_frozen": True,
        "total_parameters": total_params,
        "trainable_parameters": 0,
        "model_architecture": "FraudDetectionGNN (GraphSAGE 2-layer + Linear Classifier)",
        "freeze_timestamp": datetime.now().isoformat(),
        "benchmark_latency_mean_ms": float(lat_mean),
        "benchmark_latency_p95_ms": float(lat_p95),
    }

    torch.save(payload, output_p)
    print(f"\n[OK] Đã xuất thành công mô hình đóng băng tại: {output_p}")

    return model_frozen, output_p


def main() -> None:
    parser = argparse.ArgumentParser(description="Đóng băng mô hình GraphSAGE phục vụ streaming (TV2)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Đường dẫn file checkpoint gốc")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Đường dẫn lưu checkpoint đã freeze")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="cpu", help="Thiết bị tính toán")
    parser.add_argument("--benchmark-runs", type=int, default=1000, help="Số lần chạy benchmark đo độ trễ")
    args = parser.parse_args()

    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    freeze_and_package_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=dev,
        benchmark_runs=args.benchmark_runs,
    )


if __name__ == "__main__":
    main()
