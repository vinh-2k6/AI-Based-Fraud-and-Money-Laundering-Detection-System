"""Mô phỏng luồng giao dịch thời gian thực từ tệp CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import cast

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_FILE_PATH = Path("data") / "processed" / "test_cleaned.csv"
MAX_VERBOSE_ROWS = 10


def parse_row(row: dict[str, str | None]) -> dict[str, object]:
    """Phân tích và ép kiểu các giá trị từ một dòng CSV."""
    parsed_row: dict[str, object] = {}

    for column, value in row.items():
        if value is None or value == "":
            parsed_row[column] = None
            continue

        try:
            parsed_row[column] = int(value)
            continue
        except ValueError:
            pass

        try:
            parsed_row[column] = float(value)
            continue
        except ValueError:
            parsed_row[column] = value

    return parsed_row


def infer_transaction(tx_json: str) -> None:
    """Hàm giữ chỗ cho việc tích hợp GraphSAGE và GNNExplainer sau này."""
    pass


def process_transaction(row: dict[str, str | None]) -> tuple[str, float]:
    """Phân tích, chuyển một dòng CSV thành JSON và trả về JSON cùng độ trễ."""
    start_time = time.perf_counter()
    parsed_row = parse_row(row)
    tx_json = json.dumps(parsed_row, ensure_ascii=False)
    infer_transaction(tx_json)
    end_time = time.perf_counter()
    latency = end_time - start_time
    return tx_json, latency


def _repository_root() -> Path:
    """Trả về thư mục gốc repository dựa trên vị trí của script."""
    return Path(__file__).resolve().parent.parent


def _resolve_file_path(file_path: str | Path) -> Path:
    """Phân giải đường dẫn đầu vào tương đối từ thư mục gốc repository."""
    path = Path(file_path)
    return path if path.is_absolute() else _repository_root() / path


def _validate_row(
    row: dict[str | None, str | None],
    fieldnames: list[str],
) -> dict[str, str | None]:
    """Từ chối các dòng có cấu trúc CSV không khớp với header."""
    if None in row:
        raise ValueError("Dòng chứa nhiều trường hơn số cột trong header")

    missing_columns = [
        column for column in fieldnames if row.get(column) is None
    ]
    if missing_columns:
        raise ValueError(
            "Dòng bị thiếu các trường: " + ", ".join(missing_columns)
        )

    return cast(dict[str, str | None], row)


def load_and_simulate_transactions(
    file_path: str | Path,
    limit: int | None = None,
    verbose: bool = False,
) -> tuple[list[float], int, int, float]:
    """Đọc tuần tự giao dịch từ CSV và thu thập số liệu xử lý."""
    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")

    latencies: list[float] = []
    total_rows = 0
    error_rows = 0
    start_time = time.perf_counter()
    resolved_path = _resolve_file_path(file_path)

    try:
        with resolved_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames

            if not fieldnames:
                raise ValueError("tệp CSV rỗng hoặc không có header")
            if any(not field.strip() for field in fieldnames):
                raise ValueError("header CSV chứa tên cột rỗng")
            if len(set(fieldnames)) != len(fieldnames):
                raise ValueError("header CSV chứa tên cột trùng lặp")

            for raw_row in reader:
                total_rows += 1
                try:
                    row = _validate_row(raw_row, fieldnames)
                    tx_json, latency = process_transaction(row)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    error_rows += 1
                    print(
                        f"Bỏ qua dòng {total_rows}: {exc}",
                        file=sys.stderr,
                    )
                    if limit is not None and total_rows >= limit:
                        break
                    continue

                latencies.append(latency)
                if verbose and len(latencies) <= MAX_VERBOSE_ROWS:
                    print(
                        f"Giao dịch {len(latencies)}: "
                        f"{tx_json} (độ trễ={latency * 1000:.3f} ms)"
                    )
                if limit is not None and total_rows >= limit:
                    break

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tệp CSV không tồn tại: {resolved_path}"
        ) from None
    except PermissionError:
        raise PermissionError(
            f"Không có quyền đọc tệp CSV: {resolved_path}"
        ) from None
    except csv.Error as exc:
        error_rows += 1
        print(f"Lỗi phân tích CSV: {exc}", file=sys.stderr)
    except (UnicodeDecodeError, OSError, ValueError):
        raise
    except KeyboardInterrupt:
        print(
            "\nĐã dừng mô phỏng; đang báo cáo các giao dịch đã xử lý.",
            file=sys.stderr,
        )

    total_time = time.perf_counter() - start_time
    return latencies, total_rows, error_rows, total_time


def print_summary(
    latencies: list[float],
    total_rows: int,
    error_rows: int,
    total_time: float,
) -> None:
    """In thống kê xử lý giao dịch và hiệu năng."""
    successful_rows = len(latencies)
    print("\nTóm tắt mô phỏng")
    print(f"Tổng số dòng đã đọc: {total_rows}")
    print(f"Số giao dịch thành công: {successful_rows}")
    print(f"Số dòng lỗi/bỏ qua: {error_rows}")
    print(f"Tổng thời gian chạy: {total_time * 1000:.3f} ms")

    if not latencies:
        print("Độ trễ trung bình: N/A")
        print("Độ trễ nhỏ nhất: N/A")
        print("Độ trễ lớn nhất: N/A")
        print("Độ lệch chuẩn: N/A")
        return

    latency_ms = [latency * 1000 for latency in latencies]
    standard_deviation = (
        statistics.stdev(latency_ms) if len(latency_ms) > 1 else 0.0
    )
    print(f"Độ trễ trung bình: {statistics.mean(latency_ms):.3f} ms")
    print(f"Độ trễ nhỏ nhất: {min(latency_ms):.3f} ms")
    print(f"Độ trễ lớn nhất: {max(latency_ms):.3f} ms")
    print(f"Độ lệch chuẩn: {standard_deviation:.3f} ms")


def _positive_int(value: str) -> int:
    """Phân tích giá trị số nguyên dương cho argparse."""
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit phải là một số nguyên") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("limit phải lớn hơn 0")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Đọc tuần tự giao dịch CSV qua pipeline kiểm thử thời gian thực."
    )
    parser.add_argument(
        "--file_path",
        default=str(DEFAULT_FILE_PATH),
        help="đường dẫn tệp CSV đầu vào (mặc định: data/test_cleaned.csv)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="số giao dịch tối đa cần xử lý",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="in chi tiết tối đa 10 giao dịch đầu tiên",
    )
    return parser


def main() -> None:
    """Phân tích tham số, chạy mô phỏng và in phần tóm tắt."""
    args = build_argument_parser().parse_args()
    try:
        metrics = load_and_simulate_transactions(
            file_path=args.file_path,
            limit=args.limit,
            verbose=args.verbose,
        )
    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return

    print_summary(*metrics)


if __name__ == "__main__":
    main()
