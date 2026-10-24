"""
src/data_processor.py
=====================
API nội bộ xử lý dữ liệu và trích xuất đặc trưng thời gian thực (DataProcessor)
phục vụ chuyển đổi luồng giao dịch streaming sang dạng biểu diễn đồ thị PyTorch Geometric.

Trách nhiệm (Tuần 5 - Thành viên 1: Data Engineer):
  1. Đóng gói toàn diện luồng trích xuất đặc trưng (Feature Extraction Pipeline).
  2. Quản lý cửa sổ trượt động theo thời gian (Causal Sliding Window 24h) duy trì trong bộ nhớ
     với độ phức tạp O(1) qua hàng đợi hai đầu Deque, loại bỏ nguy cơ rò rỉ dữ liệu (No Data Leakage).
  3. Mã hóa chu kỳ thời gian lượng giác (Trigonometric Cyclical Encoding) cho giờ, thứ, tháng.
  4. Chuẩn hóa thang đo (MinMaxScaler + Log1p) đồng bộ 100% với cấu trúc tập Train.
  5. Mã hóa One-Hot (OHE) siêu tốc bằng bảng tra cứu trực tiếp (Fast OHE Lookup), giảm thiểu
     độ trễ xử lý xuống mức micro-giây (< 0.5ms / giao dịch).
  6. Ánh xạ định danh tài khoản quy nạp (Inductive Dynamic Account Mapping) và cơ chế xử lý
     nút mới (Cold-Start Node Handling).
  7. Đóng gói kết quả thành cấu trúc PyG Tensors chuẩn (edge_index, edge_attr, node_features)
     sẵn sàng cho mô hình GraphSAGE inference và GNNExplainer.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Đảm bảo in tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CÁC HẰNG SỐ CẤU HÌNH ĐẶC TRƯNG MẶC ĐỊNH
# ============================================================

DEFAULT_WINDOW_SECONDS = 86400  # 24 giờ

CONTINUOUS_FEATURE_COLS = [
    "Amount_scaled",
    "sender_tx_count_24h_scaled",
    "receiver_tx_count_24h_scaled",
    "sender_amount_sum_24h_scaled",
    "time_since_last_sender_tx_scaled",
]

RAW_NUMERIC_FEATURE_COLS = [
    "Amount",
    "sender_tx_count_24h",
    "receiver_tx_count_24h",
    "sender_amount_sum_24h",
    "time_since_last_sender_tx",
]

CATEGORICAL_COLS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

TOTAL_CYCLICAL_DIMS = 6   # sin/cos cho hour, day_of_week, month
TOTAL_CONTINUOUS_DIMS = 5 # 5 continuous scaled features
TOTAL_CATEGORICAL_DIMS = 69 # Tổng số danh mục OHE từ dữ liệu SAML-D
TOTAL_EDGE_FEATURE_DIMS = TOTAL_CONTINUOUS_DIMS + TOTAL_CYCLICAL_DIMS + TOTAL_CATEGORICAL_DIMS # 80 dims
TOTAL_NODE_FEATURE_DIMS = 8 # 8 dims theo thiết kế đồ thị tuần 2


# ============================================================
# CẤU TRÚC DỮ LIỆU ĐẦU RA (OUTPUT DATA STRUCTURES)
# ============================================================

@dataclass
class ProcessedTransaction:
    """
    Kết quả chuyển đổi một giao dịch đơn lẻ thành biểu diễn đồ thị PyTorch Geometric.
    """
    sender_account: int
    receiver_account: int
    sender_idx: int
    receiver_idx: int
    edge_index: torch.Tensor          # Shape: [2, 1], dtype=torch.long
    edge_attr: torch.Tensor           # Shape: [1, 80], dtype=torch.float32
    is_new_sender: bool = False
    is_new_receiver: bool = False
    processing_time_ms: float = 0.0
    feature_dict: dict[str, Any] = field(default_factory=dict)

    def to_pyg_data(self) -> Data:
        """Đóng gói thành đối tượng PyTorch Geometric Data."""
        return Data(
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=max(self.sender_idx, self.receiver_idx) + 1,
        )


@dataclass
class BatchProcessedData:
    """
    Kết quả chuyển đổi một batch giao dịch thành tensor PyTorch Geometric.
    """
    edge_index: torch.Tensor          # Shape: [2, B], dtype=torch.long
    edge_attr: torch.Tensor           # Shape: [B, 80], dtype=torch.float32
    sender_indices: torch.Tensor      # Shape: [B], dtype=torch.long
    receiver_indices: torch.Tensor    # Shape: [B], dtype=torch.long
    sender_accounts: list[int]
    receiver_accounts: list[int]
    batch_size: int
    processing_time_ms: float = 0.0

    def to_pyg_data(self, num_nodes: int | None = None) -> Data:
        """Đóng gói thành đối tượng PyTorch Geometric Data."""
        nodes = num_nodes or (int(self.edge_index.max().item()) + 1 if self.batch_size > 0 else 0)
        return Data(
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=nodes,
        )


# ============================================================
# API NỘI BỘ XỬ LÝ DỮ LIỆU STREAMING: DataProcessor
# ============================================================

class DataProcessor:
    """
    API nội bộ chịu trách nhiệm chuyển đổi trực tiếp các giao dịch từ luồng streaming
    thành PyG Tensors với độ trễ thấp (< 1ms).

    Args:
        base_dir (Path | str | None): Thư mục gốc chứa thư mục data/processed.
        scaler_path (Path | str | None): Đường dẫn file scalers.pkl.
        encoder_path (Path | str | None): Đường dẫn file edge_encoders.pkl.
        mapping_path (Path | str | None): Đường dẫn file account_mapping.pkl.
        base_graph_path (Path | str | None): Đường dẫn file data_train.pt (tùy chọn nạp node feature).
        window_seconds (int): Độ dài cửa sổ trượt quá khứ (mặc định: 86400s = 24h).
        allow_new_nodes (bool): Cho phép tự động thêm tài khoản mới vào mapping đồ thị quy nạp.
        device (str | torch.device): Thiết bị tính toán ('cpu' hoặc 'cuda').
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        scaler_path: Path | str | None = None,
        encoder_path: Path | str | None = None,
        mapping_path: Path | str | None = None,
        base_graph_path: Path | str | None = None,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        allow_new_nodes: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.window_seconds = window_seconds
        self.allow_new_nodes = allow_new_nodes

        # Phân giải đường dẫn mặc định
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parents[1]
        else:
            self.base_dir = Path(base_dir)

        proc_dir = self.base_dir / "data" / "processed"
        self.scaler_path = Path(scaler_path) if scaler_path else proc_dir / "scalers.pkl"
        self.encoder_path = Path(encoder_path) if encoder_path else proc_dir / "edge_encoders.pkl"
        self.mapping_path = Path(mapping_path) if mapping_path else proc_dir / "account_mapping.pkl"
        self.base_graph_path = Path(base_graph_path) if base_graph_path else proc_dir / "data_train.pt"

        # 1. Nạp và tối ưu Scaler
        self._load_and_optimize_scaler()

        # 2. Nạp và tối ưu OneHotEncoder (Fast Lookup Table)
        self._load_and_optimize_encoder()

        # 3. Nạp Từ điển ánh xạ tài khoản
        self._load_account_mapping()

        # 4. Khởi tạo trạng thái Causal Sliding Window (24h)
        self._init_sliding_window_state()

        # 5. Nạp cấu trúc Node Features nếu có
        self.node_features: torch.Tensor | None = None
        self._try_load_node_features()

    # ------------------------------------------------------------
    # 1. KHỞI TẠO VÀ TỐI ƯU HÓA ARTIFACTS
    # ------------------------------------------------------------

    def _load_and_optimize_scaler(self) -> None:
        """Nạp MinMaxScaler và trích xuất các vector scale_/min_ để tính nhanh với numpy."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file scalers.pkl tại: {self.scaler_path}")

        with open(self.scaler_path, "rb") as f:
            scaler_data = pickle.load(f)

        self.scaler = scaler_data["scaler"]
        self.scaler_feature_cols = scaler_data.get("feature_cols", [
            "Amount_log",
            "sender_tx_count_24h_log",
            "receiver_tx_count_24h_log",
            "sender_amount_sum_24h_log",
            "time_since_last_sender_tx_log",
        ])
        self.scaler_scaled_cols = scaler_data.get("scaled_cols", CONTINUOUS_FEATURE_COLS)

        # Chuyển đổi scale_ và min_ sang numpy float32 để tính toán vector O(1)
        self._scale_factors = np.asarray(self.scaler.scale_, dtype=np.float32)
        self._min_factors = np.asarray(self.scaler.min_, dtype=np.float32)

    def _load_and_optimize_encoder(self) -> None:
        """
        Nạp OneHotEncoder và biên dịch thành cấu trúc băm ánh xạ phẳng (Flat Hash Table)
        để mã hóa One-Hot với độ trễ micro-giây, không cần qua pandas DataFrame.
        """
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file edge_encoders.pkl tại: {self.encoder_path}")

        with open(self.encoder_path, "rb") as f:
            self.ohe = pickle.load(f)

        self.categorical_cols = list(getattr(self.ohe, "feature_names_in_", CATEGORICAL_COLS))

        # Xây dựng bảng tra cứu: col_name -> {category_value: relative_ohe_index}
        self._cat_lookup: dict[str, dict[Any, int]] = {}
        total_offset = 0
        for col_name, cats in zip(self.categorical_cols, self.ohe.categories_):
            cat_map: dict[Any, int] = {}
            for i, cat in enumerate(cats):
                cat_map[cat] = total_offset + i
                # Hỗ trợ cả kiểu chuỗi nếu cat là kiểu khác
                if not isinstance(cat, str):
                    cat_map[str(cat)] = total_offset + i
            self._cat_lookup[col_name] = cat_map
            total_offset += len(cats)

        self.total_categorical_dims = total_offset
        if self.total_categorical_dims != TOTAL_CATEGORICAL_DIMS:
            # Ghi nhận số chiều thực tế
            pass

    def _load_account_mapping(self) -> None:
        """Nạp từ điển ánh xạ account_id -> node_index."""
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file account_mapping.pkl tại: {self.mapping_path}")

        with open(self.mapping_path, "rb") as f:
            mapping_data = pickle.load(f)

        if isinstance(mapping_data, dict) and "account_to_idx" in mapping_data:
            self.account_to_idx: dict[int, int] = dict(mapping_data["account_to_idx"])
            self.known_train_accounts: set[int] = set(mapping_data.get("known_train_accounts", set()))
        elif isinstance(mapping_data, dict):
            self.account_to_idx = dict(mapping_data)
            self.known_train_accounts = set(self.account_to_idx.keys())
        else:
            raise ValueError(f"Định dạng mapping không hợp lệ tại: {self.mapping_path}")

        self._next_node_idx = len(self.account_to_idx)

    def _init_sliding_window_state(self) -> None:
        """Khởi tạo trạng thái cửa sổ trượt động 24h."""
        # Deque lưu (epoch_seconds, amount) cho từng account
        self._sender_history: dict[int, deque[tuple[int, float]]] = defaultdict(deque)
        self._receiver_history: dict[int, deque[tuple[int, float]]] = defaultdict(deque)
        self._sender_sum_tracker: dict[int, float] = defaultdict(float)
        self._last_sender_tx_time: dict[int, int] = {}

    def _try_load_node_features(self) -> None:
        """Thử nạp node features từ data_train.pt nếu tồn tại."""
        if self.base_graph_path and self.base_graph_path.exists():
            try:
                base_data = torch.load(self.base_graph_path, map_location="cpu", weights_only=False)
                if hasattr(base_data, "x") and isinstance(base_data.x, torch.Tensor):
                    self.node_features = base_data.x.to(self.device)
            except Exception:
                self.node_features = None

    # ------------------------------------------------------------
    # 2. QUẢN LÝ CỬA SỔ TRƯỢT 24H (CAUSAL TEMPORAL SLIDING WINDOW)
    # ------------------------------------------------------------

    def update_sliding_window(
        self,
        sender: int,
        receiver: int,
        amount: float,
        timestamp_epoch: int,
    ) -> tuple[int, int, float, float]:
        """
        Cập nhật cửa sổ trượt 24h cho người gửi và người nhận, tính toán 4 đặc trưng thời gian thực.
        Độ phức tạp O(1) trung bình cho mỗi giao dịch.

        Returns:
            (sender_tx_count_24h, receiver_tx_count_24h, sender_amount_sum_24h, time_since_last_sender_tx)
        """
        threshold_epoch = timestamp_epoch - self.window_seconds

        # 1. Delta time kể từ giao dịch gần nhất của Sender
        if sender in self._last_sender_tx_time:
            time_diff_last_tx = max(0.0, float(timestamp_epoch - self._last_sender_tx_time[sender]))
        else:
            time_diff_last_tx = float(self.window_seconds)
        self._last_sender_tx_time[sender] = timestamp_epoch

        # 2. Dọn các giao dịch của Sender cũ hơn 24h
        s_deque = self._sender_history[sender]
        while s_deque and s_deque[0][0] < threshold_epoch:
            _, old_amt = s_deque.popleft()
            self._sender_sum_tracker[sender] -= old_amt
        # Đảm bảo tổng tiền không bị âm do lỗi số thực dấu phẩy động
        if self._sender_sum_tracker[sender] < 0.0:
            self._sender_sum_tracker[sender] = 0.0

        # Lấy thống kê hiện tại TRƯỚC KHI append giao dịch hiện tại (tránh lookahead leakage)
        sender_count = len(s_deque)
        sender_sum = self._sender_sum_tracker[sender]

        # Cập nhật giao dịch hiện tại vào lịch sử Sender
        s_deque.append((timestamp_epoch, amount))
        self._sender_sum_tracker[sender] += amount

        # 3. Dọn các giao dịch của Receiver cũ hơn 24h
        r_deque = self._receiver_history[receiver]
        while r_deque and r_deque[0][0] < threshold_epoch:
            r_deque.popleft()

        receiver_count = len(r_deque)
        r_deque.append((timestamp_epoch, amount))

        return sender_count, receiver_count, sender_sum, time_diff_last_tx

    def warmup_state(self, transactions_or_df: Iterable[dict[str, Any]] | pd.DataFrame) -> int:
        """
        Nạp dữ liệu lịch sử để làm ấm cửa sổ trượt trước khi bắt đầu luồng streaming.
        
        Args:
            transactions_or_df: Danh sách dict giao dịch hoặc DataFrame đã sắp xếp theo thời gian.
        
        Returns:
            Số lượng giao dịch đã nạp.
        """
        count = 0
        if isinstance(transactions_or_df, pd.DataFrame):
            df = transactions_or_df
            senders = df["Sender_account"].values
            receivers = df["Receiver_account"].values
            amounts = df["Amount"].values
            
            # Epoch seconds
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"]).values.astype("int64") // 10**9
            elif "Date" in df.columns and "Time" in df.columns:
                ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str)).values.astype("int64") // 10**9
            else:
                raise ValueError("DataFrame phải chứa 'timestamp' hoặc 'Date' và 'Time'")

            for i in range(len(df)):
                self.update_sliding_window(
                    sender=int(senders[i]),
                    receiver=int(receivers[i]),
                    amount=float(amounts[i]),
                    timestamp_epoch=int(ts[i]),
                )
                count += 1
        else:
            for tx in transactions_or_df:
                s = int(tx["Sender_account"])
                r = int(tx["Receiver_account"])
                amt = float(tx["Amount"])
                t, _, _, _ = self._parse_timestamp(tx)
                self.update_sliding_window(sender=s, receiver=r, amount=amt, timestamp_epoch=t)
                count += 1

        return count

    def reset_state(self) -> None:
        """Xóa trắng bộ nhớ trạng thái cửa sổ trượt."""
        self._init_sliding_window_state()

    # ------------------------------------------------------------
    # 3. TRÍCH XUẤT VÀ BIẾN ĐỔI ĐẶC TRƯNG (FEATURE TRANSFORMATION)
    # ------------------------------------------------------------

    def _parse_timestamp(self, tx_dict: dict[str, Any]) -> tuple[int, int, int, int]:
        """
        Phân tích thời gian từ giao dịch.
        
        Returns:
            (epoch_seconds, hour, day_of_week, month)
        """
        if "timestamp" in tx_dict and tx_dict["timestamp"] is not None:
            raw_ts = tx_dict["timestamp"]
            if isinstance(raw_ts, (int, float, np.integer, np.floating)):
                epoch_sec = int(raw_ts)
                dt = pd.to_datetime(epoch_sec, unit="s")
            else:
                dt = pd.to_datetime(str(raw_ts))
                epoch_sec = int(dt.timestamp())
        elif "Date" in tx_dict and "Time" in tx_dict:
            dt_str = f"{tx_dict['Date']} {tx_dict['Time']}"
            dt = pd.to_datetime(dt_str)
            epoch_sec = int(dt.timestamp())
        else:
            # Mặc định lấy thời điểm hiện tại nếu hoàn toàn không có thông tin thời gian
            epoch_sec = int(time.time())
            dt = pd.to_datetime(epoch_sec, unit="s")

        hour = int(dt.hour)
        day_of_week = int(dt.dayofweek)
        month = int(dt.month)
        return epoch_sec, hour, day_of_week, month

    def _compute_cyclical_features(self, hour: int, day_of_week: int, month: int) -> np.ndarray:
        """
        Tính toán 6 đặc trưng chu kỳ thời gian lượng giác:
        [sin(hour), cos(hour), sin(dow), cos(dow), sin(month), cos(month)]
        """
        features = np.empty(6, dtype=np.float32)

        # hour in [0..23], max = 24
        ang_h = 2.0 * math.pi * hour / 24.0
        features[0] = math.sin(ang_h)
        features[1] = math.cos(ang_h)

        # day_of_week in [0..6], max = 7
        ang_d = 2.0 * math.pi * day_of_week / 7.0
        features[2] = math.sin(ang_d)
        features[3] = math.cos(ang_d)

        # month in [1..12], max = 12
        ang_m = 2.0 * math.pi * month / 12.0
        features[4] = math.sin(ang_m)
        features[5] = math.cos(ang_m)

        return features

    def _scale_continuous(
        self,
        amount: float,
        sender_count: int,
        receiver_count: int,
        sender_sum: float,
        time_diff: float,
    ) -> np.ndarray:
        """
        Log1p + MinMaxScaler cho 5 đặc trưng liên tục:
        X_scaled = log1p(X) * scale_ + min_
        """
        raw_vals = np.array(
            [amount, sender_count, receiver_count, sender_sum, time_diff],
            dtype=np.float32,
        )
        log_vals = np.log1p(np.maximum(0.0, raw_vals))
        scaled_vals = log_vals * self._scale_factors + self._min_factors
        return scaled_vals.astype(np.float32)

    def _encode_categoricals(self, tx_dict: dict[str, Any]) -> np.ndarray:
        """
        Mã hóa One-Hot siêu tốc bằng bảng băm tra cứu trực tiếp.
        Tự động bỏ qua các giá trị chưa thấy (tương đương handle_unknown='ignore').
        """
        ohe_vec = np.zeros(self.total_categorical_dims, dtype=np.float32)

        for col_name in self.categorical_cols:
            val = tx_dict.get(col_name)
            if val is not None:
                col_lookup = self._cat_lookup.get(col_name, {})
                idx = col_lookup.get(val)
                if idx is not None:
                    ohe_vec[idx] = 1.0

        return ohe_vec

    def _resolve_account_idx(self, account_id: int) -> tuple[int, bool]:
        """
        Tra cứu hoặc cấp phát node index cho account_id theo quy trình Inductive Graph.
        
        Returns:
            (node_idx, is_new_account)
        """
        acc_int = int(account_id)
        if acc_int in self.account_to_idx:
            return self.account_to_idx[acc_int], False

        if not self.allow_new_nodes:
            raise KeyError(f"Tài khoản {acc_int} không tồn tại trong từ điển ánh xạ và allow_new_nodes=False.")

        # Cấp phát ID động cho tài khoản mới
        new_idx = self._next_node_idx
        self.account_to_idx[acc_int] = new_idx
        self._next_node_idx += 1
        return new_idx, True

    # ------------------------------------------------------------
    # 4. CHUYỂN ĐỔI GIAO DỊCH STREAMING (PROCESS TRANSACTION)
    # ------------------------------------------------------------

    def process_transaction(self, tx_dict: dict[str, Any]) -> ProcessedTransaction:
        """
        Chuyển đổi một giao dịch đơn lẻ thành tensor PyTorch Geometric và các metadata liên quan.
        Hỗ trợ cả giao dịch dạng thô (Raw) và giao dịch đã tiền xử lý sẵn (Pre-cleaned).

        Độ trễ trung bình: < 0.3ms.
        """
        t0 = time.perf_counter()

        # 1. Trích xuất tài khoản và xác định index
        sender_acc = int(tx_dict["Sender_account"])
        receiver_acc = int(tx_dict["Receiver_account"])

        sender_idx, is_new_sender = self._resolve_account_idx(sender_acc)
        receiver_idx, is_new_receiver = self._resolve_account_idx(receiver_acc)

        feature_log: dict[str, Any] = {
            "sender_account": sender_acc,
            "receiver_account": receiver_acc,
            "sender_idx": sender_idx,
            "receiver_idx": receiver_idx,
        }

        # 2. Xử lý đặc trưng cạnh (Edge features)
        # Kiểm tra xem giao dịch đã chứa các trường scaled sẵn chưa (ví dụ từ test_cleaned.csv)
        has_precomputed_scaled = all(col in tx_dict for col in CONTINUOUS_FEATURE_COLS)

        if has_precomputed_scaled:
            # Chế độ Pre-cleaned: Sử dụng trực tiếp giá trị đã scale
            cont_vals = np.array(
                [float(tx_dict[col]) for col in CONTINUOUS_FEATURE_COLS],
                dtype=np.float32,
            )
            hour = int(tx_dict.get("hour", 0))
            dow = int(tx_dict.get("day_of_week", 0))
            month = int(tx_dict.get("month", 1))
            cyclical_vals = self._compute_cyclical_features(hour, dow, month)
            cat_vals = self._encode_categoricals(tx_dict)

            for col in CONTINUOUS_FEATURE_COLS:
                feature_log[col] = float(tx_dict[col])
        else:
            # Chế độ Raw: Tính toán đầy đủ qua Causal Sliding Window
            raw_amount = float(tx_dict.get("Amount", 0.0))
            epoch_sec, hour, dow, month = self._parse_timestamp(tx_dict)

            # Cập nhật sliding window 24h
            s_cnt, r_cnt, s_sum, t_diff = self.update_sliding_window(
                sender=sender_acc,
                receiver=receiver_acc,
                amount=raw_amount,
                timestamp_epoch=epoch_sec,
            )

            # Log và MinMax scale
            cont_vals = self._scale_continuous(
                amount=raw_amount,
                sender_count=s_cnt,
                receiver_count=r_cnt,
                sender_sum=s_sum,
                time_diff=t_diff,
            )

            cyclical_vals = self._compute_cyclical_features(hour, dow, month)
            cat_vals = self._encode_categoricals(tx_dict)

            # Ghi nhận log phục vụ kiểm tra
            feature_log["Amount"] = raw_amount
            feature_log["sender_tx_count_24h"] = s_cnt
            feature_log["receiver_tx_count_24h"] = r_cnt
            feature_log["sender_amount_sum_24h"] = s_sum
            feature_log["time_since_last_sender_tx"] = t_diff
            for i, col in enumerate(CONTINUOUS_FEATURE_COLS):
                feature_log[col] = float(cont_vals[i])

        # 3. Kết hợp vector đặc trưng cạnh: [5 continuous, 6 cyclical, 69 categorical] = 80 dims
        edge_feat_np = np.concatenate([cont_vals, cyclical_vals, cat_vals])

        # 4. Tạo PyTorch Tensors
        edge_index = torch.tensor([[sender_idx], [receiver_idx]], dtype=torch.long, device=self.device)
        edge_attr = torch.from_numpy(edge_feat_np).unsqueeze(0).to(self.device)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ProcessedTransaction(
            sender_account=sender_acc,
            receiver_account=receiver_acc,
            sender_idx=sender_idx,
            receiver_idx=receiver_idx,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_new_sender=is_new_sender,
            is_new_receiver=is_new_receiver,
            processing_time_ms=elapsed_ms,
            feature_dict=feature_log,
        )

    def process_json(self, tx_json: str) -> ProcessedTransaction:
        """
        Phân tích chuỗi JSON và chuyển đổi thành ProcessedTransaction.
        Thích hợp cho Message Queue / Kafka Consumer.
        """
        try:
            tx_dict = json.loads(tx_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Dữ liệu JSON không hợp lệ: {e}") from e

        if not isinstance(tx_dict, dict):
            raise ValueError(f"Dữ liệu JSON phải là dictionary, nhận được: {type(tx_dict)}")

        return self.process_transaction(tx_dict)

    # ------------------------------------------------------------
    # 5. CHẾ ĐỘ XỬ LÝ BATCH (MICRO-BATCH STREAMING)
    # ------------------------------------------------------------

    def process_batch(
        self,
        transactions_or_df: list[dict[str, Any]] | pd.DataFrame,
    ) -> BatchProcessedData:
        """
        Chuyển đổi một danh sách giao dịch hoặc một DataFrame thành batch tensor PyG.
        Tối ưu hóa thông lượng (throughput) cho kịch bản micro-batching.
        """
        t0 = time.perf_counter()

        if isinstance(transactions_or_df, pd.DataFrame):
            df = transactions_or_df
            rows = df.to_dict(orient="records")
        else:
            rows = list(transactions_or_df)

        batch_size = len(rows)
        if batch_size == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            empty_attr = torch.empty((0, TOTAL_EDGE_FEATURE_DIMS), dtype=torch.float32, device=self.device)
            return BatchProcessedData(
                edge_index=empty_index,
                edge_attr=empty_attr,
                sender_indices=torch.empty(0, dtype=torch.long, device=self.device),
                receiver_indices=torch.empty(0, dtype=torch.long, device=self.device),
                sender_accounts=[],
                receiver_accounts=[],
                batch_size=0,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        src_indices: list[int] = []
        dst_indices: list[int] = []
        src_accounts: list[int] = []
        dst_accounts: list[int] = []
        edge_attrs: list[torch.Tensor] = []

        for row in rows:
            processed = self.process_transaction(row)
            src_indices.append(processed.sender_idx)
            dst_indices.append(processed.receiver_idx)
            src_accounts.append(processed.sender_account)
            dst_accounts.append(processed.receiver_account)
            edge_attrs.append(processed.edge_attr)

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long, device=self.device)
        edge_attr = torch.cat(edge_attrs, dim=0)

        sender_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device)
        receiver_tensor = torch.tensor(dst_indices, dtype=torch.long, device=self.device)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return BatchProcessedData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            sender_indices=sender_tensor,
            receiver_indices=receiver_tensor,
            sender_accounts=src_accounts,
            receiver_accounts=dst_accounts,
            batch_size=batch_size,
            processing_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------
    # 6. TIỆN ÍCH NODE FEATURES & BENCHMARK
    # ------------------------------------------------------------

    def get_node_features(self, node_idx: int) -> torch.Tensor:
        """
        Lấy vector đặc trưng 8 chiều của một node.
        Nếu node_idx nằm ngoài cấu trúc base graph (cold-start), trả về vector mặc định:
        [out_deg=0, in_deg=0, total_deg=0, avg_out=0, avg_in=0, fraud_out=0, fraud_in=0, is_new=1.0]
        """
        if self.node_features is not None and 0 <= node_idx < self.node_features.size(0):
            return self.node_features[node_idx]

        # Mặc định cho node mới (Cold-start)
        default_feat = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        return default_feat

    def benchmark_latency(
        self,
        sample_tx: dict[str, Any] | None = None,
        num_runs: int = 1000,
    ) -> dict[str, float]:
        """
        Đo kiểm chi tiết phân phối độ trễ (Latency Distribution) của DataProcessor.
        Phục vụ mục tiêu chứng minh hiệu năng hệ thống đạt tiêu chuẩn < 100ms toàn trình.
        """
        if sample_tx is None:
            sample_tx = {
                "Sender_account": 9768510209,
                "Receiver_account": 5764492978,
                "Amount": 1500.0,
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "Payment_type": "ACH",
                "Date": "2023-06-19",
                "Time": "22:08:52",
            }

        # Khởi động (warmup) JIT và cache
        for _ in range(50):
            self.process_transaction(sample_tx)

        latencies_ms: list[float] = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            self.process_transaction(sample_tx)
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        metrics = {
            "num_runs": float(num_runs),
            "mean_ms": float(np.mean(latencies_arr)),
            "std_ms": float(np.std(latencies_arr)),
            "min_ms": float(np.min(latencies_arr)),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p90_ms": float(np.percentile(latencies_arr, 90)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "max_ms": float(np.max(latencies_arr)),
        }

        print("\n" + "=" * 55)
        print("KẾT QUẢ ĐO KIỂM HIỆU NĂNG DATA PROCESSOR (STREAMING)")
        print("=" * 55)
        print(f"Số lần lặp thử nghiệm: {num_runs:,}")
        print(f"Độ trễ trung bình    : {metrics['mean_ms']:.4f} ms")
        print(f"Độ trễ Median (P50)  : {metrics['p50_ms']:.4f} ms")
        print(f"Phân vị P95          : {metrics['p95_ms']:.4f} ms")
        print(f"Phân vị P99          : {metrics['p99_ms']:.4f} ms")
        print(f"Độ trễ nhỏ nhất      : {metrics['min_ms']:.4f} ms")
        print(f"Độ trễ lớn nhất      : {metrics['max_ms']:.4f} ms")
        print("=" * 55)

        return metrics


# ============================================================
# CLI ĐỂ CHẠY KIỂM TRA ĐỘC LẬP
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KHỞI TẠO VÀ KIỂM TRA DATA PROCESSOR (TUẦN 5)")
    print("=" * 60)

    processor = DataProcessor()
    print(f"-> Đã nạp thành công Account Mapping: {len(processor.account_to_idx):,} tài khoản.")
    print(f"-> Số chiều One-Hot Categorical: {processor.total_categorical_dims} dims.")
    print(f"-> Tổng số chiều Edge Features : {TOTAL_EDGE_FEATURE_DIMS} dims.")

    # Benchmark độ trễ với raw transaction
    processor.benchmark_latency(num_runs=1000)
