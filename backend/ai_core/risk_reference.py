"""Trình bày risk score theo bằng chứng: percentile so với tập train, nhóm theo tam phân vị train,
đường sống còn chỉ khi đã đạt tiêu chí hiệu chuẩn trên tập test (file do research/prognosis/evaluate.py sinh)."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from typing import Any

CURVE_TIMES = (0, 6, 12, 18, 24, 30, 36)
GROUPS = ("Low", "Medium", "High")


def missing_reference_meta() -> dict[str, Any]:
    """Meta khi chưa có risk_reference.json: không có percentile, nhóm hay đường sống còn."""
    return {
        "calibrated": False, "percentile": None, "group": None,
        "c_index_test": None, "c_index_ci": None, "n_train": None, "n_test": None,
        "limitation_text": "Chưa có dữ liệu tham chiếu của tập huấn luyện – điểm nguy cơ chỉ để tham khảo, bác sĩ quyết định.",
    }


@dataclass(frozen=True)
class RiskReference:
    """Tham chiếu risk của tập train và kết quả kiểm chứng trên tập test."""

    train_risk_sorted: list[float]
    tertile_cuts: list[float]
    s0_t: list[float]
    s0_s: list[float]
    calibrated: bool
    test_metrics: dict[str, Any]
    n_train: int
    n_test: int

    @classmethod
    def load(cls, path: str) -> "RiskReference":
        """Đọc risk_reference.json do evaluate.py ghi."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            train_risk_sorted=sorted(data["train_risk_sorted"]), tertile_cuts=list(data["tertile_cuts"]),
            s0_t=list(data["s0"]["t"]), s0_s=list(data["s0"]["s"]), calibrated=bool(data["calibrated"]),
            test_metrics=dict(data["test_metrics"]), n_train=int(data["n_train"]), n_test=int(data["n_test"]),
        )

    def percentile(self, score: float) -> float:
        """Phần trăm bệnh nhân train có risk ≤ score, trong [0, 100]."""
        rank = bisect.bisect_right(self.train_risk_sorted, score)
        return round(100.0 * rank / len(self.train_risk_sorted), 1)

    def group(self, score: float) -> str:
        """Low / Medium / High theo tam phân vị risk của tập train."""
        return GROUPS[bisect.bisect_right(self.tertile_cuts, score)]

    def survival_curve(self, score: float) -> list[dict[str, float]] | None:
        """S(t) = S0(t) ** exp(score) tại 0–36 tháng; None nếu chưa đạt tiêu chí hiệu chuẩn."""
        if not self.calibrated:
            return None
        hazard_ratio = math.exp(max(-20.0, min(20.0, score)))
        curve = []
        for t in CURVE_TIMES:
            position = bisect.bisect_right(self.s0_t, t) - 1
            s0 = 1.0 if position < 0 else self.s0_s[position]
            curve.append({"time": t, "survival_probability": round(s0 ** hazard_ratio, 3)})
        return curve

    def meta(self, score: float) -> dict[str, Any]:
        """Thông tin trình bày kèm câu giới hạn hiển thị cạnh kết quả."""
        c_index = self.test_metrics.get("c_index")
        ci = self.test_metrics.get("c_index_ci") or [None, None]
        if self.calibrated:
            text = (f"Xác suất ước tính từ {self.n_train} bệnh nhân huấn luyện, kiểm chứng trên {self.n_test} bệnh nhân độc lập "
                    f"(C-index {c_index:.2f}, KTC95% {ci[0]:.2f}–{ci[1]:.2f}).")
        else:
            text = (f"Điểm nguy cơ tương đối so với {self.n_train} bệnh nhân huấn luyện – không phải xác suất sống còn đã hiệu chuẩn. "
                    + (f"Khả năng xếp hạng trên {self.n_test} bệnh nhân độc lập: C-index {c_index:.2f} "
                       f"(KTC95% {ci[0]:.2f}–{ci[1]:.2f}). " if c_index is not None and None not in ci else "")
                    + "Chỉ dùng tham khảo, bác sĩ quyết định.")
        return {
            "calibrated": self.calibrated, "percentile": self.percentile(score), "group": self.group(score),
            "c_index_test": c_index, "c_index_ci": ci, "n_train": self.n_train, "n_test": self.n_test,
            "limitation_text": text,
        }
