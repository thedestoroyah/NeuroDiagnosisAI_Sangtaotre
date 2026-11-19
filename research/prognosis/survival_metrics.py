"""Chỉ số sống còn cho đánh giá holdout: baseline Breslow, dự đoán S(t), bảng calibration, bootstrap."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def breslow_baseline(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ước lượng hàm sống còn nền S0(t) theo Breslow từ log-hazard dự đoán trên tập train.

    Args:
        risk: log-hazard (đầu ra mô hình) của từng bệnh nhân train.
        time, event: thời gian (tháng) và cờ sự kiện (1 = tử vong).

    Returns:
        (các mốc thời gian có sự kiện, tăng dần; S0 tại các mốc đó).
    """
    risk, time, event = (np.asarray(a, dtype=float) for a in (risk, time, event))
    hazard = np.exp(risk - risk.max())  # trừ max để tránh tràn số; bù lại ở cuối
    event_times = np.unique(time[event == 1])
    increments = []
    for t in event_times:
        deaths = np.sum((time == t) & (event == 1))
        increments.append(deaths / hazard[time >= t].sum())
    cumulative = np.cumsum(increments) * np.exp(-risk.max())
    return event_times, np.exp(-cumulative)


def predict_survival(risk, times_grid: np.ndarray, s0_grid: np.ndarray, t: float):
    """S(t | risk) = S0(t) ** exp(risk); S0 nội suy bậc thang (giá trị tại mốc sự kiện gần nhất ≤ t)."""
    position = np.searchsorted(times_grid, t, side="right") - 1
    s0 = 1.0 if position < 0 else float(s0_grid[position])
    return np.power(s0, np.exp(np.asarray(risk, dtype=float)))


def calibration_table(pred_s: np.ndarray, time: np.ndarray, event: np.ndarray, groups: np.ndarray, t: float) -> pd.DataFrame:
    """So xác suất sống tới t dự đoán trung bình với Kaplan–Meier quan sát trong từng nhóm nguy cơ."""
    rows = []
    for group in sorted(np.unique(groups)):
        members = groups == group
        km = KaplanMeierFitter().fit(np.asarray(time)[members], np.asarray(event)[members])
        observed = float(km.survival_function_at_times(t).iloc[0])
        predicted = float(np.mean(np.asarray(pred_s)[members]))
        rows.append({"group": int(group), "n": int(members.sum()), "predicted_mean": predicted,
                     "km_observed": observed, "abs_diff": abs(predicted - observed)})
    return pd.DataFrame(rows, columns=["group", "n", "predicted_mean", "km_observed", "abs_diff"])


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float], size: int, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy 95% (phân vị 2,5–97,5) của stat_fn trên n mẫu bootstrap có hoàn lại.

    Args:
        stat_fn: nhận mảng chỉ số đã lấy mẫu, trả về một số.
        size: số phần tử của dữ liệu gốc.
    """
    rng = np.random.default_rng(seed)
    stats = [stat_fn(rng.integers(0, size, size)) for _ in range(n)]
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
