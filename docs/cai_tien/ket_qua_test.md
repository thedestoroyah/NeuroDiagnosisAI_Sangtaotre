# Kết quả tiên lượng trên tập test độc lập (Yêu cầu 2)

Đánh giá lúc 2026-11-20T02:36:39, **một lần** sau khi đã chốt checkpoint bằng tập val. Train 629 ca, val 135 ca, test 135 ca (`research/prognosis/split_manifest.csv`).
Checkpoint triển khai: seed 3, epoch 3 (chọn theo C-index val = 0.6721).

## Chỉ số chính

| Chỉ số | Giá trị |
|---|---|
| C-index test (seed triển khai) | **0.5992** (KTC95% bootstrap 0.5335–0.6565, 1000 mẫu) |
| C-index test qua 5 seed | 0.6261 ± 0.0164 |
| Integrated Brier score (IPCW, 3–36 tháng) | 0.1958 (Kaplan–Meier không đặc trưng: 0.2025) |
| Đối chứng CoxPH lâm sàng (tuổi, gender, IDH, MGMT) | C-index test 0.6519 |
| C-index val cũ (checkpoint 0,6936, chọn theo val, không có test) | không so trực tiếp được: khác split, khác dữ liệu WSI |

## Ablation trên tập test (tắt mô thức lúc suy luận)

| Cấu hình | C-index test |
|---|---|
| Đầy đủ mô thức | 0.5992 |
| Tắt WSI | 0.5929 |
| Tắt RNA | 0.6030 |
| Tắt cả WSI và RNA | 0.6037 |

## Calibration (nhóm theo tam phân vị risk của tập train)

### 12 tháng

| Nhóm | n | Dự đoán TB | KM quan sát | Chênh lệch |
|---|---:|---:|---:|---:|
| Low | 62 | 0.694 | 0.700 | 0.005 |
| Medium | 41 | 0.562 | 0.585 | 0.023 |
| High | 32 | 0.532 | 0.500 | 0.032 |

### 24 tháng

| Nhóm | n | Dự đoán TB | KM quan sát | Chênh lệch |
|---|---:|---:|---:|---:|
| Low | 62 | 0.370 | 0.469 | 0.099 |
| Medium | 41 | 0.200 | 0.220 | 0.020 |
| High | 32 | 0.173 | 0.219 | 0.046 |

## Tiêu chí hiển thị xác suất sống còn: **ĐẠT**

Tiêu chí chốt trước khi train: tại 12 và 24 tháng, mọi nhóm có |dự đoán − KM| ≤ 0.10, và IBS mô hình < IBS Kaplan–Meier.
Không đạt thì giao diện chỉ hiển thị điểm nguy cơ tương đối (percentile so với tập train) kèm khung giới hạn.

Hình: `figures/calibration_12m.png`, `figures/calibration_24m.png`, `figures/km_by_risk_group_test.png`.
