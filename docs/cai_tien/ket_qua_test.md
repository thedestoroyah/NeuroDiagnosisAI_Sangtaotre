# Kết quả tiên lượng trên tập test độc lập (Yêu cầu 2)

Đánh giá lúc 2026-11-20T20:31:06, sau khi đã chốt mô hình mà không đọc tập test. Train 764 ca, test 135 ca (`research/prognosis/split_manifest.csv`).
Mô hình triển khai: v2: ensemble 5 seed, 23 epoch, 1000 cột RNA (cấu hình 1000, C-index CV 0.6657).

## Chỉ số chính

| Chỉ số | Giá trị |
|---|---|
| C-index test (mô hình triển khai) | **0.6482** (KTC95% bootstrap 0.5854–0.7019, 1000 mẫu) |
| C-index test từng seed (5 seed) | 0.6410 ± 0.0166 |
| Integrated Brier score (IPCW, 3–36 tháng) | 0.1880 (Kaplan–Meier không đặc trưng: 0.2021) |
| Đối chứng CoxPH lâm sàng (tuổi, gender, IDH, MGMT) | C-index test 0.6513 |
| C-index val cũ (checkpoint 0,6936, chọn theo val, không có test) | không so trực tiếp được: khác split, khác dữ liệu WSI |

## Ablation trên tập test (tắt mô thức lúc suy luận)

| Cấu hình | C-index test |
|---|---|
| Đầy đủ mô thức | 0.6482 |
| Tắt WSI | 0.6390 |
| Tắt RNA | 0.6339 |
| Tắt cả WSI và RNA | 0.6286 |

## Calibration (nhóm theo tam phân vị risk của tập train)

### 12 tháng

| Nhóm | n | Dự đoán TB | KM quan sát | Chênh lệch |
|---|---:|---:|---:|---:|
| Low | 50 | 0.840 | 0.835 | 0.005 |
| Medium | 40 | 0.590 | 0.525 | 0.065 |
| High | 45 | 0.356 | 0.456 | 0.101 |

### 24 tháng

| Nhóm | n | Dự đoán TB | KM quan sát | Chênh lệch |
|---|---:|---:|---:|---:|
| Low | 50 | 0.619 | 0.484 | 0.135 |
| Medium | 40 | 0.206 | 0.250 | 0.044 |
| High | 45 | 0.052 | 0.216 | 0.164 |

## Tiêu chí hiển thị xác suất sống còn: **KHÔNG ĐẠT**

Tiêu chí chốt trước khi train: tại 12 và 24 tháng, mọi nhóm có |dự đoán − KM| ≤ 0.10, và IBS mô hình < IBS Kaplan–Meier.
Không đạt thì giao diện chỉ hiển thị điểm nguy cơ tương đối (percentile so với tập train) kèm khung giới hạn.

Hình: `figures/calibration_12m.png`, `figures/calibration_24m.png`, `figures/km_by_risk_group_test.png`.

## Các lần đánh giá test trước

Đây là lần đánh giá test thứ 2. Quy trình chọn mô hình lần này được chốt trước khi chạy (`docs/cai_tien/ke_hoach_v2.md`) và không dùng kết quả test lần trước để chọn.

| Lần | Thời điểm | Mô hình | C-index test (KTC95%) |
|---:|---|---|---|
| 1 | 2026-11-20T02:36:39 | v1: seed 3, epoch 3 | 0.5992 (0.5335–0.6565) |
| 2 | 2026-11-20T20:31:06 | v2: ensemble 5 seed, 23 epoch, 1000 cột RNA (cấu hình 1000, C-index CV 0.6657) | 0.6482 (0.5854–0.7019) |
