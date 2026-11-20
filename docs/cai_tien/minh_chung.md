# Minh chứng cải tiến Chặng 1 (Vòng Chung kết AI2026)

Nhánh `chang1-cai-tien`. Cài môi trường: `uv venv .venv --python 3.10`, sau đó cài `backend/requirements.txt` (kèm constraints) và `research/requirements.txt` (xem `docs/superpowers/plans/2026-11-19-chang1-cai-tien.md`, Task 0).
Chạy toàn bộ test backend: `cd backend && ../.venv/bin/python -m pytest -q tests`. Chạy test research: `.venv/bin/python -m unittest discover -s research/tests`.

## Yêu cầu 1: Sửa sai lệch schema giữa huấn luyện và phục vụ

| Minh chứng | Vị trí |
|---|---|
| Bảng mapping từng trường (tên, đơn vị, chuẩn hóa, cờ thiếu; trước và sau khi sửa) | [bang_mapping_truong.md](bang_mapping_truong.md) |
| Nguồn duy nhất của phép mã hóa, dùng chung cho train và serving | `backend/ai_core/feature_schema.py` |
| So **từng ô** với đoạn code chép nguyên văn từ notebook train, trên 6 bệnh nhân thật | `backend/tests/test_feature_schema.py::test_matches_notebook_on_real_rows` |
| Đi đường DB (chuỗi "Male"/"1", …) vẫn ra đúng vector của lúc train | `backend/tests/test_clinical_parity.py::test_db_path_equals_training_path` |
| RNA: căn theo tên cột, giữ 4 cột `N_*` (trước đây toàn bộ gene lệch 4 vị trí) | `backend/tests/test_feature_schema_rna.py`, `test_clinical_parity.py::test_rna_series_keeps_n_star` |
| Ảnh: 256px, 6 lát nguyên lấy đều (trước đây 224px, 1 ảnh crop) | `backend/tests/test_image_parity.py` |
| Forward từ đặc trưng cache bằng forward đầy đủ của mô hình serving | `research/tests/test_feature_model.py` |
| **Đầu–cuối:** cùng bệnh nhân test C3L-01327, mô hình v2: đường train cho risk 1,534188, đường serving thật cho 1,531384 (lệch 0,17% độ lệch chuẩn risk của tập train, cùng nhóm nguy cơ; phần lệch do cache đặc trưng tính bằng float16 trên GPU). Với v1: 0,481479 và 0,481394 | `backend/tests/test_end_to_end_parity.py` |

## Yêu cầu 2: Không trình bày tiên lượng quá mức bằng chứng

| Minh chứng | Vị trí |
|---|---|
| Manifest chia train/val/test 629/135/135 theo bệnh nhân, commit trước khi train | `research/prognosis/split_manifest.csv`, [split_summary.md](split_summary.md) |
| v1: chọn mô hình chỉ bằng val (5 seed) | [ket_qua_val.md](ket_qua_val.md), `research/prognosis/runs/selection.json` |
| v2: kế hoạch và luật quyết định commit **trước khi chạy**; chọn bằng cross-validation 5-fold trên train+val (764 ca), ensemble 5 seed, 1000 gen RNA | [ke_hoach_v2.md](ke_hoach_v2.md), [ket_qua_cv_v2.md](ket_qua_cv_v2.md), `research/prognosis/runs_v2/cv_results.json` |
| Bảng chỉ số trên tập test độc lập, mô hình v2 đang triển khai (lần đánh giá test thứ 2): C-index **0,648** (KTC95% 0,585–0,702), đối chứng lâm sàng 0,651, IBS, ablation | [ket_qua_test.md](ket_qua_test.md), `research/prognosis/runs_v2/test_report.json` |
| Lần đánh giá test thứ 1 (v1): C-index 0,599 (KTC95% 0,534–0,657) | [ket_qua_test_v1.md](ket_qua_test_v1.md), `research/prognosis/runs/test_report.json`, `figures/v1/` |
| Baseline Breslow ước lượng từ tập train; calibration ở 12 và 24 tháng | `figures/calibration_12m.png`, `figures/calibration_24m.png`, `figures/km_by_risk_group_test.png` |
| Bất định: bootstrap 1000 lần, trung bình ± độ lệch chuẩn qua 5 seed | [ket_qua_test.md](ket_qua_test.md) |
| Tiêu chí hiển thị xác suất sống còn được chốt trước khi train; không đạt thì chỉ hiện điểm tương đối. v1 đạt (sát ngưỡng); **v2 không đạt**, nên giao diện chỉ hiện percentile so với tập train kèm C-index test | `backend/ai_core/risk_reference.py`, `backend/tests/test_risk_reference.py`, `backend/ai_core/weights/risk_reference.json` (`calibrated: false`) |
| Bỏ baseline sống còn viết cứng, ngưỡng nhóm tùy ý, dữ liệu mẫu giả trên trang Báo cáo, câu diễn giải tuyệt đối | commit `feat: show calibrated survival only when validated…` |
| Script chạy lại được: tải lại WSI từ GDC, cắt patch, chia tập, cache đặc trưng, train, đánh giá | `research/data_prep/`, `research/prognosis/` |
| Ảnh chụp thẻ tiên lượng v2 trên app thật: percentile so với 764 bệnh nhân huấn luyện, nhóm tam phân vị, câu giới hạn có C-index test, không có đường sống còn | `figures/demo_prognosis_v2.png` |

## Yêu cầu 3: Xử lý lỗi MRI theo nguyên tắc an toàn

| Minh chứng | Vị trí |
|---|---|
| Ép **toàn bộ lát lỗi** → `status=error`, `mri_unreadable`, không có `no_tumor_detected` | `backend/tests/test_mri_safety.py::test_all_slices_fail_is_error_not_negative` |
| Lỗi một phần hoặc âm tính đều phải chờ bác sĩ rà soát; quét đủ mọi lát | `test_mri_safety.py` (partial, clean negative, no silent step=2) |
| API trả trạng thái lỗi tách biệt với âm tính | `backend/tests/test_mri_safety_api.py` |
| Giao diện: thẻ đỏ "Không đọc được MRI – chưa có kết luận" khác hẳn thẻ vàng "Không phát hiện u – cần bác sĩ xác nhận"; danh sách và lịch sử ghi "chờ bác sĩ xác nhận" | `frontend/src/components/ai/MriResultCard.tsx` |
| Ảnh chụp app chạy thật (Docker local, 20/11): chuỗi 4 lát toàn file hỏng → thẻ đỏ, không có kết luận âm tính, không có nút PDF; ảnh MRI không có u → thẻ vàng chờ bác sĩ xác nhận | `figures/demo_mri_unreadable.png`, `figures/demo_no_tumor.png` |

## Giới hạn được nêu rõ

- Tập test đã được đánh giá **hai lần**: v1 (0,599), rồi v2 (0,648) theo kế hoạch chốt trước. Lần 2 không dùng kết quả lần 1 để chọn cấu hình, nhưng việc đánh giá lại vẫn làm con số kém "độc lập" hơn một lần duy nhất; cả hai lần đều được báo cáo.
- Mô hình v2 (C-index test 0,648) chỉ **ngang** mô hình CoxPH chỉ dùng lâm sàng (0,651; trên CV là 0,666 so với 0,662). Ablation v2: tắt cả RNA và WSI thì C-index giảm từ 0,648 xuống 0,629. Mức giảm 0,02 nhỏ hơn nhiều so với độ rộng KTC95% (khoảng ±0,06) nên chỉ là dấu hiệu, chưa chứng minh được hai nhánh này có đóng góp; và cũng chưa đủ để vượt các biến lâm sàng với cỡ mẫu hiện có (178 ca có RNA, 151 ca có WSI trong train+val).
- v2 phân biệt tốt hơn v1 nhưng **không hiệu chuẩn**: ở 24 tháng, nhóm nguy cơ cao dự đoán sống 5% trong khi quan sát 22%. Vì vậy giao diện không hiện xác suất sống còn.
- 172/194 ca WSI cũ bị mất. Đã tải lại 177 slide TCGA từ GDC; 17 ca UPENN được coi là thiếu WSI.
- Backbone DenseNet121 dùng trọng số ImageNet, đóng băng và không fine-tune, vì phần cứng không đủ (RTX 3050 4 GB) để Cox loss có risk set đủ lớn.
