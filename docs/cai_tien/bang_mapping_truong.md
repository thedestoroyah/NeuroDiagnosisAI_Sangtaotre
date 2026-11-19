# Bảng mapping trường: huấn luyện ↔ serving (Yêu cầu 1)

Nguồn sự thật về lúc train: cell 2 (mã hóa lâm sàng) và cell 16 (RNA) của notebook `Bản sao của 07_training_loop.ipynb`. Code serving sau khi sửa: `backend/ai_core/feature_schema.py`, được cả `research/` (train lại) và `backend/ai_core/pipeline.py` (serving) import chung.

## Vector lâm sàng 18 chiều

Ô 0–8 là giá trị, ô 9–17 là cờ có dữ liệu (1 = có, 0 = thiếu; giá trị thiếu ghi 0). `has_clinical = 1` khi có ít nhất một cờ bằng 1.

| Ô giá trị / cờ | Trường CSV train | Trường DB / API | Đơn vị, miền | Mã hóa | Khi thiếu | Serving trước khi sửa | Sau khi sửa |
|---|---|---|---|---|---|---|---|
| 0 / 9 | `age` | `patients.age` | năm | age / 100 | 0 / cờ 0 | ô 12 = age/100; ô 0 = ki67/100 | ô 0, cờ ô 9 |
| 1 / 10 | `gender` | `patients.gender` | "Male"/"Female"/"M"/"F"/0/1 | **male = 0, female = 1** | 0 / cờ 0 | ô 13, **male = 1 (đảo ngược)** | ô 1, cờ ô 10 |
| 2 / 11 | `race` | `clinical_data.race` (**cột mới**) | 0 White, 1 Black, 2 Asian, 3 khác | race / 3 | 0 / cờ 0 | không có | ô 2, cờ ô 11 |
| 3 / 12 | `grade` | `clinical_data.grade` | WHO 2–4 | grade / 4 | luôn 0 / cờ 0 | ô 14 = grade/4 | **luôn coi là thiếu**: grade thiếu ở 899/899 ca train, mô hình chưa từng thấy giá trị này |
| 4 / 13 | `prior_treatment` | `clinical_data.prior_treatment` | 0/1, "yes"/"no" | 0/1 | 0 / cờ 0 | ô 15 | ô 4, cờ ô 13 |
| 5 / 14 | `radiation_therapy` | `clinical_data.radiation_therapy` (**cột mới**) | 0/1 | 0/1 | 0 / cờ 0 | không có | ô 5, cờ ô 14 |
| 6 / 15 | `pharmaceutical_therapy` | `clinical_data.pharmaceutical_therapy` (**cột mới**) | 0/1 | 0/1 | 0 / cờ 0 | ô 16 (DB không có, nên luôn trống) | ô 6, cờ ô 15 |
| 7 / 16 | `idh_mutation` | `clinical_data.idh_mutation` | 0/1 | 0/1 | 0 / cờ 0 | **không được truyền vào** pipeline | ô 7, cờ ô 16 |
| 8 / 17 | `mgmt_methylation` | `clinical_data.mgmt_methylation` | 0/1 | 0/1 | 0 / cờ 0 | **không được truyền vào** pipeline | ô 8, cờ ô 17 |

Các thứ serving cũ đưa vào vector nhưng lúc train **không có**, nay đã bỏ: `ki67_index` (ô 0), confidence phân loại (ô 1), confidence bbox (ô 2), tỉ lệ diện tích mask (ô 3), xác suất lớp (ô 4–11). Ki67 và các marker sinh hóa vẫn hiển thị trên giao diện, nhưng không vào mô hình.

Giá trị dạng chuỗi lấy từ DB/form được chuẩn hóa bởi `normalize_clinical_value`. Chuỗi không nhận ra (ví dụ "unknown", " ") được coi là thiếu và không gây lỗi.

## RNA-seq (60.664 chiều)

| Bước | Lúc train | Serving trước khi sửa | Sau khi sửa |
|---|---|---|---|
| Cột | 60.664 cột của `rna_matrix.csv`, **gồm 4 cột `N_*` ở đầu** | bỏ 4 cột `N_*` rồi thêm 4 số 0 vào cuối: **toàn bộ gene lệch 4 vị trí**. Đường phân tích MRI bắt đầu đọc từ cột `ENSG` đầu tiên nên cũng lệch | căn **theo tên cột** với danh sách lúc train (`rna_reference.npz`); bỏ hậu tố version `.NN`; cột lạ bị bỏ |
| Giá trị thiếu | `fillna(0)` | `nan_to_num` | có cột nhưng NaN → 0 (như train); không có cột → bằng trung bình tập train (0 sau z-score) |
| Biến đổi | `np.log1p` (log tự nhiên) | `np.log2(x + 1)` | `np.log1p` |
| Chuẩn hóa | z-score theo gene, mean/std tính **trên cả 899 ca** (lộ thông tin val vào lúc train) | **không có** | z-score với mean/std **chỉ của tập train mới**, lưu trong `rna_reference.npz` |

## Ảnh

| | Lúc train | Serving trước khi sửa | Sau khi sửa (Task 5) |
|---|---|---|---|
| Kích thước | 256 × 256 | 224 × 224 | 256 × 256 |
| MRI đưa vào mô hình | 6 lát **nguyên** (lát cắt đã tiền xử lý), lấy đều qua SliceAttention | 1 ảnh **u đã crop theo YOLO** | tối đa 6 lát nguyên lấy đều trong số các lát YOLO phát hiện có u |
| WSI | 8 patch lấy đều | toàn bộ tile được tải lên | tối đa 8 tile lấy đều |
| Chuẩn hóa | mean/std ImageNet | mean/std ImageNet | mean/std ImageNet |

## Kiểm thử chứng minh

| Test | Chứng minh điều gì |
|---|---|
| `backend/tests/test_feature_schema.py::test_matches_notebook_on_real_rows` | Đoạn mã hóa chép nguyên văn từ notebook và `encode_clinical` cho ra cùng 18 giá trị trên 6 bệnh nhân thật |
| `backend/tests/test_clinical_parity.py::test_db_path_equals_training_path` | Đi đường DB (Patient/ClinicalData dạng chuỗi) → `build_clinical_record` → `encode_clinical` vẫn bằng đường train |
| `backend/tests/test_clinical_parity.py::test_pipeline_tensor_equals_schema` | Tensor mà `TumorAnalysisPipeline` đưa vào mô hình đúng bằng vector của schema |
| `backend/tests/test_feature_schema_rna.py::test_encode_matches_training_formula` | Công thức RNA khớp cell 16 của notebook |
| `backend/tests/test_feature_schema_rna.py::test_alignment_by_name` | File có cột xáo trộn, hậu tố version, thiếu hoặc thừa gene vẫn được căn đúng |
| `backend/tests/test_clinical_parity.py::test_rna_series_keeps_n_star` | File RNA mẫu thật giữ đủ 60.664 cột, `N_*` nằm đầu |
| `backend/tests/test_end_to_end_parity.py` (Task 12) | Cùng một bệnh nhân test, đường train và đường serving thật cho ra cùng risk score |

Chạy lại: `cd backend && ../.venv/bin/python -m unittest tests.test_feature_schema tests.test_feature_schema_rna tests.test_clinical_parity -v`
