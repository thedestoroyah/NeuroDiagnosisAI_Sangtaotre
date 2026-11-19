# Chặng 1 – Cải tiến NeuroDiagnosis AI: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sửa 3 yêu cầu BTC: schema train/serving khớp nhau, tiên lượng đánh giá trên holdout có hiệu chuẩn, MRI lỗi không thành âm tính. Mỗi yêu cầu có test và minh chứng.

**Architecture:**
- Module `backend/ai_core/feature_schema.py` là nguồn duy nhất cho phép mã hóa lâm sàng, RNA và ảnh. Script train/eval trong `research/prognosis/` và `pipeline.py` cùng import module này.
- Train lại phần đầu mô hình trên đặc trưng DenseNet121 (ImageNet) đã tính sẵn, rồi lưu thành `state_dict` của `MultimodalBrainTumorModel` đầy đủ để serving nạp như cũ.
- Pipeline MRI trả trạng thái tường minh (`mri_status`, `review_required`).

**Tech Stack:** Python 3.10, PyTorch 2.5.1/torchvision 0.20.1 (theo `backend/requirements.constraints.txt`), FastAPI/SQLAlchemy, `unittest`, scikit-survival + lifelines, openslide-python + openslide-bin, Next.js (frontend).

**Spec:** `docs/superpowers/specs/2026-11-19-chang1-cai-tien-design.md`

## Global Constraints

- Không sửa gì trong `Vong_chung_ket/NeuroProject` (bản gốc). Mọi thay đổi chỉ nằm trong `NeuroDiagnosisAI_Sangtaotre`.
- Dữ liệu nằm ngoài repo: `DATA=…/Vong_chung_ket/BrainTumor_Research_2025/01_Datasets_Zipped`, gồm `dataset_multimodal_final.csv`, `PROCESSED_MRI/<pid>/{flair|t1ce}/*.png`, `PROCESSED_RNA/rna_matrix.csv`, `WSI_Raw/*.svs`, và thư mục ghi ra `PROCESSED_WSI_v2/<pid>/patch_<n>.png`. Đặc trưng cache ghi vào `DATA/features/`. Không commit dữ liệu.
- Thứ tự 9 trường lâm sàng cố định: `age, gender, race, grade, prior_treatment, radiation_therapy, pharmaceutical_therapy, idh_mutation, mgmt_methylation`. Ô 0–8 là giá trị (thiếu thì ghi 0.0), ô 9–17 là cờ (1.0 = có dữ liệu).
- Mã hóa: `age/100`, gender **male=0, female=1**, `race/3`, `grade/4`, các trường còn lại 0/1. `has_clinical = 1.0` nếu có ít nhất một cờ bằng 1.
- `grade` luôn bị coi là thiếu khi đưa vào mô hình. Ki67 và marker sinh hóa không vào mô hình.
- RNA: căn theo 60.664 cột của `rna_matrix.csv`, **tính cả 4 cột `N_*`**. Thiếu thì ghi 0, sau đó `np.log1p`, rồi z-score bằng mean/std **chỉ của tập train**, cộng `1e-8` vào std.
- Ảnh: 256×256, mean/std ImageNet `[0.485,0.456,0.406]/[0.229,0.224,0.225]`. MRI 6 lát, WSI 8 patch, lấy đều bằng `np.linspace(0, n-1, k, dtype=int)` cho val/test/serving, lấy ngẫu nhiên không lặp khi train. MRI lúc serving là **nguyên lát cắt**, không phải ảnh crop.
- Split 70/15/15 theo bệnh nhân, phân tầng `event × study_type`, seed 2026. Chọn mô hình chỉ dựa trên C-index của val. Tập test chỉ được đọc trong `evaluate.py`.
- 5 seed train: `[0,1,2,3,4]`. Bootstrap 1000 lần cho khoảng tin cậy 95%. Mốc calibration 12 và 24 tháng.
- Tiêu chí hiện xác suất sống còn: tại 12 và 24 tháng, ở mọi nhóm tam phân vị risk, `|dự đoán − KM| ≤ 0.10`, **và** IBS của mô hình < IBS của KM không đặc trưng.
- Docstring và comment viết tiếng Việt, tên hàm/biến viết tiếng Anh.
- Chạy test backend: `cd backend && ../.venv/bin/python -m unittest <module> -v`. Chạy test research: `.venv/bin/python -m unittest discover -s research/tests -v`.
- Commit trên nhánh `chang1-cai-tien`, mỗi task một commit theo dạng `type: mô tả`.

## Review Focus

1. Giá trị lâm sàng dạng chuỗi lấy từ DB/form (`"Male"`, `"F"`, `"yes"`, `"1.0"`, `" "`, `"unknown"`) phải được chuẩn hóa. Chuỗi không nhận ra thì coi là thiếu (cờ 0), không ném lỗi. Test nằm ở Task 1.
2. File RNA upload có Ensembl ID kèm hậu tố version (`ENSG00000000003.15`), cột bị xáo thứ tự, hoặc file TSV: vẫn phải căn đúng. Test nằm ở Task 2.
3. Chuỗi MRI rỗng (0 lát) hoặc mọi lát là byte rác phải ra `mri_unreadable`, không ra `no_tumor_detected`. Test nằm ở Task 4.
4. Risk score nằm ngoài khoảng của tập train: percentile kẹp trong [0, 100], nhóm nguy cơ vẫn hợp lệ. Test nằm ở Task 10.
5. Bệnh nhân không có trường lâm sàng nào: `has_clinical=0` và vector toàn 0, mô hình không trả NaN. Test nằm ở Task 1 và Task 8.

---

### Task 0: Môi trường và nhánh làm việc

**Files:** Create `research/requirements.txt`, sửa `.gitignore` (thêm `.venv/`).

- [ ] **Step 1:** `git checkout -b chang1-cai-tien`
- [ ] **Step 2:** `research/requirements.txt` gồm `scikit-survival==0.23.1`, `lifelines>=0.28`, `openslide-python==1.4.1`, `openslide-bin==4.0.0.6`, `scikit-learn`, `matplotlib`.
- [ ] **Step 3:** Cài môi trường:
  - `uv venv .venv --python 3.10`
  - `uv pip install -p .venv -r backend/requirements.txt -c backend/requirements.constraints.txt -r research/requirements.txt`
  - Torch bản CUDA: dùng `--index-url https://download.pytorch.org/whl/cu121` cho torch/torchvision.
- [ ] **Step 4:** Kiểm tra: `.venv/bin/python -c "import torch,openslide,sksurv;print(torch.cuda.is_available())"`. Kết quả mong đợi là `True`.
- [ ] **Step 5:** `cd backend && ../.venv/bin/python -m unittest discover -s tests -v`. Ghi lại kết quả ban đầu: test nào đã fail từ trước thì ghi vào commit message, không sửa ở plan này.
- [ ] **Step 6:** Commit `research/requirements.txt` và `.gitignore` với message `chore: research env`. *(Spec, plan và `research/data_prep/*` đã được commit trên `main` ở commit f1f06a5, ngày 19/11. Nhánh `chang1-cai-tien` tạo từ commit đó.)*

### Task 1: Mã hóa lâm sàng dùng chung (yêu cầu 1)

**Files:**
- Create: `backend/ai_core/feature_schema.py`
- Test: `backend/tests/test_feature_schema.py`
- Fixture: `backend/tests/fixtures/clinical_rows.csv`, gồm 6 dòng chép nguyên từ `dataset_multimodal_final.csv`. Chọn sao cho có đủ: ca thiếu gender, ca có race=2, ca có idh/mgmt, ca chỉ có age.

**Interfaces:**
- Produces:
  - `CLINICAL_FIELDS: tuple[str, ...]`
  - `SCHEMA_VERSION = "clinical18-v1"`
  - `normalize_clinical_value(field: str, raw: Any) -> float | None`: trả giá trị đã chuẩn hóa nhưng **chưa chia scale**, ví dụ age 65 → 65.0, gender "Male" → 0.0.
  - `encode_clinical(record: Mapping[str, Any]) -> tuple[list[float], float]`: trả 18 giá trị và `has_clinical`.

- [ ] **Step 1: Viết test.** Trong file test, chép **nguyên văn** đoạn mã hóa lâm sàng từ cell 2 của `Bản sao của 07_training_loop.ipynb` thành hàm `_notebook_reference(row)`. Đây là minh chứng "đường train gốc".
  - `test_matches_notebook_on_real_rows`: với mỗi dòng fixture, `encode_clinical(row)[0] == _notebook_reference(row)[0]` từng ô (`assertAlmostEqual`, places=6). `has_clinical` cũng bằng nhau. Riêng grade: notebook gặp NaN nên vẫn khớp.
  - `test_gender_encoding`: `"male"`, `"Male"`, `"M"`, `0` → ô 1 = 0.0, cờ ô 10 = 1.0. `"female"`, `"F"`, `1` → ô 1 = 1.0.
  - `test_grade_always_missing`: `{"grade": "4"}` → ô 3 == 0.0, ô 12 == 0.0.
  - `test_unparseable_is_missing`: `{"prior_treatment": "unknown", "gender": " "}` → các cờ tương ứng bằng 0, không ném lỗi.
  - `test_empty_record`: `encode_clinical({}) == ([0.0]*18, 0.0)`.
  - `test_ignores_mri_and_ki67`: thêm `ki67_index=40` và `classification_confidence=0.9` thì vector không đổi.
- [ ] **Step 2:** `cd backend && ../.venv/bin/python -m unittest tests.test_feature_schema -v`. Mong đợi: FAIL (`ModuleNotFoundError: ai_core.feature_schema`).
- [ ] **Step 3: Viết `feature_schema.py`.** Bộ chuẩn hóa nhận được:
  - bool/int/float;
  - chuỗi số (`"1"`, `"1.0"`);
  - `yes/no/true/false` cho các trường 0/1;
  - `male/m/female/f` cho gender.

  NaN, `None`, chuỗi rỗng hoặc chuỗi lạ trả `None`. Scale: `{"age":100,"race":3,"grade":4}`, các trường còn lại scale 1. `MODEL_IGNORED_FIELDS = {"grade"}`.
- [ ] **Step 4:** Chạy lại test như Step 2. Mong đợi: PASS.
- [ ] **Step 5:** Commit `feat: shared clinical feature schema matching training notebook`.

### Task 2: Mã hóa RNA và hằng số ảnh

**Files:** Modify `backend/ai_core/feature_schema.py`. Test `backend/tests/test_feature_schema_rna.py`.

**Interfaces:**
- Produces:
  - `RnaReference(columns: list[str], mean: np.ndarray, std: np.ndarray)`, là dataclass frozen.
  - `fit_rna_reference(train_matrix: pd.DataFrame) -> RnaReference`: input là các dòng `rna_matrix.csv` của tập train, index là patient_id. Hàm điền thiếu bằng 0, áp `log1p`, rồi tính mean/std theo cột (`ddof=1`, giống pandas).
  - `save_rna_reference(ref, path: str)` / `load_rna_reference(path: str) -> RnaReference`, lưu thành `.npz` có `columns`.
  - `encode_rna(values: Mapping[str, float] | pd.Series, ref: RnaReference) -> np.ndarray`: trả float32 có độ dài `len(ref.columns)`.
  - Hằng số: `IMAGE_SIZE=256`, `IMAGENET_MEAN`, `IMAGENET_STD`, `MRI_SLICES=6`, `WSI_PATCHES=8`.
  - `uniform_indices(n: int, k: int) -> list[int]`: trả `[]` nếu n=0, trả `list(range(n))` nếu n ≤ k, còn lại dùng `linspace`.
  - `image_transform() -> transforms.Compose`, gồm Resize 256 → ToTensor → Normalize.

- [ ] **Step 1: Viết test.**
  - `test_encode_matches_training_formula`: tạo ma trận giả 5×(4 cột `N_*` + 6 gene) có NaN. Hai kết quả sau phải bằng nhau với sai số 1e-6:
    - `encode_rna(row, fit_rna_reference(train))`;
    - `((np.log1p(row.fillna(0)) - mean)/(std+1e-8))`, với mean/std tính bằng pandas đúng như cell 16 của notebook.
  - `test_alignment_by_name`: đưa vào Series có cột bị xáo thứ tự, gene có hậu tố `.15`, thiếu 1 gene, thừa 1 gene lạ. Kết quả phải bằng vector đã căn. Gene thiếu ra 0.0, gene lạ bị bỏ.
  - `test_n_star_columns_keep_position`: `ref.columns[:4] == ["N_unmapped","N_multimapping","N_noFeature","N_ambiguous"]` khi fit từ header thật (đọc dòng đầu của `rna_matrix.csv`; nếu không có file thì skip).
  - `test_uniform_indices`: `(0,6)→[]`, `(3,6)→[0,1,2]`, `(50,6)→[0,9,19,29,39,49]`.
- [ ] **Step 2:** Chạy, mong đợi FAIL.
- [ ] **Step 3:** Cài đặt các hàm. Bỏ hậu tố version bằng `name.split(".")[0]` khi tên bắt đầu bằng `ENSG`.
- [ ] **Step 4:** Chạy, mong đợi PASS.
- [ ] **Step 5:** Commit `feat: RNA alignment/normalization and image constants in feature schema`.

### Task 3: Nối schema vào serving (DB, API, task, pipeline, form)

**Files:**
- Modify `backend/models.py:72-93`: thêm cột `race`, `radiation_therapy`, `pharmaceutical_therapy`, kiểu `String`, cho phép null.
- Modify `backend/database.py::run_migrations`: thêm 3 câu `ALTER TABLE clinical_data ADD COLUMN IF NOT EXISTS … VARCHAR;`.
- Modify `backend/schemas.py:26-49` (`ClinicalDataUpdate`/`Response`) và `backend/routers/multimodal.py:128-166` (PATCH) để có 3 trường mới.
- Create `backend/clinical_inputs.py` chứa `build_clinical_record(patient, clinical) -> dict`.
- Modify `backend/task.py` ở 2 chỗ đang dựng `clinical_dict` (khoảng dòng 159 và 403): đổi sang gọi `build_clinical_record`.
- Modify `backend/ai_core/pipeline.py::prepare_clinical_tensor`: đổi chữ ký thành `prepare_clinical_tensor(self, clinical_data: dict) -> tuple[torch.Tensor, float]`, thân hàm gọi `encode_clinical`. Cập nhật lời gọi trong `run_full_prognosis`, bỏ tham số `mri_result`/`seg_mask`.
- Modify `backend/ai_core/pipeline.py::prepare_rna_tensor`: dùng `encode_rna` với `self.rna_reference`. `prepare_rna_tensor(self, rna_data: pd.Series | None)`.
- Modify `task.py` ở 2 chỗ đọc RNA: trả về `pd.Series` có tên cột (dòng đầu của file), không `drop` các cột `N_*` nữa.
- Modify các form lâm sàng ở frontend: `frontend/src/app/(dashboard)/patients/[id]/page.tsx` và `upload/page.tsx`. Thêm 3 trường (race: chọn 0/1/2 kèm nhãn White/Black/Asian theo mã hóa của notebook 04, cần kiểm tra lại; radiation/pharma: Có/Không). Cạnh ô grade thêm dòng chú thích "Grade không tham gia tiên lượng vì dữ liệu huấn luyện không có trường này".
- Create `docs/cai_tien/bang_mapping_truong.md`: một bảng gồm 18 ô và 4 cột `N_*`/RNA. Cột: ô | trường CSV train | trường DB/API | đơn vị | mã hóa | khi thiếu | trước sửa (serving cũ) | sau sửa.
- Test `backend/tests/test_clinical_parity.py`.

**Interfaces:**
- Consumes: Task 1 và Task 2.
- Produces: `build_clinical_record(patient, clinical) -> dict[str, Any]`, trả các key đúng bằng `CLINICAL_FIELDS`. `age` và `gender` lấy từ `patient`, các trường còn lại lấy từ `clinical` (có thể None).
  - `TumorAnalysisPipeline.rna_reference: RnaReference | None`, nạp từ `weights_dir/rna_reference.npz` trong `__init__`. Nếu thiếu file thì `has_rna=0` và ghi cảnh báo, không dùng RNA chưa chuẩn hóa.

- [ ] **Step 1: Viết test.**
  - `test_db_path_equals_training_path`: với 3 dòng fixture, dựng `SimpleNamespace` patient (age, gender dạng `"Male"`/`"Female"` như DB) và clinical (chuỗi `"1"`/`"0"`/None). Kiểm tra `encode_clinical(build_clinical_record(p, c))` bằng `_notebook_reference(row)` từng ô.
  - `test_pipeline_tensor_equals_schema`: gọi `TumorAnalysisPipeline.prepare_clinical_tensor` trên instance tạo bằng `object.__new__` (gán `device="cpu"`). Tensor `[1,18]` phải bằng `encode_clinical`.
  - `test_rna_series_keeps_n_star`: hàm đọc RNA mới của task (tách ra `read_rna_series(bytes, fmt) -> pd.Series`, đặt trong `backend/rna_import.py`) đọc `sample_data/TCGA-12-1093/RNA_sequence.csv`, kiểm tra `"N_unmapped" in series.index` và `len == 60664`.
- [ ] **Step 2:** Chạy, mong đợi FAIL.
- [ ] **Step 3:** Cài đặt theo danh sách Files.
- [ ] **Step 4:** Chạy cả 3 file test của Task 1–3, cùng các test sẵn có `tests.test_multimodal_upload` và `tests.test_local_demo`. Mong đợi: PASS, không có test cũ nào mới bị vỡ.
- [ ] **Step 5:** `cd frontend && npm run lint && npx tsc --noEmit`. Mong đợi: không lỗi mới.
- [ ] **Step 6:** Commit `fix: serve clinical/RNA features with the exact training schema`.

### Task 4: Trạng thái MRI an toàn (yêu cầu 3)

**Files:**
- Modify `backend/ai_core/pipeline.py::run_series_inference` (khoảng dòng 351-435), `_run_mri_core` và `run_full_prognosis`.
- Modify `backend/task.py`: 2 chỗ lưu `analysis` và khối `if result["status"] != "success"`.
- Modify `backend/models.py` (`AnalysisResult`) và `database.py`: thêm cột `mri_status VARCHAR`, `review_required BOOLEAN DEFAULT FALSE`, `review_reasons JSON`, `prognosis_meta JSON` (`prognosis_meta` dùng ở Task 10).
- Modify `backend/schemas.py` (`AnalysisResultResponse`, `ImageAIResultResponse`) và `backend/routers/analysis.py`: chỗ dựng response (khoảng dòng 1097-1160 và 1233-1290) trả thêm 4 trường mới.
- Modify `frontend/src/components/ai/MriResultCard.tsx`.
- Test `backend/tests/test_mri_safety.py`.

**Interfaces:**
- Produces: kết quả pipeline luôn có các key:
  - `mri_status ∈ {"tumor_detected","no_tumor_detected","mri_unreadable"}`;
  - `mri_partial: bool`, `slices_total`, `slices_examined`, `slices_failed: int`;
  - `failed_slice_reasons: list[str]` (tối đa 5 mục);
  - `review_required: bool`, `review_reasons: list[str]`.
- Hằng mã lý do: `"mri_unreadable"`, `"slices_failed"`, `"negative_needs_confirmation"`, `"low_classification_confidence"` (dùng `CLASSIFICATION_REVIEW_THRESHOLD` của `review_utils`).
- `"mri_unreadable"` đi cùng `status="error"`, `error_code="mri_unreadable"`, `error_msg="Không đọc được ảnh MRI (0/{N} lát) – chưa có kết luận, cần kiểm tra lại dữ liệu."`, và không có key `no_tumor_detected`.

- [ ] **Step 1: Viết test.** Tạo pipeline bằng `object.__new__`, `detector`/`classifier` là stub.
  - `test_all_slices_fail_is_error_not_negative`: `load_image` luôn ném `ValueError`, đưa vào 5 lát byte rác `b"\x00garbage"`. Kiểm tra `status=="error"`, `mri_status=="mri_unreadable"`, `"no_tumor_detected" not in result`, `slices_failed==5`, `review_required is True`.
  - `test_empty_series_is_unreadable`: list rỗng ra `mri_unreadable`.
  - `test_partial_failure_flags_review`: 2 lát lỗi, 3 lát đọc được nhưng không có u. Kiểm tra `mri_status=="no_tumor_detected"`, `mri_partial`, và `review_reasons` chứa cả `"slices_failed"` lẫn `"negative_needs_confirmation"`.
  - `test_clean_negative_still_needs_review`: 5 lát đọc được, không có u. Kiểm tra `review_required` và `review_reasons==["negative_needs_confirmation"]`.
  - `test_no_silent_step_two`: 30 lát. Kiểm tra `slices_examined == 30`.
  - `test_task_marks_failed_with_code`: gọi hàm lưu kết quả của task (tách ra `persist_mri_failure(db, task_record, result)` trong `task.py`). Dùng SQLite in-memory như `test_local_demo.py`. Kiểm tra `task_record.status=="failed"`, `error_message` bắt đầu bằng `"Không đọc được ảnh MRI"`, `task_record.result["mri_status"]=="mri_unreadable"`.
- [ ] **Step 2:** Chạy, mong đợi FAIL.
- [ ] **Step 3: Cài đặt.**
  - Thay `except: continue` bằng `except Exception as exc`: tăng `slices_failed`, ghi lý do, `logging.warning`.
  - Bỏ `step = 2`.
  - `_run_mri_core` khi đọc ảnh lỗi trả `mri_unreadable` với cùng các key, `slices_total=1`.
  - `run_full_prognosis`: nếu `mri_status=="mri_unreadable"` thì trả ngay, không chạy fusion. Nếu `no_tumor_detected` thì giữ hành vi hiện tại (bỏ qua risk) và giữ các key review.
  - `task.py`: nếu `result.get("error_code")=="mri_unreadable"` thì gọi `persist_mri_failure`. Ở chỗ lưu `analysis`, ghi thêm 4 cột mới.
- [ ] **Step 4:** Chạy, mong đợi PASS, kèm các test cũ của backend.
- [ ] **Step 5: Frontend.** `MriResultCard`:
  - `mri_status==="mri_unreadable"` (hoặc task failed có `error_message` bắt đầu bằng "Không đọc được ảnh MRI"): khung đỏ, tiêu đề "Không đọc được MRI – chưa có kết luận", hiện `slices_failed/slices_total`.
  - `no_tumor_detected` hoặc `review_required`: khung vàng "Không phát hiện u – cần bác sĩ xác nhận", liệt kê lý do bằng tiếng Việt, cùng nút xác nhận chuyên gia đang có.
  - Không hiện nhãn "Âm tính" ở bất kỳ đâu khi `review_required` và chưa có review `confirmed`.
  - Chạy `npm run lint && npx tsc --noEmit`.
- [ ] **Step 6:** Commit `fix: explicit MRI failure state, never report unreadable MRI as no tumor`.

### Task 5: Đầu vào ảnh của serving khớp với lúc train

**Files:** Modify `backend/ai_core/pipeline.py` (`preprocess_for_multimodal`, `preprocess_tiles_for_multimodal`, `run_series_inference`, `run_full_prognosis`). Test `backend/tests/test_image_parity.py`.

**Interfaces:**
- Produces:
  - `preprocess_slices_for_multimodal(self, images_bgr: list[np.ndarray]) -> torch.Tensor`, trả `[1, S, 3, 256, 256]` với S ≤ 6, lấy đều bằng `uniform_indices`.
  - `preprocess_tiles_for_multimodal` trả `[1, ≤8, 3, 256, 256]`.
  - `run_full_prognosis(..., mri_model_slices: list[np.ndarray] | None = None)`: khi có tham số này, đầu vào MRI của mô hình là các **lát nguyên** đó, không phải `_cropped_img`. Các bước YOLO/U-Net/XAI vẫn chạy trên lát key như cũ.
- `run_series_inference` truyền vào các lát (ảnh BGR đã đọc) mà YOLO phát hiện có u.

- [ ] **Step 1: Viết test.**
  - `test_slice_tensor_shape_and_uniform`: 50 ảnh có pixel đánh dấu theo chỉ số lát, kiểm tra shape `[1,6,3,256,256]` và chọn đúng các chỉ số `[0,9,19,29,39,49]`.
  - `test_same_pixels_as_training_transform`: một PNG trong `PROCESSED_MRI` (skip nếu thiếu), so `preprocess_slices_for_multimodal([cv2.imread])` với `image_transform()(Image.open().convert("RGB"))`, sai số 1e-5. Mục đích là bắt lỗi BGR/RGB.
  - `test_wsi_tiles_capped_at_8`: 20 tile ra 8 tile.
- [ ] **Step 2:** Chạy, mong đợi FAIL.
- [ ] **Step 3:** Cài đặt. Các tensor zero dùng cho mô thức thiếu đổi sang `[1,1,3,256,256]`. Mask `mri_mask`/`wsi_mask` có độ dài S (đều bằng 1) khi có dữ liệu.
- [ ] **Step 4:** Chạy, mong đợi PASS.
- [ ] **Step 5:** Commit `fix: serve MRI/WSI inputs with training resolution and slice sampling`.

### Task 6: Cắt patch WSI

**Files:** Create `research/data_prep/tile_wsi.py`. Test `research/tests/test_tile_wsi.py`.

**Interfaces:**
- Produces: CLI `python research/data_prep/tile_wsi.py --raw $DATA/WSI_Raw --manifest research/data_prep/wsi_gdc_manifest.tsv --out $DATA/PROCESSED_WSI_v2`, và hàm `is_tissue(patch_rgb: np.ndarray, sat_threshold=40, min_fraction=0.3) -> bool`.
- Ảnh ra: `<out>/<pid>/patch_<n>.png`, n chạy từ 0 theo thứ tự quét lưới (y rồi x).
- Logic giống `02_wsi_preprocessing`: level 1 (nếu không có level 1 thì dùng level 0), bước 256 trên level đó, tọa độ đọc là tọa độ level 0 nhân với `int(level_downsamples[level])`.
- Bỏ qua patient đã có trên 10 patch. Ghi `<out>/tiling_log.csv` gồm `patient_id, slide, n_patches, status`.

- [ ] **Step 1: Viết test.**
  - `test_is_tissue`: ảnh trắng → False, ảnh tím H&E bão hòa → True.
  - `test_tiles_real_slide`: nếu `$DATA/WSI_Raw` có file TCGA-06-0130 (slide nhỏ nhất, 22 MB) thì cắt vào thư mục tạm, kiểm tra có trên 0 patch và mọi patch đúng 256×256 RGB. Không có file thì skip.
- [ ] **Step 2:** Chạy, mong đợi FAIL. Step 3: cài đặt. Step 4: chạy, mong đợi PASS.
- [ ] **Step 5:** Chạy thật trên toàn bộ slide đã tải. Kiểm tra `tiling_log.csv` có 177 dòng (hoặc số slide đã có đến hạn chót 22/11), không dòng nào `status=error`.
- [ ] **Step 6:** Commit script, test và `tiling_log.csv` (sao chép vào `research/data_prep/`) với message `feat: reproducible WSI tiling from GDC slides`.

### Task 7: Manifest chia tập

**Files:** Create `research/prognosis/make_split.py` và `research/prognosis/split_manifest.csv` (commit). Test `research/tests/test_split.py`.

**Interfaces:**
- Produces: `split_manifest.csv` có các cột `patient_id, split∈{train,val,test}, study_type, event, has_mri, has_wsi, has_rna`.
  - `has_wsi` được **tính lại** = 1 khi `PROCESSED_WSI_v2/<pid>` có trên 0 patch.
  - `has_mri` = 1 khi thư mục flair/t1ce có PNG.
  - `has_rna` = 1 khi `pid` có trong index của `rna_matrix.csv`.
- Hàm `make_split(df: pd.DataFrame, seed: int = 2026) -> pd.Series`, dùng 2 lần `train_test_split` có `stratify=df.event.astype(int).astype(str)+"_"+df.study_type.astype(int).astype(str)`. Tầng nào ít hơn 3 ca thì gộp vào tầng `event` tương ứng.

- [ ] **Step 1: Viết test** trên dataframe giả 200 dòng:
  - `test_no_overlap_and_ratios`: tập hợp patient của 3 tập rời nhau, kích thước lệch ±2 so với 70/15/15.
  - `test_deterministic`: cùng seed thì cùng kết quả.
  - `test_each_study_in_test`: mọi `study_type` có ít nhất 3 ca đều xuất hiện trong test.
- [ ] **Step 2:** FAIL. Step 3: cài đặt. Step 4: PASS.
- [ ] **Step 5:** Chạy thật. In bảng số ca và số event theo split × study_type, rồi chép bảng này vào `docs/cai_tien/split_summary.md`.
- [ ] **Step 6:** Commit `feat: patient-level train/val/test split manifest (frozen before training)`.

### Task 8: Tính sẵn đặc trưng và forward từ đặc trưng

**Files:**
- Create `research/prognosis/extract_features.py` và `research/prognosis/feature_model.py`.
- Test `research/tests/test_feature_model.py`.

**Interfaces:**
- Consumes: `MultimodalBrainTumorModel`, `ImageEncoder` (`backend/ai_core/architectures/survival_net.py`), `image_transform`.
- Produces:
  - `pooled_backbone_features(encoder: ImageEncoder, images: Tensor[N,3,256,256]) -> Tensor[N,1024]`, tính `global_pool(relu(feature_extractor(x)))` rồi flatten, đúng như `ImageEncoder.forward`.
  - `forward_from_features(model, mri_feats [B,S,1024], wsi_feats [B,T,1024], rna, clinical, has_mri, has_wsi, has_rna, has_clinical, mri_mask, wsi_mask) -> (risk [B,1], attn)`. Hàm này tái hiện `MultimodalBrainTumorModel.forward` từ sau bước backbone: dropout → fc → act → slice_attention cho từng nhánh ảnh, rồi đi tiếp y như forward gốc. Người làm đọc `survival_net.py:351-410` để chép đúng thứ tự, kể cả `layer_norm` và nhánh `s == 0`.
  - Cache: `$DATA/features/{mri,wsi}/<pid>.npy` kiểu float16, shape `[n_ảnh, 1024]`, theo thứ tự file đã sắp xếp như notebook (sort số theo hậu tố `_N`). MRI chọn thư mục `flair` trước, không có thì `t1ce`. Kèm `$DATA/features/index.json` ghi tên file theo thứ tự.
- Backbone: `MultimodalBrainTumorModel(num_genes=60664)` khởi tạo mới (DenseNet121 ImageNet), `eval()`, chạy dưới `torch.no_grad()` + autocast fp16, batch 64.

- [ ] **Step 1: Viết test.**
  - `test_forward_from_features_equals_full_forward`: seed 0, model ở `eval()`, batch 2 với ảnh ngẫu nhiên `[2,3,3,256,256]` và `[2,4,3,256,256]`, rna và clinical ngẫu nhiên, một mẫu `has_wsi=0`. Kết quả `forward_from_features(pooled…)` bằng `model(...)`, `atol=1e-4`. **Đây là bằng chứng đường train và đường serving là cùng một mô hình.**
  - `test_all_missing_no_nan`: mọi `has_* = 0`, clinical toàn 0, risk là số hữu hạn.
- [ ] **Step 2:** FAIL. Step 3: cài đặt. Step 4: PASS.
- [ ] **Step 5:** Chạy `extract_features.py` cho MRI (khoảng 43 nghìn ảnh) và WSI. Kiểm tra số file `.npy` bằng số ca `has_mri=1` / `has_wsi=1` trong manifest.
- [ ] **Step 6:** Commit `feat: cached DenseNet121 features and feature-level forward`.

### Task 9: Train lại

**Files:** Create `research/prognosis/train.py`. Test `research/tests/test_train_smoke.py`.

**Interfaces:**
- Consumes: Task 1, 2, 7, 8. `CoxPHLoss` và `c_index_score` lấy từ `survival_net.py`.
- Produces:
  - CLI `python research/prognosis/train.py --data $DATA --out research/prognosis/runs --seeds 0 1 2 3 4`.
  - Mỗi seed ghi `runs/seed<k>/{best.pth, history.json}`. `best.pth` là `state_dict` của **toàn bộ** `MultimodalBrainTumorModel`, gồm cả backbone ImageNet không đổi.
  - `runs/rna_reference.npz`, fit chỉ trên train.
  - `runs/selection.json` gồm `{"chosen_seed": k, "val_c_index": {...}, "chosen_epoch": e}`.
  - Chép seed được chọn sang `backend/ai_core/weights/best_multimodal_model.pth` (LFS) và `backend/ai_core/weights/rna_reference.npz`. Checkpoint cũ đổi tên thành `best_multimodal_model_legacy_0.6936.pth`, giữ lại làm đối chiếu.
- Cấu hình:
  - AdamW, lr 5e-5, weight_decay 5e-2 (như notebook; chỉnh trên val nếu không hội tụ, ghi vào `history.json`).
  - Batch 64, tối đa 60 epoch, early stop patience 15 theo C-index val.
  - Chỉ tham số **không thuộc** `feature_extractor` có `requires_grad=True`.
  - Giữ `WeightedRandomSampler` theo `study_type` như notebook.
  - Lúc train bốc ngẫu nhiên 6/8 ảnh, lúc val bốc theo `uniform_indices`.
- **Không đọc các dòng `split=="test"`.** Kiểm tra bằng `assert` lúc lọc dataframe.

- [ ] **Step 1: Viết test.** `test_smoke_two_epochs`: tạo cache giả 30 bệnh nhân (vector ngẫu nhiên), chạy 2 epoch. Kiểm tra `best.pth` nạp được vào `MultimodalBrainTumorModel` mới bằng `strict=True`, và hàm lọc dữ liệu ném lỗi nếu nhận dòng test.
- [ ] **Step 2:** FAIL. Step 3: cài đặt. Step 4: PASS.
- [ ] **Step 5:** Chạy thật 5 seed. Ghi C-index val từng seed vào `docs/cai_tien/ket_qua_val.md`.
- [ ] **Step 6:** Commit code, `selection.json`, `history.json` và weights mới qua LFS với message `feat: retrain prognosis heads on frozen-backbone features (5 seeds)`.

### Task 10: Đánh giá trên tập test và tham chiếu risk

**Files:** Create `research/prognosis/evaluate.py` và `research/prognosis/survival_metrics.py`. Test `research/tests/test_survival_metrics.py`.

**Interfaces:**
- Produces:
  - `breslow_baseline(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, trả các mốc thời gian và S0(t).
  - `predict_survival(risk: float | np.ndarray, times_grid, s0_grid, t: float) -> np.ndarray`, tính `S0(t)**exp(risk)`. S0 nội suy bậc thang.
  - `calibration_table(pred_s, time, event, groups, t) -> pd.DataFrame`, gồm các cột `group, n, predicted_mean, km_observed, abs_diff`.
  - `bootstrap_ci(stat_fn, n=1000, seed=0) -> (lo, hi)`.
  - `backend/ai_core/weights/risk_reference.json`, gồm:
    - `schema_version`;
    - `train_risk_sorted` (danh sách float của tập train);
    - `tertile_cuts: [c1, c2]`;
    - `s0: {"t": [...], "s": [...]}`;
    - `calibrated: bool`;
    - `test_metrics: {"c_index", "c_index_ci", "ibs", "ibs_km"}`;
    - `n_train`, `n_test`.
  - `docs/cai_tien/ket_qua_test.md` cùng `docs/cai_tien/figures/{calibration_12m.png, calibration_24m.png, km_by_risk_group_test.png}`.
- Nội dung báo cáo trên test, đọc **một lần**:
  - C-index của seed được chọn, kèm khoảng tin cậy bootstrap.
  - C-index trung bình ± độ lệch chuẩn của 5 seed.
  - IBS (sksurv `integrated_brier_score`, lưới 3–36 tháng) so với KM.
  - Bảng calibration ở 12 và 24 tháng.
  - Đối chứng: `lifelines.CoxPHFitter` (penalizer 0.1) trên age, gender, idh, mgmt (thiếu thì điền trung vị của train và thêm cờ), fit trên train.
  - Ablation: tắt WSI, tắt RNA, tắt cả hai (đặt `has_*=0`).
  - `calibrated` = tiêu chí trong Global Constraints.

- [ ] **Step 1: Viết test.**
  - `test_breslow_zero_risk_matches_nelson_aalen`: khi mọi risk = 0, kiểm tra S0 bằng `exp(-NelsonAalen)` của lifelines, sai số 1e-6.
  - `test_predict_survival_monotone`: S giảm dần theo t, và risk cao hơn cho S thấp hơn.
  - `test_calibration_perfect_model`: dữ liệu giả sinh từ mô hình exponential, kiểm tra `abs_diff < 0.05`.
  - `test_bootstrap_ci_contains_point`.
- [ ] **Step 2:** FAIL. Step 3: cài đặt. Step 4: PASS.
- [ ] **Step 5:** Chạy `evaluate.py` **đúng một lần** trên test. Commit kết quả nguyên trạng, kể cả khi số xấu.
- [ ] **Step 6:** Commit `feat: independent test evaluation, calibration and risk reference`.

### Task 11: Hiển thị tiên lượng trung thực

**Files:**
- Create `backend/ai_core/risk_reference.py`.
- Modify `backend/ai_core/pipeline.py`: thay `get_risk_level` và `build_survival_curve` bằng `RiskReference`, nạp từ `weights_dir/risk_reference.json`.
- Modify `backend/task.py`: lưu `prognosis_meta`.
- Modify `routers/analysis.py`: trả `prognosis_meta`. Phần vẽ đường sống còn trong PDF report (khoảng dòng 756-790) chỉ vẽ khi `calibrated`.
- Modify `frontend/src/components/ai/MriResultCard.tsx` (khoảng dòng 765-830), `frontend/src/app/(dashboard)/reports/page.tsx` và `frontend/src/components/ui/SurvivalCurve.tsx`.
- Modify `README.md` (các dòng 345-430 có 0,6936) và chuỗi C-index trong `_generate_xai_narrative` nếu có.
- Test `backend/tests/test_risk_reference.py`.

**Interfaces:**
- Produces:
  - `RiskReference.load(path) -> RiskReference`.
  - `.percentile(score: float) -> float`, kẹp trong [0, 100].
  - `.group(score: float) -> str`, một trong `"Low"`, `"Medium"`, `"High"` theo tam phân vị của train. Giữ nhãn tiếng Anh vì `_generate_xai_narrative`, agent tools và frontend đang so sánh chuỗi này; frontend tự dịch khi hiển thị. Cập nhật mọi chỗ đang so với `"Very High"`.
  - `.survival_curve(score) -> list[{"time","survival_probability"}] | None`, trả None khi `not calibrated`. Các mốc thời gian 0, 6, …, 36.
  - `.meta(score) -> dict`, gồm `{"calibrated", "percentile", "group", "c_index_test", "c_index_ci", "n_train", "n_test", "limitation_text"}`.
- `limitation_text` khi chưa hiệu chuẩn, đúng nguyên văn: `"Điểm nguy cơ tương đối so với {n_train} bệnh nhân huấn luyện – không phải xác suất sống còn đã hiệu chuẩn. Chỉ dùng tham khảo, bác sĩ quyết định."`
- Khi đã hiệu chuẩn: `"Xác suất ước tính từ {n_train} bệnh nhân huấn luyện, kiểm chứng trên {n_test} bệnh nhân độc lập (C-index {c:.2f}, KTC95% {lo:.2f}–{hi:.2f})."`

- [ ] **Step 1: Viết test.**
  - `test_percentile_clamped`: score rất lớn ra 100.0, rất nhỏ ra 0.0.
  - `test_group_by_tertiles`.
  - `test_curve_hidden_when_not_calibrated`: `survival_curve(...) is None`, và `meta` có `limitation_text` bắt đầu bằng "Điểm nguy cơ tương đối".
  - `test_curve_shown_when_calibrated`: 7 điểm, giảm dần.
  - `test_missing_reference_file`: pipeline không có file thì `risk_score` vẫn được trả, `prognosis_meta.calibrated=False`, không có đường sống còn.
- [ ] **Step 2:** FAIL. Step 3: cài đặt. Step 4: PASS cùng toàn bộ test backend.
- [ ] **Step 5: Frontend.**
  - Có `survival_curve_data` thì vẽ như cũ, kèm `limitation_text` bên dưới.
  - Không có thì hiện thẻ "Điểm nguy cơ tương đối: cao hơn {percentile}% bệnh nhân huấn luyện · Nhóm {group}", kèm khung giới hạn màu xám.
  - Bỏ hiển thị risk score thô 4 chữ số ở trang bệnh nhân, thay bằng percentile và nhóm.
  - Chạy `npm run lint && npx tsc --noEmit`.
- [ ] **Step 6:** Commit `feat: show calibrated survival only when validated, otherwise relative risk with limits`.

### Task 12: Kiểm thử đầu–cuối và gói minh chứng

**Files:** Test `backend/tests/test_end_to_end_parity.py`. Create `docs/cai_tien/minh_chung.md`.

- [ ] **Step 1: Viết test** `test_same_patient_same_risk`. Chọn một bệnh nhân **thuộc tập test** có đủ MRI và RNA (đọc từ manifest, không có dữ liệu thì skip).
  - **Đường train:** feature cache cộng `encode_clinical(row CSV)` cộng `encode_rna(rna_matrix row)`, đưa qua `forward_from_features`.
  - **Đường serving:** `TumorAnalysisPipeline` thật nạp weights mới. Đưa vào 6 PNG lát nguyên (lấy theo `uniform_indices` giống đường train), RNA đọc qua `read_rna_series`, lâm sàng qua `build_clinical_record`. Gọi `run_full_prognosis(..., mri_model_slices=...)`.
  - Hai risk score phải bằng nhau, `atol=1e-3`. Sai số do fp16 cache; nếu vượt thì cache lại bằng fp32 cho ca này.
- [ ] **Step 2:** Chạy, mong đợi PASS. Nếu FAIL thì quay lại task tương ứng, không nới `atol`.
- [ ] **Step 3: Demo trên app local.**
  - Upload một chuỗi MRI toàn file rác: thẻ đỏ.
  - Upload một chuỗi MRI không có u: thẻ vàng.
  - Chạy tiên lượng cho bệnh nhân mẫu: thẻ risk tương đối, hoặc đường sống còn nếu đã hiệu chuẩn.

  Chụp màn hình và lưu vào `docs/cai_tien/figures/demo_*.png`.
- [ ] **Step 4:** `minh_chung.md` liệt kê, theo từng yêu cầu BTC, các mục: file test (đường dẫn và lệnh chạy), bảng, hình, commit.
- [ ] **Step 5:** Commit `docs: evidence package for round-1 improvements`.

---

Báo cáo cải tiến dạng Word/PDF (đặt trên Google Drive) và việc deploy demo nằm ngoài các task code. Hai việc này làm sau Task 12 và dùng nội dung trong `docs/cai_tien/`.
