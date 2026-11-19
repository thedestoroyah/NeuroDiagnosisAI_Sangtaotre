# Thiết kế cải tiến Chặng 1 — Vòng Chung kết Sáng tạo trẻ (AI2026)

Ngày: 19/11/2026 · Hạn nộp: 23:59 26/11/2026 · Repo: `NeuroDiagnosisAI_Sangtaotre` (clone từ `NeuroProject`, bản gốc giữ nguyên)

## 1. Mục đích

BTC yêu cầu sửa 3 điểm, mỗi điểm phải có minh chứng (test, bảng số liệu, màn hình demo). Mục tiêu là sản phẩm **an toàn và trung thực hơn**, không phải điểm số cao hơn. Nếu C-index mới thấp hơn 0,6936 thì báo cáo ghi đúng như vậy.

Nộp: (1) link mã nguồn, (2) link sản phẩm chạy thử, (3) link báo cáo cải tiến PDF/Word.

### Mục tiêu
1. Vector lâm sàng 18 chiều lúc serving **có cùng ý nghĩa** với lúc train. Có bảng mapping và test chứng minh cùng một bệnh nhân cho ra cùng một tensor.
2. Tiên lượng được đánh giá trên **tập test độc lập theo bệnh nhân**. Baseline sống còn ước lượng từ tập train, có kiểm tra calibration, Brier và độ bất định. Xác suất sống còn chỉ được hiển thị khi đạt tiêu chí hiệu chuẩn.
3. MRI lỗi thì trả **trạng thái lỗi riêng**, không bao giờ trả "không phát hiện u". Mọi kết quả không chắc chắn đều phải qua bác sĩ rà soát.

### Không làm
- Không train lại YOLO, U-Net, DenseNet phân loại.
- Không thêm tính năng mới, không sửa NeuroBoard, Agent hay RAG.
- Không đụng vào thư mục `NeuroProject` gốc.

### Thất bại dù vẫn "chạy được"
- Tập test bị dùng trong lúc chọn mô hình, chọn epoch hoặc chọn ngưỡng.
- Giao diện vẫn hiện "xác suất sống còn" khi calibration không đạt.
- Bất kỳ đường nào trả `no_tumor_detected` khi ảnh không đọc được.
- Demo bị hỏng ở ngày nộp.

## 2. Hiện trạng (đã kiểm tra trong code)

| # | Chỗ lỗi | Mô tả |
|---|---|---|
| 1 | `backend/ai_core/pipeline.py::prepare_clinical_tensor` | Ô 1–11 chứa confidence MRI và xác suất lớp, mô hình chưa từng thấy các biến này. Ô 9–17 không phải cờ thiếu. **Gender bị đảo ngược** (train: male=0, female=1; serving: male=1). Có `ki67` dù lúc train không có. Thiếu race và radiation. |
| 1 | `task.py` (2 đường gọi) | Một đường chỉ truyền ki67/biochem/status. Đường kia không truyền idh/mgmt. |
| 1 | `pipeline.py::prepare_rna_tensor` | Train dùng `ln(1+x)` rồi z-score theo gene. Serving dùng `log2(1+x)` và không z-score. **Gene không được căn theo thứ tự cột lúc train.** |
| 1 | `pipeline.py::preprocess_for_multimodal` | Serving resize ảnh về 224px, train dùng 256px. Train đưa 6 lát MRI qua SliceAttention, serving chỉ đưa 1 lát. **Serving đưa ảnh u đã crop theo YOLO, còn train dùng nguyên lát cắt.** |
| 1 | `task.py` đọc RNA | Lúc train, vector gồm 60.664 cột, tính cả 4 cột `N_*` ở đầu. Serving bỏ 4 cột này rồi thêm 4 số 0 vào cuối, nên **toàn bộ gene bị lệch 4 vị trí**. |
| 2 | Notebook `Bản sao của 07_training_loop.ipynb` | Chia 80/20 (`seed=50`). Checkpoint được chọn theo C-index trên chính tập val, không có test. Val dùng transform có augmentation ngẫu nhiên. Mean/std RNA tính trên cả 899 ca, lộ thông tin val vào lúc train. |
| 2 | `pipeline.py::build_survival_curve`, `get_risk_level` | Baseline sống còn viết cứng. Ngưỡng nhóm nguy cơ ±0,5/1,5 chọn tùy ý. |
| 3 | `pipeline.py::run_series_inference` | `except: continue` nuốt mọi lỗi. Lát lỗi và lát không có u bị coi như nhau, nên khi mọi lát đều lỗi vẫn trả `{"status":"success","no_tumor_detected":True}`. Với chuỗi ≥ 20 lát thì chỉ quét cách một (step=2) mà không báo cho người dùng. |

Schema lúc train (nguồn: cell 2 của notebook, kiểm tra lại với `dataset_multimodal_final.csv`):

| Ô | Trường | Mã hóa lúc train | Thiếu trong 899 ca |
|---|---|---|---|
| 0 / 9 | age | năm / 100 | 0 |
| 1 / 10 | gender | male=0, female=1 | 31 |
| 2 / 11 | race | {0,1,2} / 3 | 623 |
| 3 / 12 | grade | / 4 | **899 (thiếu toàn bộ)** |
| 4 / 13 | prior_treatment | 0/1 | 645 |
| 5 / 14 | radiation_therapy | 0/1 | 627 |
| 6 / 15 | pharmaceutical_therapy | 0/1 | 627 |
| 7 / 16 | idh_mutation | 0/1 | 417 |
| 8 / 17 | mgmt_methylation | 0/1 | 616 |

Ô 0–8 là giá trị (thiếu thì ghi 0). Ô 9–17 là cờ (1 = có dữ liệu). `has_clinical` = 1 nếu có ít nhất một cờ bằng 1.

## 3. Các hướng đã cân nhắc và lựa chọn

**Tiên lượng (yêu cầu 2):**
- A. Train lại với split mới 70/15/15 theo bệnh nhân, đánh giá đầy đủ. **Nhóm chọn A.**
- B. Không train lại, chỉ hiển thị điểm nguy cơ tương đối. Bị loại vì yếu khi chấm. B vẫn là chế độ dự phòng của giao diện khi calibration không đạt.

**Dữ liệu WSI bị mất 172/194 ca (zip trên Drive đã xóa):**
- 1. Train chỉ với 22 ca còn lại.
- 2. Tải lại slide TCGA từ GDC. **Nhóm chọn 2:** 177 ca TCGA, mỗi ca 1 slide DX, tổng 56,7 GB, đang tải. 17 ca UPENN được coi là thiếu WSI. Hạn chót phục hồi là tối 22/11, quá hạn thì train với số ca đã có.

**Cách train (máy RTX 3050 4 GB, hết credit Colab A100):**
- 1. Đóng băng DenseNet121 (trọng số ImageNet), tính sẵn đặc trưng cho mọi lát MRI và patch WSI, chỉ train phần đầu mô hình với batch đủ lớn. **Nhóm chọn 1.**
- 2. Fine-tune toàn bộ trên máy: mỗi batch chỉ được khoảng 2 ca, Cox loss gần như vô nghĩa. Bị loại.
- 3. Fine-tune trên Colab T4: phiên hay bị ngắt, khó kịp hạn. Bị loại.
- Không dùng lại trọng số checkpoint cũ ở bất kỳ phần nào, vì nó đã được train trên những ca nay nằm trong tập test mới.

## 4. Thiết kế

### 4.1 Yêu cầu 1: Một schema dùng chung cho train và serving

- Tạo module mới `backend/ai_core/feature_schema.py`, là **nguồn duy nhất** của:
  - `CLINICAL_FIELDS`: thứ tự, đơn vị và phép mã hóa của từng trường, theo bảng ở mục 2.
  - `encode_clinical(record: dict) -> (list[float] 18 phần tử, has_clinical)`.
  - `encode_rna(raw_by_gene: dict|Series, stats) -> vector`. Hàm này căn theo danh sách gene lúc train, áp `ln(1+x)`, rồi z-score bằng mean/std của tập train. Gene thiếu nhận giá trị 0 sau z-score, tức là bằng giá trị trung bình.
  - Hằng số tiền xử lý ảnh: 256px, mean/std ImageNet, 6 lát MRI, 8 patch WSI, lấy đều.
- Script train (mục 4.2) và `pipeline.py` **cùng import** module này. Notebook cũ được giữ nguyên làm tư liệu.
- Quy tắc riêng:
  - **grade**: dữ liệu train thiếu 100%, nên lúc serving luôn ghi là thiếu (giá trị 0, cờ 0), kể cả khi bác sĩ có nhập. Giao diện ghi rõ "Grade không tham gia tiên lượng vì dữ liệu huấn luyện không có trường này".
  - **ki67** và các marker sinh hóa: vẫn hiển thị, không đưa vào mô hình.
- Mở rộng DB `clinical_data` bằng một migration, thêm các cột `race`, `radiation_therapy`, `pharmaceutical_therapy` (cho phép null). Form nhập liệu lâm sàng thêm các trường tương ứng. Cả hai đường gọi trong `task.py` truyền **đủ 9 trường** qua một hàm dựng dict dùng chung.
- File `artifacts` đi kèm checkpoint mới: `rna_genes.txt`, `rna_stats.npz` (mean/std tính trên train), `schema_version`, `risk_reference.json` (xem 4.2).
- MRI lúc serving: resize về 256px. Với chuỗi ảnh, lấy đều 6 lát trong số các lát YOLO phát hiện có u, cho khớp với cách train lấy 6 lát rải đều. Lát cắt lúc train là 50 lát có diện tích u lớn nhất, nên đây là cách gần nhất làm được khi không có nhãn chuyên gia.

**Minh chứng:**
- `docs/cai_tien/bang_mapping_truong.md`: bảng từng trường gồm tên trong CSV train, tên trong DB/API, đơn vị, phép mã hóa, quy tắc khi thiếu, và vị trí ô.
- Test (`unittest`, đặt trong `backend/tests/`):
  - Lấy vài bệnh nhân thật trong `dataset_multimodal_final.csv`, mã hóa theo đường train và theo đường serving (dict giả lập bản ghi DB và Patient), rồi **so giá trị từng ô**, không chỉ so kích thước.
  - Ca riêng: male → 0, female → 1; grade có nhập vẫn ra giá trị 0 và cờ 0; trường thiếu thì cờ 0; `has_clinical`.
  - RNA: file có gene xáo trộn thứ tự và thiếu vài gene phải cho ra đúng vector sau khi căn.
  - Đầu–cuối: cùng một bệnh nhân, risk tính từ đặc trưng đã cache (đường train) bằng risk tính qua toàn bộ mô hình trong `pipeline.py` (đường serving), sai số ≤ 1e-4.

### 4.2 Yêu cầu 2: Train lại và đánh giá trung thực

Code đặt trong `research/prognosis/`, chạy được lại toàn bộ:

1. `make_split.py` tạo `split_manifest.csv` gồm `patient_id, split, study_type, event`.
   - Chia 70/15/15, phân tầng theo `event × study_type`, seed cố định.
   - Commit file này **trước** khi train.
   - Kiểm tra không có `patient_id` nào trùng giữa các tập.
2. `extract_features.py` cho DenseNet121 (ImageNet, chế độ eval) chạy qua **mọi** lát MRI và **mọi** patch WSI, lưu vector 1024 chiều (fp16) cho từng ảnh. Mỗi patch WSI được cắt bằng logic của `02_wsi_preprocessing` (level 1, 256px, độ bão hòa > 40, tỉ lệ mô ≥ 30%) và ghi vào `PROCESSED_WSI_v2/`.
3. `train.py`:
   - Dùng đúng phần đầu của `MultimodalBrainTumorModel` (fc, SliceAttention, OmicEncoder, ClinicalEncoder, AttentionFusion, classifier) trên đặc trưng đã cache.
   - Mỗi epoch bốc ngẫu nhiên 6 lát MRI và 8 patch WSI. Lúc val/test thì lấy đều, cố định.
   - Mean/std RNA chỉ tính trên tập train.
   - Batch ≥ 64. Early stopping và chọn epoch **chỉ dựa trên C-index của val**.
   - Chạy **5 seed**. Checkpoint được triển khai là seed có C-index val tốt nhất, không bao giờ chọn theo test.
   - Lưu thành `state_dict` của mô hình đầy đủ (gồm cả backbone ImageNet) để `pipeline.py` nạp như hiện nay.
4. `evaluate.py` chỉ chạy trên test, **một lần** sau khi đã chốt checkpoint:
   - Ước lượng baseline Breslow `S0(t)` từ risk score trên tập train.
   - C-index (Harrell) trên test, kèm khoảng tin cậy 95% bằng bootstrap 1000 lần. Báo thêm trung bình ± độ lệch chuẩn qua 5 seed.
   - Calibration tại mốc 12 và 24 tháng: chia test thành 3 nhóm theo risk, so xác suất dự đoán trung bình với Kaplan–Meier quan sát được, vẽ biểu đồ.
   - Integrated Brier score có trọng số IPCW, so với mô hình Kaplan–Meier không dùng đặc trưng.
   - Mô hình đối chứng: CoxPH tuyến tính chỉ dùng tuổi, gender, IDH, MGMT trên cùng split.
   - Ablation khi tắt từng mô thức (WSI, RNA) trên test.
   - Xuất `docs/cai_tien/ket_qua_test.md` cùng các hình.
5. `risk_reference.json` lưu phân phối risk của tập train, `S0(t)`, ngưỡng nhóm nguy cơ (theo tam phân vị của train, thay cho ±0,5/1,5) và kết quả calibration (đạt hay không đạt).

**Tiêu chí hiển thị xác suất sống còn.** Định nghĩa trước khi train, không sửa sau khi đã xem test. Phải đạt cả hai:
- Tại 12 và 24 tháng, ở mọi nhóm risk, |xác suất dự đoán − Kaplan–Meier quan sát| ≤ 0,10.
- Integrated Brier score của mô hình nhỏ hơn của Kaplan–Meier không dùng đặc trưng.

**Giao diện** (`SurvivalCurve.tsx` và các trang hiển thị risk):
- **Đạt:** hiện đường sống còn kèm ghi chú "ước lượng từ N bệnh nhân train, kiểm chứng trên M bệnh nhân test, C-index test = x [CI]".
- **Không đạt:** ẩn đường sống còn. Chỉ hiện "Điểm nguy cơ tương đối: cao hơn X% bệnh nhân trong tập huấn luyện" cùng nhóm nguy cơ, và một khung giới hạn ghi rõ "Không phải xác suất sống còn đã hiệu chuẩn".
- Gỡ con số 0,6936 khỏi các chỗ trình bày như hiệu năng của sản phẩm (README, UI, prompt diễn giải XAI). Thay bằng số test mới và ghi rõ nguồn gốc.

### 4.3 Yêu cầu 3: Xử lý lỗi MRI an toàn

- `run_series_inference` đếm riêng ba loại lát: đọc được, lỗi (kèm lý do, ghi log thay cho `except: continue`), và có u. Kết quả trả thêm `mri_status`, `slices_total`, `slices_examined`, `slices_failed`, `review_required` và `review_reasons`.
- Bảng trạng thái:

| Tình huống | `status` | `mri_status` | Kết luận |
|---|---|---|---|
| Mọi lát đều lỗi, hoặc chuỗi rỗng | `error` | `mri_unreadable` | Không kết luận, không chạy tiên lượng dựa trên MRI |
| Có lát đọc được, có u | `success` | `tumor_detected` | Như hiện nay |
| Có lát đọc được, không thấy u | `success` | `no_tumor_detected` | **Luôn cần bác sĩ xác nhận**, không tự kết luận âm tính |
| Có cả lát lỗi lẫn lát đọc được | như hai dòng trên | thêm `partial` | `review_required = true`, ghi rõ "x/y lát không đọc được" |

- Bỏ cơ chế step=2 ngầm. Nếu vẫn cần cho nhanh thì phải báo `slices_examined < slices_total` và đặt `review_required`.
- `run_full_prognosis` và đường MRI đơn ảnh dùng chung logic trạng thái này.
- `task.py` lưu đủ các trường trạng thái. Frontend (`MriResultCard` và trang bệnh nhân) hiển thị **khác hẳn** cho từng trạng thái: đỏ "Không đọc được MRI – chưa có kết luận", vàng "Không phát hiện u – cần bác sĩ xác nhận". Kết quả có `review_required` phải qua bước bác sĩ xác nhận sẵn có thì mới được chốt hoặc xuất báo cáo.

**Minh chứng:**
- Test ép mọi lát đều lỗi (giả lập `load_image` ném lỗi, kèm byte rác thật) phải ra `status=error`, `mri_unreadable`, và **không** có `no_tumor_detected`.
- Test lỗi một phần ra `partial` cùng `review_required`.
- Test có lát đọc được nhưng không có u ra `review_required`.
- Ảnh chụp màn hình demo đặt hai trạng thái cạnh nhau.

## 5. Kế hoạch theo ngày

| Ngày | Việc |
|---|---|
| 19–20/11 | Yêu cầu 3 và module schema của yêu cầu 1, kèm test (không cần dữ liệu). Tải WSI chạy nền. |
| 21/11 | Cắt patch WSI, `make_split` (commit manifest), `extract_features`. |
| 22/11 | Hạn chót phục hồi WSI. Train 5 seed, đối chứng, ablation. |
| 23/11 | `evaluate` trên test (một lần). Xuất bảng và hình. |
| 24/11 | Nạp checkpoint mới vào serving, test đầu–cuối, sửa giao diện tiên lượng. |
| 25/11 | Báo cáo cải tiến (Word/PDF), deploy demo, tài khoản dùng thử. |
| 26/11 | Dự phòng, nộp bài. |

## 6. Để người thực hiện tự quyết
- Thư viện đánh giá sống còn (`scikit-survival` hoặc `lifelines`) và môi trường Python cho research, ví dụ dùng `uv`.
- Siêu tham số head (lr, weight decay, dropout) chọn trên val.
- Định dạng lưu đặc trưng cache.
- Chi tiết migration DB và cách sửa form lâm sàng.
- Kiểm tra lại cách serving cắt tile WSI có khớp 256px/level 1 không, nếu lệch thì đưa về `feature_schema`.

## 7. Còn chờ nhóm cung cấp
- Repo GitHub mới để đẩy mã nguồn. Hiện remote cũ đã được đổi tên thành `upstream`.
- Nơi deploy demo: dùng lại cách tunnel trong `TUNNEL_DEPLOY_GUIDE.md` hay chỗ khác.
- Nơi đặt báo cáo để có link HTTPS (ví dụ Google Drive).
