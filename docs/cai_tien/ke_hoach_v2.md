# Kế hoạch cải tiến mô hình tiên lượng lần 2 (chốt trước khi chạy)

Viết ngày 20/11/2026, **trước** khi chạy bất kỳ thí nghiệm nào dưới đây. Tập test (135 ca) đã được đánh giá một lần với mô hình v1 (`ket_qua_test.md`, C-index 0,599). File này cố định cách chọn mô hình v2 để kết quả test lần 2 không bị chọn theo test.

## Vì sao làm lại

Ở v1, mô hình triển khai là seed 3, epoch 3, chọn theo C-index trên một tập val 135 ca. 5 seed cho val 0,64–0,67, chênh nhau trong phạm vi nhiễu của 135 ca, nên việc chọn này gần như chọn theo nhiễu. Ablation trên test cho thấy nhánh RNA không đóng góp: 60.664 chiều đầu vào trong khi chỉ 149 ca train có RNA.

## Ba cải tiến

1. **Ensemble 5 seed:** risk = trung bình log-hazard của 5 mô hình train với 5 seed. Không chọn 1 seed.
2. **Giảm chiều RNA:** chỉ giữ k gen Ensembl có phương sai lớn nhất (sau log1p) tính trên phần train của từng fold. Không dùng nhãn sống còn nên không rò nhãn. Serving chỉ cần danh sách cột trong `rna_reference.npz`.
3. **Chọn bằng cross-validation:** 5-fold phân tầng theo (nguồn dữ liệu, event) trên train + val (764 ca), thay cho một tập val 135 ca. Tập test không được đọc ở bước này.

## Cách chạy CV

- Lưới cấu hình: k ∈ {tất cả 60.664 cột (như v1), 5000, 1000}; với mỗi k so sánh "đơn seed" (trung bình C-index của 5 seed) và "ensemble 5 seed".
- Mỗi fold, mỗi seed: train tối đa 30 epoch, learning rate cố định (AdamW 5e-5, weight decay 5e-2, batch 64, lấy mẫu cân bằng theo nguồn như v1), không early stopping. Ghi risk trên fold giữ lại sau mỗi epoch.
- C-index CV của một cấu hình tại epoch e = trung bình C-index của 5 fold. Số epoch E của cấu hình là epoch có C-index CV cao nhất.
- Tham chiếu thêm: CoxPH chỉ dùng lâm sàng (như `evaluate.py`) trên cùng 5 fold.

## Luật quyết định (cố định)

- R0 = C-index CV của cấu hình gần v1 nhất: tất cả cột, đơn seed.
- Ứng viên c* = cấu hình ensemble có C-index CV cao nhất. Nếu hai cấu hình ensemble chênh nhau dưới 0,005 thì lấy cấu hình ít gen hơn.
- **Chỉ thay mô hình triển khai nếu CV(c*) − R0 ≥ 0,01.** Nếu không đạt, giữ v1 và chỉ báo cáo kết quả CV.
- Nếu đạt: train 5 seed của c* trên toàn bộ 764 ca với đúng E epoch, rồi đánh giá test **một lần** (`evaluate.py --force`, ghi là lần 2). Báo cáo cả lần 1 (v1) và lần 2 (v2).
- **Kết quả test lần 2 không được dùng để quay lại v1 hay chọn lại cấu hình.** v2 được triển khai dù test lần 2 cao hay thấp hơn lần 1.
- Tiêu chí hiển thị xác suất sống còn giữ nguyên như v1 (mọi nhóm |dự đoán − KM| ≤ 0,10 tại 12 và 24 tháng, IBS < Kaplan–Meier). Không đạt thì giao diện chỉ hiện điểm tương đối.
