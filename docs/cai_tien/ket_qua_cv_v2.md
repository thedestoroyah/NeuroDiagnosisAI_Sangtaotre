# Kết quả cross-validation chọn mô hình v2

Chạy theo đúng [ke_hoach_v2.md](ke_hoach_v2.md) (đã commit trước khi chạy, `1a91e62`). Dữ liệu: train + val = 764 ca, 5-fold phân tầng theo (nguồn dữ liệu, event), 5 seed mỗi fold, tối đa 30 epoch. Tập test không được đọc. Số liệu gốc: `research/prognosis/runs_v2/cv_results.json`.

## C-index CV (trung bình 5 fold, tại epoch tốt nhất của từng cấu hình)

| Cấu hình RNA | Số cột | Đơn seed | Ensemble 5 seed |
|---|---:|---:|---:|
| Tất cả cột (như v1) | 60.664 | 0,6402 (epoch 27) | 0,6545 (epoch 28) |
| 5000 gen phương sai lớn nhất | 5.000 | 0,6612 (epoch 29) | 0,6643 (epoch 29) |
| 1000 gen phương sai lớn nhất | 1.000 | 0,6633 (epoch 21) | **0,6657 (epoch 23)** |
| Đối chứng CoxPH chỉ dùng lâm sàng | | 0,6615 | |

## Áp luật quyết định

- R0 (tất cả cột, đơn seed) = 0,6402.
- Ứng viên: ensemble 1000 gen, 0,6657. Ensemble 5000 gen thấp hơn 0,0014, nằm trong ngưỡng hòa 0,005, nên theo luật lấy cấu hình ít gen hơn. Ở đây cấu hình ít gen hơn cũng là cấu hình cao nhất.
- Chênh lệch +0,0256 ≥ 0,01, nên **triển khai v2**: ensemble 5 seed, 1000 gen, 23 epoch, train trên toàn bộ 764 ca.

## Nhận xét

- Phần lớn cải thiện đến từ việc giảm chiều RNA (+0,021 khi xét đơn seed). Ensemble thêm +0,002 đến +0,014; tác dụng của ensemble lớn nhất khi mô hình còn nhiều nhiễu (60.664 cột).
- v2 chỉ ngang đối chứng lâm sàng trên CV (+0,004). Chênh lệch này nhỏ hơn nhiều so với độ dao động giữa các fold (0,62–0,70), nên **không thể nói mô hình đa mô thức tốt hơn mô hình chỉ dùng lâm sàng**.
- Với các cấu hình giảm chiều, epoch tốt nhất nằm gần mức trần 30. Có thể train lâu hơn sẽ tốt hơn, nhưng trần 30 epoch đã được chốt trước nên giữ nguyên.
