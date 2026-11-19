# Kết quả chọn mô hình trên tập val (Task 9)

Train lại phần đầu mô hình trên đặc trưng DenseNet121 (ImageNet, đóng băng) đã cache; 629 ca train, 135 ca val (`research/prognosis/split_manifest.csv`). Tập test **không** được đọc ở bước này.

Cấu hình (giữ như notebook gốc): AdamW lr 5e-05, weight decay 0.05, batch 64, ReduceLROnPlateau (×0.5, patience 5) theo val loss, early stopping patience 15 theo C-index val, tối đa 60 epoch. RNA z-score bằng mean/std chỉ của tập train.

| Seed | Số epoch chạy | Epoch tốt nhất | C-index val |
|---:|---:|---:|---:|
| 0 | 20 | 5 | 0.6366 | |
| 1 | 23 | 8 | 0.6561 | |
| 2 | 24 | 9 | 0.6407 | |
| 3 | 18 | 3 | 0.6721 | **được chọn** |
| 4 | 24 | 9 | 0.6621 | |

Checkpoint triển khai: seed 3, epoch 3 (chọn theo C-index val cao nhất). Vì chọn cái tốt nhất trong 5 seed nên C-index val này **lạc quan**; con số dùng để báo cáo là C-index trên tập test (`ket_qua_test.md`).

Nhận xét: mọi seed đạt đỉnh val sớm (epoch 3–9), sau đó train loss tiếp tục giảm còn val loss không giảm, tức overfit nhanh do cỡ mẫu nhỏ (629 ca, nhánh RNA 60.664 chiều chỉ có 149 ca train). Early stopping theo val ngăn chọn các epoch overfit.
