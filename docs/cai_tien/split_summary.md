# Manifest chia tập (seed 2026, 70/15/15 theo bệnh nhân, phân tầng event × nguồn)

Nguồn: `research/prognosis/split_manifest.csv` (commit trước khi train). Không bệnh nhân nào nằm ở hai tập.

## Số ca (số event) theo nguồn

| Nguồn | train | val | test |
|---|---:|---:|---:|
| CPTAC | 20 (16) | 4 (3) | 4 (3) |
| IvyGAP | 21 (21) | 5 (5) | 5 (5) |
| TCGA-GBM | 93 (83) | 20 (18) | 20 (18) |
| TCGA-LGG | 86 (23) | 18 (5) | 18 (5) |
| UPENN | 409 (409) | 88 (88) | 88 (88) |
| **Tổng** | 629 (552) | 135 (119) | 135 (119) |

## Số ca có từng mô thức

| Mô thức | train | val | test |
|---|---:|---:|---:|
| MRI | 602 | 128 | 131 |
| WSI (tải lại từ GDC) | 126 | 25 | 24 |
| RNA | 149 | 29 | 29 |
