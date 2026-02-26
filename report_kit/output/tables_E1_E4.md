# Tables E1-E4 - SRGAN x4

## E1 - Dataset Summary

| Tập dữ liệu | Thư mục LR | Thư mục HR | Số ảnh LR | Số ảnh HR | Số cặp match LR-HR | Duplicate key | Ghi chú |
|---|---|---|---:|---:|---:|---:|---|
| Train | `input/datasets/tyantran/Anh_ve_tinh_2/train/train_lr` | `input/datasets/tyantran/Anh_ve_tinh_2/train/train_hr` | 11544 | 11544 | 11544 | 0 | Pairing theo key bỏ suffix `_lr/_hr` |
| Val | `input/datasets/tyantran/Anh_ve_tinh_2/val/val_lr` | `input/datasets/tyantran/Anh_ve_tinh_2/val/val_hr` | 3298 | 3298 | 3298 | 0 | Kích thước hợp lệ với quy trình val |
| Test | `input/datasets/tyantran/Anh_ve_tinh_2/test/test_lr` | `input/datasets/tyantran/Anh_ve_tinh_2/test/test_hr` | 1649 | 1649 | 1649 | 0 | Dùng để đánh giá và minh họa qualitative |

Đọc bảng này để kết luận gì: Bộ dữ liệu đủ cặp LR-HR đầy đủ, pairing ổn định, không phát hiện duplicate key.

## E2 - Cấu hình train chính (Phase 1 + Phase 2)

| Thông số | Phase 1 - SRResNet | Phase 2 - GAN (`srgan`/`lpips_adv`) | Nguồn |
|---|---|---|---|
| Scale | 4 | 4 | `configs.py:3`, `01_train_srresnet.ipynb:85` |
| hr_crop | 96 (override notebook) | 96 (override notebook) | `01_train_srresnet.ipynb:86`, `02_train_gan.ipynb:77` |
| Batch size train | 32 | 32 | `01_train_srresnet.ipynb:89`, `02_train_gan.ipynb:78` |
| Batch size val | 12 | 12 | `01_train_srresnet.ipynb:94`, `02_train_gan.ipynb:79` |
| Epoch target | 300 | 120 | `01_train_srresnet.ipynb:91`, `configs.py:27` |
| Epoch log thực tế | l1: 25, lpips: 20, ssim: 69 | srgan: 38, lpips_adv: 120 | CSV logs |
| Optimizer | Adam | Adam cho G và D | `train_srresnet_ddp.py`, `train_gan_ddp.py` |
| Learning rate | `lr=1e-4` | `lr_g=1e-4`, `lr_d=3e-5` | `01_train_srresnet.ipynb:92`, `02_train_gan.ipynb:91-92` |
| Scheduler | StepLR (có step/gamma) | Multistep (milestones 60,90; gamma 0.5) | `train_srresnet_ddp.py:251`, `02_train_gan.ipynb:103-105` |
| Loss mode | `l1` / `lpips` / `ssim` | `srgan` hoặc `lpips_adv` | `train_srresnet_ddp.py:187`, `train_gan_ddp.py:178-183` |
| Weight loss (`srgan`) | [CAN_BO_SUNG] | perc=1.0, adv=1e-3, lpips=1.0, pixel=0.0 | `02_train_gan.ipynb:110-112` |
| Weight loss (`lpips_adv`) | [CAN_BO_SUNG] | perc=1.0, adv=3e-3, lpips=1.0, pixel=0.0 | `02_train_gan.ipynb:114-116` |
| Label smoothing | [CAN_BO_SUNG] | real=0.9, fake=0.0 | `02_train_gan.ipynb:100-101`, `train_gan_ddp.py:159-161` |
| Ổn định D | [CAN_BO_SUNG] | R1 weight=2.0, interval=8, noise 0.03->0.005 trong 60 epoch | `02_train_gan.ipynb:95-99` |
| Metric theo dõi | loss, PSNR, SSIM, LPIPS | loss G/D, prob D real/fake, PSNR/SSIM/LPIPS, loss_adv, noise_std | `engine.py:7-28` |

Đọc bảng này để kết luận gì: Cấu hình train được quản lý rõ, phase 2 bổ sung nhiều kỹ thuật ổn định và perceptual objective hơn phase 1.

## E3 - Kết quả định lượng theo từng run

| Run | Epoch log | Best val PSNR (epoch) | Best val SSIM (epoch) | Best val LPIPS (epoch) | Final val PSNR | Final val SSIM | Final val LPIPS | Nhận xét nhanh |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| srresnet_l1 | 25 | 33.6870 (23) | 0.8082 (23) | 0.5265 (24) | 33.6676 | 0.8078 | 0.5276 | Mạnh về fidelity (PSNR/SSIM) |
| srresnet_lpips | 20 | 31.7554 (14) | 0.7627 (1) | 0.3658 (20) | 31.6982 | 0.7471 | 0.3658 | LPIPS tốt hơn l1, nhưng PSNR giảm |
| srresnet_ssim | 69 | 29.7316 (29) | 0.7969 (63) | 0.5473 (3) | 24.7897 | 0.7727 | 0.5758 | Có dấu hiệu dao động cuối run |
| srgan_38 | 38 | 31.7195 (7) | 0.7316 (3) | 0.3883 (1) | 30.5676 | 0.6817 | 0.4135 | GAN cải thiện perceptual so với l1, cần theo dõi ổn định |
| lpips_adv_120 | 120 | 31.8171 (8) | 0.7559 (8) | 0.3512 (62) | 31.1628 | 0.7270 | 0.3575 | LPIPS tốt nhất trong các run đã log |

Đọc bảng này để kết luận gì: Không có một run "tốt nhất mọi mặt"; l1 dẫn đầu PSNR/SSIM, còn lpips_adv dẫn đầu LPIPS.

## E4 - Tổng hợp ưu/nhược điểm và bài học

| Hạng mục | Ưu điểm | Nhược điểm | Bằng chứng chính | Bài học / Hành động tiếp |
|---|---|---|---|---|
| Tổ chức dữ liệu | Pairing LR-HR rõ ràng, có kiểm tra duplicate | Chưa có báo cáo thống kê theo nhóm cảnh | `data.py:34-56`, E1 | Giữ pairing key nhất quán khi bổ sung data mới |
| Phase 1 | Dùng để benchmark fidelity rõ ràng | Chưa tối ưu perceptual chất lượng cao | E3: `srresnet_l1` vs `srresnet_lpips` | Chọn phase 1 theo mục tiêu metric ưu tiên |
| Phase 2 GAN | Cải thiện LPIPS, có bộ kỹ thuật ổn định D | Dao động theo epoch, cần chọn checkpoint cẩn thận | E3 + `srgan_log_38.csv`, `lpips_adv_log.csv` | Chọn checkpoint theo best metric + qualitative, không chỉ final |
| Logging & đánh giá | Log csv chi tiết, dễ vẽ chart | Chưa có runtime/GPU/latency | `engine.py:GAN_LOG_FIELDS` | Bổ sung cột runtime ở lần train sau [CAN_BO_SUNG] |
| Kiểm thử | Có test cho pairing và parity loss D | Chưa thấy test end-to-end inference | `tests/*.py` | Thêm test suy luận trên 1 batch mẫu [CAN_BO_SUNG] |

Đọc bảng này để kết luận gì: Dự án đã đặt nền tảng kỹ thuật tốt cho bài cuối kỳ, nhưng cần bổ sung benchmark runtime và đánh giá qualitative hệ thống hơn.

