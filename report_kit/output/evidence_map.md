# Evidence Map - SRGAN x4

| Claim | Evidence source | Evidence snippet | Confidence | Missing? |
|---|---|---|---|---|
| Bài toán được đặt ở upscale x4 | `configs.py:3`; `01_train_srresnet.ipynb:85` | `scale=4` trong config và notebook train phase 1 | High | [OK] |
| Giá trị `hr_crop` mặc định là 128, nhưng run notebook dùng 96 | `configs.py:4`; `01_train_srresnet.ipynb:86`; `02_train_gan.ipynb:77` | Config gốc ghi `hr_crop=128`, notebook override `hr_crop=96` | High | [OK] |
| Dữ liệu được pairing theo key sau khi bỏ suffix `_lr/_hr` | `data.py:34`; `data.py:39` | `_pair_key()` bỏ suffix và `_build_map_with_collision_check()` kiểm tra trùng key | High | [OK] |
| Dataset không bị duplicate key trên train/val/test | Quét file hệ thống thư mục dữ liệu; `data.py:39`; `tests/test_data_pairing.py:43` | Đếm thực tế: train 11544, val 3298, test 1649; duplicate key = 0 | High | [OK] |
| Train augmentation gồm random crop + flip ngang/dọc + rotate 90 | `data.py:86`; `data.py:102` | `_paired_random_crop()` và `_paired_augment()` được gọi khi `train=True` | High | [OK] |
| Phase 1 train SRResNet với nhiều loss mode | `train_srresnet_ddp.py:187`; logs phase1 | `build_loss()` hỗ trợ `l1`, `ssim`, `lpips`; có 3 log phase1 | High | [OK] |
| Phase 2 train GAN có 2 mode `srgan` và `lpips_adv` | `train_gan_ddp.py:178`; `train_gan_ddp.py:179`; `02_train_gan.ipynb:86-116` | `valid_g_loss_modes = {'srgan','lpips_adv'}` và notebook set weight theo mode | High | [OK] |
| Hệ thống theo dõi đầy đủ PSNR/SSIM/LPIPS và loss G/D theo epoch | `engine.py:7-28`; `metrics.py:5`; `metrics.py:22`; csv GAN logs | `GAN_LOG_FIELDS` chứa train/val metric và loss; phase1 logs có psnr/ssim/lpips | High | [OK] |
| Run `srresnet_l1` đạt PSNR/SSIM val cao nhất trong nhóm phase1 | `weights/(Phase_1)Weights_and_Logs/srresnet_l1_log.csv` | Best val PSNR `33.6870` @epoch 23; best val SSIM `0.8082` @epoch 23 | High | [OK] |
| Run `lpips_adv` đạt LPIPS val tốt nhất toàn bộ các run đã log | `weights/(Phase_2)Weights_and_Logs/lpips_adv/lpips_adv_log.csv` | Best val LPIPS `0.35118` @epoch 62 | High | [OK] |
| Run `srresnet_ssim` có dấu hiệu giảm PSNR ở cuối run | `weights/(Phase_1)Weights_and_Logs/srresnet_ssim_log.csv` | Best val PSNR `29.7316` @epoch 29, final val PSNR `24.7897` @epoch 69 | Medium | [OK] |
| Noise schedule của discriminator giảm dần và về 0 ở run 120 epoch | `02_train_gan.ipynb:97-99`; `lpips_adv_log.csv` cột `noise_std` | noise_std đầu run `0.02`, cuối run `0.0` | High | [OK] |
| Có test để bảo vệ data pairing và parity loss D | `tests/test_data_pairing.py:21,43`; `tests/test_gan_loss_parity.py:15,39` | Test bảo vệ logic pairing và so sánh loss với label policy | High | [OK] |
| Chưa có thông tin thời gian train, cấu hình GPU, latency suy luận trong log chính | Không tìm thấy cột/field tương ứng trong logs và scripts | Chưa có bằng chứng định lượng cho runtime/hardware | High | [CAN_BO_SUNG] |

## Ghi chú sử dụng evidence
- Nếu viết nhận định về "chất lượng cảm nhận", ưu tiên dẫn LPIPS + hình qualitative (C7, C8).
- Nếu viết nhận định về "độ trung thành pixel", ưu tiên PSNR/SSIM + run `srresnet_l1`.
- Mỗi kết luận nên có ít nhất 1 dòng dẫn nguồn trong bảng này.

