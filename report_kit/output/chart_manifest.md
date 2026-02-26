# Chart Manifest - SRGAN x4

## C1 - Diễn biến PSNR train/val theo epoch (Phase 1)
- Mục đích biểu đồ: Trả lời câu hỏi "loss nào cho khả năng tái tạo cao hơn ở phase 1?"
- Nguồn dữ liệu:
  - `weights/(Phase_1)Weights_and_Logs/srresnet_l1_log.csv` (train_psnr, val_psnr, epoch)
  - `weights/(Phase_1)Weights_and_Logs/srresnet_lpips_log.csv` (train_psnr, val_psnr, epoch)
  - `weights/(Phase_1)Weights_and_Logs/srresnet_ssim_log.csv` (train_psnr, val_psnr, epoch)
- Loại biểu đồ: Line chart (3 run, mỗi run 2 đường train/val).
- Trục X/Y, xử lý:
  - X: epoch
  - Y: PSNR
  - Xử lý: Vẽ từng run riêng hoặc chuẩn hóa trục X theo epoch có sẵn.
- Cách đọc kết quả:
  - Run nào có val PSNR cao và ổn định hơn thì tốt hơn theo fidelity.
  - Khoảng cách train-val quá lớn cho thấy nguy cơ overfit.
- Caption mẫu: "C1 cho thấy xu hướng PSNR theo epoch của 3 loss mode ở phase 1; run L1 duy trì mức PSNR cao nhất."
- Hướng dẫn thủ công:
  1. Import 3 file csv vào Excel/Sheets.
  2. Tạo line chart cho cột `val_psnr` theo `epoch`.
  3. Nếu cần rõ hơn, làm 3 subplots (mỗi run một ô).

## C2 - Diễn biến LPIPS train/val theo epoch (Phase 1)
- Mục đích biểu đồ: Trả lời "loss nào giúp cải thiện perceptual metric ở phase 1?"
- Nguồn dữ liệu: 3 file csv phase 1, cột `train_lpips`, `val_lpips`, `epoch`.
- Loại biểu đồ: Line chart.
- Trục X/Y, xử lý:
  - X: epoch
  - Y: LPIPS (càng thấp càng tốt)
  - Xử lý: Vẽ chung 3 đường val_lpips để nhìn chênh lệch nhanh.
- Cách đọc kết quả:
  - Đường nào thấp hơn thì perceptual tốt hơn.
  - Kiểm tra có dao động bất thường hay không.
- Caption mẫu: "C2 so sánh LPIPS giữa các loss mode phase 1, nhấn mạnh sự đánh đổi với PSNR/SSIM."
- Hướng dẫn thủ công:
  1. Lấy cột `epoch` và `val_lpips` của 3 run.
  2. Vẽ line chart chung.
  3. Đánh dấu epoch tốt nhất của mỗi run.

## C3 - So sánh LPIPS val giữa `srgan` và `lpips_adv` (Phase 2)
- Mục đích biểu đồ: Trả lời "mode GAN nào cho chất lượng cảm nhận tốt hơn?"
- Nguồn dữ liệu:
  - `weights/(Phase_2)Weights_and_Logs/srgan/srgan_log_38.csv` (val_lpips, epoch)
  - `weights/(Phase_2)Weights_and_Logs/lpips_adv/lpips_adv_log.csv` (val_lpips, epoch)
- Loại biểu đồ: Line chart.
- Trục X/Y, xử lý:
  - X: epoch
  - Y: val_lpips
  - Xử lý: Có thể cắt `lpips_adv` đến 38 epoch để so sánh công bằng theo độ dài run.
- Cách đọc kết quả:
  - Đường nào thấp hơn là mode perceptual tốt hơn.
  - Kiểm tra xu hướng hội tụ và mức dao động.
- Caption mẫu: "C3 cho thấy mode `lpips_adv` đạt LPIPS val tốt hơn so với `srgan` trên tập val."
- Hướng dẫn thủ công:
  1. Vẽ 2 line `val_lpips` theo `epoch`.
  2. Thêm marker tại best epoch mỗi run.
  3. Ghi chú giá trị best trên chart.

## C4 - D-real vs D-fake probability (Phase 2)
- Mục đích biểu đồ: Trả lời "discriminator có ổn định trong quá trình train không?"
- Nguồn dữ liệu:
  - `srgan_log_38.csv`: `train_d_real_prob`, `train_d_fake_prob`, `val_d_real_prob`, `val_d_fake_prob`, `epoch`
  - `lpips_adv_log.csv`: các cột tương tự
- Loại biểu đồ: Multi-line chart.
- Trục X/Y, xử lý:
  - X: epoch
  - Y: probability
  - Xử lý: Vẽ riêng cho từng run để tránh rối.
- Cách đọc kết quả:
  - D-real cao và D-fake thấp quá mức có thể báo hiệu D quá mạnh.
  - Khoảng cách quá lớn kéo dài có thể gây bất ổn cho G.
- Caption mẫu: "C4 thể hiện mức độ cân bằng giữa khả năng phân biệt ảnh thật và ảnh giả của discriminator."
- Hướng dẫn thủ công:
  1. Lấy 4 cột probability theo epoch.
  2. Vẽ 4 đường trên cùng chart hoặc 2 chart train/val.
  3. Đánh dấu các epoch dao động mạnh.

## C5 - Loss G/D và noise schedule (Phase 2)
- Mục đích biểu đồ: Trả lời "quá trình tối ưu GAN có ổn định theo setup noise không?"
- Nguồn dữ liệu:
  - `srgan_log_38.csv`: `train_loss_g`, `train_loss_d`, `noise_std`, `epoch`
  - `lpips_adv_log.csv`: cột tương tự
- Loại biểu đồ: Dual-axis line chart (loss ở trục trái, noise_std ở trục phải).
- Trục X/Y, xử lý:
  - X: epoch
  - Y1: train_loss_g, train_loss_d
  - Y2: noise_std
  - Xử lý: Làm 2 chart, mỗi run một chart.
- Cách đọc kết quả:
  - Kiểm tra loss có dao động quá mạnh hay không.
  - Quan sát tác động của noise_std giảm dần đến độ ổn định.
- Caption mẫu: "C5 mô tả quan hệ giữa loss GAN và noise schedule của discriminator trong phase 2."
- Hướng dẫn thủ công:
  1. Vẽ train_loss_g và train_loss_d theo epoch.
  2. Thêm series `noise_std` trên secondary axis.
  3. Thêm ghi chú các mốc noise quan trọng.

## C6 - Biểu đồ cột tổng hợp best metric giữa các run
- Mục đích biểu đồ: Trả lời nhanh "run nào mạnh về metric nào?"
- Nguồn dữ liệu:
  - 5 file log chính (3 phase1 + 2 phase2), lấy best `val_psnr`, best `val_ssim`, best `val_lpips`.
- Loại biểu đồ: Grouped bar chart.
- Trục X/Y, xử lý:
  - X: run name
  - Y: giá trị metric
  - Xử lý: Nên làm 3 chart nhỏ (PSNR, SSIM, LPIPS) để dễ đọc.
- Cách đọc kết quả:
  - Nhìn ngay trade-off giữa fidelity và perceptual.
  - Tránh kết luận một chiều chỉ dựa vào 1 metric.
- Caption mẫu: "C6 tổng hợp best metric của các run, cho thấy sự đánh đổi rõ ràng giữa PSNR/SSIM và LPIPS."
- Hướng dẫn thủ công:
  1. Tạo bảng summary 5 run x 3 metric.
  2. Vẽ 3 biểu đồ cột theo từng metric.
  3. Tô màu run dẫn đầu mỗi metric.

## C7 - Hình qualitative 1 (cảnh dễ): LR vs SR vs HR
- Mục đích biểu đồ: Minh họa trực quan chất lượng ảnh sau upscale.
- Nguồn dữ liệu:
  - LR/HR từ `input/datasets/tyantran/Anh_ve_tinh_2/test/...`
  - Ảnh SR từ notebook `test_srresnet.ipynb` hoặc `test_srgan.ipynb` (đường dẫn output: [CAN_BO_SUNG])
- Loại biểu đồ: Qualitative image panel 1x3.
- Trục X/Y, xử lý:
  - Không dùng trục số.
  - Đặt 3 ảnh cùng crop vị trí: LR, SR, HR.
- Cách đọc kết quả:
  - So sánh độ rõ cạnh và chi tiết nhỏ.
  - Kiểm tra artifact (viền răng cưa, oversharpen, noise giả).
- Caption mẫu: "C7 so sánh trực quan trên mẫu dễ, cho thấy SR giữ bố cục và cải thiện độ nét so với LR."
- Hướng dẫn thủ công:
  1. Chọn 1 ảnh test dễ.
  2. Xuất SR bằng model đã chọn.
  3. Ghép panel 1x3 và zoom 200% nếu cần.

## C8 - Hình qualitative 2 (cảnh khó): LR vs SR vs HR
- Mục đích biểu đồ: Đánh giá mô hình trong trường hợp nhiều texture/mẫu lặp.
- Nguồn dữ liệu:
  - Tương tự C7, ưu tiên ảnh có hoa văn nhỏ hoặc khu vực biên phức tạp.
  - Đường dẫn output SR: [CAN_BO_SUNG]
- Loại biểu đồ: Qualitative image panel 1x3 + patch zoom.
- Trục X/Y, xử lý:
  - Không dùng trục số.
  - Thêm 1 patch zoom 64x64 để so cận chi tiết.
- Cách đọc kết quả:
  - Quan sát mức độ "ảo họa tiết" so với HR thật.
  - Kiểm tra SR có làm mất/nhòe chi tiết nhỏ không.
- Caption mẫu: "C8 đánh giá trường hợp khó, nhấn mạnh điểm mạnh và artifact còn tồn tại của mô hình."
- Hướng dẫn thủ công:
  1. Chọn 1 ảnh test khó.
  2. Ghép LR-SR-HR và cắt thêm patch zoom.
  3. Đánh dấu vị trí patch để người đọc đối chiếu.

## Checklist chất lượng chart trước khi nộp
1. Mỗi chart có nguồn dữ liệu cụ thể (file + cột).
2. Mỗi chart có caption 1-2 câu và kết luận ngắn.
3. C7-C8 bắt buộc có cùng một vùng zoom để so sánh công bằng.
4. Nếu thiếu SR output file path thì đánh dấu [CAN_BO_SUNG], không bịa.

