# Blueprint Báo Cáo Cuối Kỳ - SRGAN x4

## 1) Giới thiệu bài toán
- Mục tiêu mục này: Nêu rõ bài toán tăng độ phân giải ảnh x4 và lý do chọn SRGAN cho đồ án cuối kỳ.
- Ý chính cần viết:
  - Đầu vào là ảnh LR, đầu ra là ảnh SR gần với ảnh HR tham chiếu.
  - Nhiệm vụ đặt trong bối cảnh thực hành: canh tác, ảnh vệ tinh, hoặc bộ ảnh có chi tiết nhỏ.
  - Mục tiêu báo cáo: đánh giá chất lượng ảnh và sự cân bằng giữa độ nét cảm nhận và metric.
  - Hướng tiếp cận: train SRResNet trước, sau đó fine-tune GAN.
- Đoạn mẫu:
  - Đề tài tập trung vào bài toán super-resolution hệ số x4 cho ảnh RGB. Mục tiêu chính là từ ảnh đầu vào chất lượng thấp, mô hình tạo ra ảnh đầu ra rõ hơn và giữ được cấu trúc quan trọng. Nhóm sử dụng hướng 2 phase để ổn định quá trình học: phase 1 train generator theo loss tái tạo, phase 2 đưa thêm discriminator để cải thiện chất lượng cảm nhận. Cách làm này phù hợp với bài tập cuối kỳ vì dễ trình bày, dễ đo lường và dễ so sánh giữa các mode loss.
- Hình/Biểu đồ nên chèn: C6, C7.

## 2) Dữ liệu và tiền xử lý
- Mục tiêu mục này: Trình bày rõ bộ dữ liệu, cách ghép cặp LR-HR, cách crop/augment trong train.
- Ý chính cần viết:
  - Số cặp dữ liệu theo train/val/test.
  - Quy tắc pairing bỏ suffix `_lr`/`_hr` và ghép theo key.
  - Kiểm tra duplicate key để tránh map sai cặp.
  - Train dùng random crop + flip ngang/dọc + xoay 90.
- Đoạn mẫu:
  - Dữ liệu được tổ chức theo cặp LR-HR ở ba tập train, val, test. Trong code, key ghép cặp được tạo bằng cách bỏ hậu tố `_lr` hoặc `_hr`, sau đó lấy giao hai tập key để tạo danh sách cặp hợp lệ. Cách này giúp tránh trường hợp map nhầm khi tên file có số thứ tự dài. Ở chế độ train, mỗi cặp ảnh được random crop theo patch và augment bằng flip/rotate để tăng đa dạng mẫu học. Các bước trên giúp mô hình học được chi tiết tốt hơn trong điều kiện dữ liệu thực tế.
- Hình/Biểu đồ nên chèn: E1 (bảng), C7, C8.

## 3) Kiến trúc mô hình (mức vừa đủ)
- Mục tiêu mục này: Giải thích ngắn gọn kiến trúc SRResNet/SRGAN và vai trò từng thành phần.
- Ý chính cần viết:
  - Generator dùng SRResNet (residual blocks + PixelShuffle upsample).
  - Discriminator dùng DiscriminatorForVGG để phân biệt HR thật/giả.
  - Loss gồm nhóm perceptual/lpips/adversarial tùy mode.
  - Metric theo dõi chính: PSNR, SSIM, LPIPS.
- Đoạn mẫu:
  - Generator trong project là SRResNet, có residual blocks để học đặc trưng và PixelShuffle để phóng to ảnh theo hệ số x4. Khi sang phase GAN, discriminator được thêm vào để đánh giá ảnh sinh ra có giống phân bố ảnh thật hay không. Hệ loss được kết hợp theo từng mode train, ví dụ mode `srgan` ưu tiên perceptual + adversarial, còn mode `lpips_adv` bổ sung trọng số LPIPS để cải thiện chất lượng cảm nhận. Ngoài loss train, báo cáo dùng PSNR, SSIM và LPIPS để đánh giá từ nhiều góc nhìn.
- Hình/Biểu đồ nên chèn: C6.

## 4) Quy trình huấn luyện Phase 1 và Phase 2
- Mục tiêu mục này: Mô tả quy trình train theo thứ tự, để người đọc thấy rõ luồng công việc.
- Ý chính cần viết:
  - Phase 1: train SRResNet với 3 biến thể loss (l1, lpips, ssim).
  - Phase 2: train GAN với 2 mode chính (`srgan`, `lpips_adv`).
  - Có scheduler, label smoothing, R1 interval, noise schedule cho D(CÂN NHẮC KHÔNG CHO VÀO).
  - Lưu log theo epoch và lưu checkpoint theo từng mốc.
- Đoạn mẫu:
  - Sau khi chuẩn bị dữ liệu, nhóm train SRResNet ở phase 1 để tạo nền generator ban đầu. Ba run khác nhau theo loss (l1, lpips, ssim) được log riêng để dễ so sánh. Từ checkpoint phase 1, phase 2 tiếp tục train GAN với hai mode `srgan` và `lpips_adv`. Quá trình này có thêm các kỹ thuật ổn định như label smoothing, R1 regularization theo chu kỳ, và noise schedule cho discriminator. Tất cả chỉ số train/val đều được ghi vào csv để tổng hợp kết quả cuối kỳ.
- Hình/Biểu đồ nên chèn: C1, C2, C3, C4, C5.

## 5) Kết quả thực nghiệm và so sánh
- Mục tiêu mục này: Đưa số liệu cụ thể, so sánh công bằng giữa các run, rút ra kết luận rõ ràng.
- Ý chính cần viết:
  - So sánh 5 run chính bằng bảng E3.
  - Nhóm metric tạo trade-off: L1 mạnh về PSNR/SSIM, lpips_adv mạnh về LPIPS.
  - Phân tích xu hướng final epoch và best epoch.
  - Chốt 1 cấu hình để demo chất lượng hình ảnh(lpips_adv).
- Đoạn mẫu:
  - Kết quả cho thấy mỗi hướng loss có điểm mạnh riêng. Run `srresnet_l1` đạt PSNR/SSIM cao nhất trong nhóm phase 1, phù hợp khi ưu tiên độ trung thành pixel. Run `lpips_adv` ở phase 2 đạt LPIPS tốt nhất, cho thấy chất lượng cảm nhận được cải thiện. Tuy nhiên, GAN có thể dao động theo epoch nên cần báo cáo cả best epoch lẫn final epoch để nhìn đúng tính ổn định. Dựa trên mục tiêu môn học, nhóm có thể chọn một cấu hình ưu tiên LPIPS để demo chất lượng thị giác, đồng thời giữ bảng so sánh PSNR/SSIM để đảm bảo tính đầy đủ.
- Hình/Biểu đồ nên chèn: E3 (bảng), C3, C6, C7, C8.

## 6) Phân tích lỗi, hạn chế và rủi ro
- Mục tiêu mục này: Nêu trung thực các điểm còn yếu và nguyên nhân hợp lý.
- Ý chính cần viết:
  - Có dấu hiệu dao động với một số run GAN.
  - Chưa có log thời gian train/GPU trong tài liệu hiện tại.
  - Chưa có baseline nội suy bicubic được log thành bảng số.
  - Chưa có bảng đánh giá chủ quan theo người dùng.
- Đoạn mẫu:
  - Dự án đạt kết quả tốt về mặt tổng thể nhưng vẫn có hạn chế. Một số run GAN cho thấy metric dao động theo epoch, vì vậy việc chọn checkpoint cần dựa trên best metric thay vì final epoch. Ngoài ra, tài liệu hiện tại chưa ghi rõ thời gian train, cấu hình GPU và baseline bicubic ở dạng bảng số, nên mức độ so sánh thực nghiệm chưa trọn vẹn. Nhóm cần bổ sung các thông tin này để báo cáo thuyết phục hơn và để tái lập kết quả dễ dàng hơn.
- Hình/Biểu đồ nên chèn: C4, C5, E4.

## 7) Kết luận và hướng phát triển
- Mục tiêu mục này: Tổng kết kết quả chính và đưa ra hướng nâng cấp phù hợp cấp độ cuối kỳ.
- Ý chính cần viết:
  - Hệ thống đã hoàn thành pipeline train và đánh giá cho SRGAN x4.
  - Có bằng chứng trade-off rõ ràng giữa metric tái tạo và perceptual.
  - Hướng nâng cấp ngắn hạn: bổ sung baseline, thêm qualitative cases khó, chốt tiêu chí chọn checkpoint.
  - Hướng nâng cấp tiếp: thử ESRGAN/lightweight SR và benchmark tốc độ suy luận.
- Đoạn mẫu:
  - Tổng kết lại, project đã hoàn thành đầy đủ các bước từ dữ liệu đến huấn luyện và đánh giá cho bài toán SRGAN x4. Kết quả cho thấy không có một cấu hình duy nhất tốt nhất cho mọi metric, mà cần chọn theo mục tiêu sử dụng. Trong phạm vi bài tập cuối kỳ, cách trình bày theo hai nhóm metric (fidelity và perceptual) là hợp lý và dễ theo dõi. Trong giai đoạn tiếp theo, nhóm nên bổ sung baseline và đánh giá tốc độ suy luận để tăng giá trị ứng dụng của hệ thống.
- Hình/Biểu đồ nên chèn: C6, E4.

