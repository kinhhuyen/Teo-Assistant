# 🎙️ Tèo Assistant — Trợ lý ảo AI Desktop

Tèo Assistant là trợ lý ảo tiếng Việt thế hệ mới dành cho hệ điều hành Windows. Ứng dụng hoạt động ngầm nhẹ nhàng dưới khay hệ thống (System Tray), luôn sẵn sàng phản hồi tức thì mỗi khi bạn cất lời gọi **"Tèo đâu"**. Với giao diện quả cầu năng lượng **Glow Orb** phát sáng hiện đại cùng công nghệ nhận diện giọng nói và bảo mật sinh trắc học tân tiến, Tèo Assistant mang đến trải nghiệm điều khiển máy tính bằng giọng nói tiện lợi, nhanh chóng và chính xác.

---

## ✨ Tính năng nổi bật

### 1. Đánh thức bằng giọng nói chuẩn xác ("Tèo đâu")
- Nhận diện từ khóa đánh thức qua kiến trúc lọc âm thanh đa tầng thông minh: kết hợp **WebRTC VAD** và bộ lọc năng lượng âm thanh **RMS**.
- Tự động bỏ qua các tiếng ồn môi trường như tiếng quạt gió, tiếng gõ phím hay tiếng thở nhẹ, loại bỏ hoàn toàn tình trạng kích hoạt ảo.

### 2. Bảo mật sinh trắc học giọng nói (Voice ID)
- Sử dụng mô hình mạng nơ-ron **Resemblyzer** để trích xuất vector đặc trưng giọng nói 256 chiều ngoại tuyến (Offline 100%).
- Đối chiếu độ tương đồng Cosine giữa giọng nói hiện tại với hồ sơ chủ nhân (`my_voice.pkl`). **Chỉ có duy nhất chủ nhân mới có thể kích hoạt trợ lý**, người lạ hay âm thanh từ TV không thể gọi được Tèo.

### 3. Điều khiển máy tính toàn diện & Phản hồi tức thì
- **Mở và tắt ứng dụng siêu tốc:** Cơ chế Fast Path với biểu thức chính quy (Regex) và Windows Shell khởi chạy ứng dụng chỉ trong chưa đầy **0.1 giây**.
- **Điều khiển phần cứng hệ thống:** Tăng/giảm hoặc đặt chính xác âm lượng và độ sáng màn hình theo phần trăm; bật/tắt Wi-Fi và Bluetooth.
- **Tiện ích hằng ngày:** Chụp ảnh màn hình, khóa máy, tắt máy tính, tra cứu giá vàng, thời tiết hoặc mở tìm kiếm trên trình duyệt web.

### 4. Hoạt động bền bỉ cả khi mất mạng (Offline Vosk)
- Kiến trúc nhận diện giọng nói kép: Ưu tiên **Google STT** cho tốc độ phản hồi cực nhanh khi có Internet; tự động chuyển sang mô hình **Vosk Offline** khi đường truyền mạng gặp sự cố hoặc ngắt kết nối.

### 5. Giao diện Glow Orb tinh tế
- Quả cầu ánh sáng trong suốt, phát sáng nhẹ nhàng giữa màn hình mỗi khi bạn gọi Tèo, hiển thị trạng thái lắng nghe và biến mất mượt mà sau khi hoàn thành lệnh.

---

## 🚀 Cài đặt & Sử dụng

### Cách 1: Sử dụng file chạy trực tiếp (.exe) — Khuyên dùng

Bạn không cần cài đặt Python hay bất kỳ thư viện nào, chỉ cần tải về và khởi chạy:

1. Vào mục **[Releases](https://github.com/kinhhuyen/Teo-Assistant/releases)** của dự án và tải về file `TeoAssistant_v1.0.0.zip`.
2. Giải nén thư mục và chạy tệp tin **`TeoAssistant.exe`**.
3. Biểu tượng chữ **T** của Tèo Assistant sẽ xuất hiện dưới khay hệ thống (góc dưới bên phải màn hình).
4. **Cấu hình Micro:** Nhấp chuột phải vào biểu tượng khay hệ thống, vào **🎧 Cấu hình Micro** để chọn đúng microphone của bạn (tai nghe, webcam hoặc micro rời).
5. **Đăng ký Voice ID:**
   - Chọn mục **🎙️ Đăng ký / Cập nhật Giọng nói (Voice ID)** từ menu.
   - Một cửa sổ dòng lệnh sẽ xuất hiện và đếm ngược 3 giây.
   - Hãy đọc 10 lần khẩu lệnh *"Tèo đâu"* theo các phong cách khác nhau (bình thường, nhỏ giọng, nhanh, chậm) để AI ghi nhớ hồ sơ giọng nói của bạn.
6. Bắt đầu trải nghiệm: Nói to rõ ràng **"Tèo đâu"**, chờ quả cầu phát sáng và ra lệnh!

---

### Cách 2: Chạy từ mã nguồn (Dành cho lập trình viên)

#### 1. Yêu cầu hệ thống:
- Hệ điều hành: Windows 10 hoặc Windows 11 (64-bit).
- Python: Phiên bản từ `3.10` trở lên.
- Microphone hoạt động ổn định.

#### 2. Cài đặt thư viện:
```bash
git clone https://github.com/kinhhuyen/Teo-Assistant.git
cd Teo-Assistant
pip install -r requirements.txt
```

#### 3. Chạy ứng dụng:
```bash
python app_tray.py
```

---

## 🛠️ Công nghệ sử dụng

| Công nghệ / Thư viện | Vai trò trong hệ thống |
|---|---|
| **Python** | Ngôn ngữ phát triển cốt lõi |
| **PyQt6** | Xây dựng giao diện System Tray và Glow Orb Overlay trong suốt |
| **Resemblyzer & PyTorch** | Xác thực sinh trắc học giọng nói ngoại tuyến (Voice ID) |
| **Google Speech Recognition** | Nhận diện giọng nói tiếng Việt trực tuyến với độ chính xác cao |
| **Vosk & Kaldi** | Bộ nhận diện giọng nói tiếng Việt ngoại tuyến (Offline fallback) |
| **WebRTC VAD** | Phát hiện giọng nói con người và loại bỏ tiếng ồn môi trường |
| **RapidFuzz** | So khớp chuỗi ký tự mờ và chuẩn hóa lệnh thoại |
| **SoundDevice & PortAudio** | Thu âm và quản lý thiết bị âm thanh đầu vào thời gian thực |

---

## 📦 Hướng dẫn đóng gói (Build PyInstaller)

Dự án đã cấu hình sẵn tệp [`TeoAssistant.spec`](file:///D:/Tools/VoicePilot/TeoAssistant.spec) tối ưu theo chế độ phân phối **ONEDIR**:

```bash
python -m PyInstaller TeoAssistant.spec --clean --noconfirm
```

Sau khi hoàn tất, gói ứng dụng độc lập hoàn chỉnh sẽ nằm trong thư mục `dist/TeoAssistant/`.

---

## 💬 Một số khẩu lệnh mẫu

| Danh mục | Câu lệnh mẫu |
|---|---|
| **Đánh thức** | *"Tèo đâu"* |
| **Mở phần mềm** | *"Mở Chrome"*, *"Bật Word"*, *"Mở Excel"*, *"Mở Spotify"*, *"Mở Zalo"* |
| **Đóng phần mềm** | *"Tắt Chrome"*, *"Đóng Notepad"*, *"Thoát Word"* |
| **Điều khiển âm lượng** | *"Tăng âm lượng"*, *"Giảm âm lượng xuống 30%"*, *"Đặt âm lượng 70%"*, *"Tắt tiếng"* |
| **Điều chỉnh độ sáng** | *"Tăng độ sáng"*, *"Giảm độ sáng màn hình"*, *"Đặt độ sáng 80%"* |
| **Kết nối mạng** | *"Bật Wi-Fi"*, *"Tắt Wi-Fi"*, *"Bật Bluetooth"*, *"Tắt Bluetooth"* |
| **Hệ thống** | *"Chụp màn hình"*, *"Khóa màn hình"*, *"Tắt máy tính"* |
| **Tra cứu & Tìm kiếm** | *"Tìm kiếm giá vàng hôm nay"*, *"Thời tiết hôm nay thế nào"* |

---

## 📄 Bản quyền & Đóng góp

Dự án được xây dựng với mục tiêu mang lại giải pháp trợ lý máy tính tiện lợi, mã nguồn mở và bảo mật cho người dùng Việt Nam. Mọi đóng góp, báo lỗi và ý tưởng phát triển thông qua **Issues** hoặc **Pull Requests** đều được hoan nghênh!
