"""
Script: record_my_voice.py
Mục đích: Ghi âm 5 lần khẩu lệnh "Tèo đâu" tự nhiên của chủ nhân, trích xuất đặc trưng giọng nói bằng Resemblyzer và lưu vào file `my_voice.pkl`.
Chức năng:
    - Thu âm 5 lần có kiểm tra âm lượng (RMS) chống ghi âm khoảng lặng.
    - Đọc tự nhiên ở khoảng cách ngồi máy tính bình thường (3 lần bình thường, 1 lần hơi nhẹ giọng, 1 lần dứt khoát).
    - Tính toán vector đặc trưng trung bình (Mean Embedding Vector) và lưu đồng thời danh sách vector từng mẫu.
    - Đo độ ổn định (Consistency Score) giữa các lần đọc.
    - Lưu vào `my_voice.pkl` để sử dụng cho tính năng Xác thực giọng nói (Voice Authentication).
"""

import os
import sys
import time
import winsound
from typing import List

import numpy as np
import sounddevice as sd

def setup_windows_console():
    """Đảm bảo process có Console hợp lệ trên Windows ngay cả khi chạy từ GUI exe."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            if not ctypes.windll.kernel32.AttachConsole(-1):
                ctypes.windll.kernel32.AllocConsole()
        except Exception:
            pass
        try:
            if sys.stdout is None or not hasattr(sys.stdout, "write"):
                sys.stdout = open("CONOUT$", "w", encoding="utf-8")
            if sys.stderr is None or not hasattr(sys.stderr, "write"):
                sys.stderr = open("CONOUT$", "w", encoding="utf-8")
            if sys.stdin is None or not hasattr(sys.stdin, "read"):
                sys.stdin = open("CONIN$", "r", encoding="utf-8")
        except Exception:
            pass


# Cấp phát / Đính kèm Console sớm nếu cần
setup_windows_console()

# Đảm bảo UTF-8 trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils import get_configured_microphone
from voice_auth import VoiceAuthenticator, get_authenticator

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_DURATION = 1.8  # 1.8 giây gọn gàng cho câu "Tèo đâu"
NUM_SAMPLES = 5
MIN_RMS_THRESHOLD = 150.0  # Ngưỡng âm lượng tối thiểu cho giọng nói tự nhiên trên mic laptop


def play_start_tone():
    """Phát âm báo bắt đầu nói."""
    try:
        winsound.Beep(1000, 150)
    except Exception:
        pass


def play_stop_tone():
    """Phát âm báo kết thúc thu âm."""
    try:
        winsound.Beep(600, 150)
    except Exception:
        pass


def record_sample(device_idx: int, duration: float = RECORD_DURATION) -> np.ndarray:
    """Thu âm một đoạn âm thanh từ micro."""
    frames_to_read = int(SAMPLE_RATE * duration)
    data = sd.rec(
        frames_to_read,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device_idx,
    )
    sd.wait()
    return data.flatten()


def main():
    print("=" * 68)
    print("      CHƯƠNG TRÌNH ĐĂNG KÝ HỒ SƠ GIỌNG NÓI CHỦ NHÂN (VOICE AUTH)")
    print("                  TÈO ASSISTANT - RESEMBLYZER")
    print("=" * 68)

    mic_idx, mic_name = get_configured_microphone()
    print(f"\n[Microphone]: Đang sử dụng '{mic_name}' (device_id={mic_idx})")
    print(f"Mỗi lần thu âm sẽ kéo dài {RECORD_DURATION:.1f} giây. Bạn chỉ cần nói tự nhiên: 'Tèo đâu'")
    print("\n💡 HƯỚNG DẪN QUAN TRỌNG:")
    print("   Hãy ngồi ở đúng khoảng cách máy tính bình thường và đọc 5 lần tự nhiên")
    print("   (3 lần giọng bình thường, 1 lần hơi nhẹ giọng, 1 lần dứt khoát).\n")

    is_auto_start = ("--record-voice" in sys.argv)
    if is_auto_start:
        print("👉 Đang chuẩn bị... Tự động bắt đầu sau 3 giây!")
        for count in range(3, 0, -1):
            print(f"   Bắt đầu sau {count}s...", end="\r", flush=True)
            time.sleep(1.0)
        print("   Bắt đầu ngay bây giờ!                     \n", flush=True)
    else:
        try:
            if sys.stdin is not None and hasattr(sys.stdin, "readline"):
                input("👉 Nhấn [ENTER] khi bạn đã sẵn sàng bắt đầu...")
            else:
                for count in range(3, 0, -1):
                    print(f"   Bắt đầu sau {count}s...", end="\r", flush=True)
                    time.sleep(1.0)
                print("   Bắt đầu ngay bây giờ!                     \n", flush=True)
        except (RuntimeError, EOFError, OSError):
            for count in range(3, 0, -1):
                print(f"   Bắt đầu sau {count}s...", end="\r", flush=True)
                time.sleep(1.0)
            print("   Bắt đầu ngay bây giờ!                     \n", flush=True)

    auth = get_authenticator()
    if auth.encoder is None:
        print("\n[LỖI] Không thể nạp VoiceEncoder từ Resemblyzer! Vui lòng kiểm tra lại cài đặt.")
        return

    embeddings: List[np.ndarray] = []
    recorded_waves: List[np.ndarray] = []

    take = 1
    while take <= NUM_SAMPLES:
        # Gợi ý đọc 5 lần tự nhiên ở đúng khoảng cách ngồi máy tính bình thường
        if take <= 3:
            style_tip = "giọng bình thường, tự nhiên"
        elif take == 4:
            style_tip = "hơi nhẹ giọng một chút"
        else:
            style_tip = "giọng dứt khoát"

        print(f"\n------------------------------------------------------------")
        print(f"🎙️  LẦN {take}/{NUM_SAMPLES} [{style_tip}]: Chuẩn bị nói 'Tèo đâu'...")
        for count in range(2, 0, -1):
            print(f"    Bắt đầu sau {count}...", end="\r", flush=True)
            time.sleep(0.6)
        print("    >> NÓI NGAY BÂY GIỜ: 'Tèo đâu' !               ")
        play_start_tone()

        # Tiến hành thu âm
        raw_audio_int16 = record_sample(mic_idx, duration=RECORD_DURATION)
        play_stop_tone()

        # Kiểm tra mức năng lượng âm thanh (RMS)
        samples_float = raw_audio_int16.astype(np.float32)
        rms = float(np.sqrt(np.mean(samples_float ** 2))) if len(samples_float) > 0 else 0.0

        if rms < MIN_RMS_THRESHOLD:
            print(f"    [!] Âm thanh quá nhỏ hoặc im lặng (RMS: {rms:.1f} < {MIN_RMS_THRESHOLD}). Vui lòng nói to rõ hơn!")
            time.sleep(0.8)
            continue

        # Trích xuất embedding
        emb = auth.compute_embedding(raw_audio_int16, sample_rate=SAMPLE_RATE)
        if emb is None:
            print("    [!] Không thể trích xuất đặc trưng giọng nói từ đoạn âm này. Thử lại...")
            time.sleep(0.8)
            continue

        embeddings.append(emb)
        recorded_waves.append(raw_audio_int16)
        print(f"    ✓ Đã ghi nhận mẫu giọng lần {take} thành công! (RMS: {rms:.1f})")
        take += 1
        time.sleep(0.4)

    # 5. Tính vector trung bình của 5 mẫu và chuẩn hóa L2
    print("\n" + "=" * 68)
    print(f"  ĐANG TÍNH TOÁN HỒ SƠ SINH TRẮC HỌC GIỌNG NÓI ({NUM_SAMPLES} MẪU)...")
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Đo độ tương đồng của từng mẫu với vector trung bình
    scores = [float(np.dot(e, avg_embedding)) for e in embeddings]
    consistency = float(np.mean(scores)) * 100

    print(f"• Điểm nhất quán giữa {NUM_SAMPLES} lần đọc: {consistency:.1f}%")
    for idx, s in enumerate(scores, 1):
        print(f"    - Mẫu {idx:02d}: {s * 100:.1f}% khớp")

    # Lưu vào my_voice.pkl (cả vector trung bình lẫn danh sách từng mẫu)
    saved_path = auth.save_profile(
        avg_embedding,
        samples_count=NUM_SAMPLES,
        sample_embeddings=embeddings,
    )

    print("\n🎉 HOÀN TẤT ĐĂNG KÝ HỒ SƠ GIỌNG NÓI CHỦ NHÂN!")
    print(f"📁 Hồ sơ đã được lưu tại: {saved_path}")
    print("🛡️ Giờ đây hệ thống sẽ tự động đối chiếu sinh trắc học mỗi khi phát hiện từ khóa 'Tèo đâu'.")
    print("=" * 68)
    print("\n[Cửa sổ sẽ tự động đóng sau 3 giây...]", flush=True)
    time.sleep(3.0)


if __name__ == "__main__":
    main()
