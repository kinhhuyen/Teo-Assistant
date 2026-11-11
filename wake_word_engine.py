"""
Module: wake_word_engine.py
Mục đích: Bộ phát hiện từ khóa đánh thức (Wake Word Engine) cho Tèo Assistant.
Kiến trúc 4 Lớp Mới (Google STT làm trọng tài chính - Vosk làm phao cứu sinh Offline):

--- 1. LỚP 1: VAD (PHÁT HIỆN TIẾNG NÓI) ---
- Sử dụng WebRTC VAD để liên tục lắng nghe trên micro nền.
- Khi phát hiện có tiếng người (và RMS > 350), lập tức ghi âm một đoạn 1.6 - 1.8 giây.
- Không dùng Vosk để bắt từ khóa liên tục nữa -> CPU cực nhẹ, loại bỏ hoàn toàn ảo giác.

--- 2. LỚP 2: VOICE AUTH (XÁC THỰC GIỌNG NÓI + LỌC NHIỄU NOISEREDUCE) ---
- Lọc bớt tạp âm môi trường bằng `noisereduce` trước khi trích xuất vector giọng nói.
- Chạy hàm verify_voice() từ voice_auth.py với đoạn âm thanh vừa ghi.
- Nếu file my_voice.pkl chưa có (người dùng mới) -> Bỏ qua xác thực, tiếp tục.
- Nếu file có và đạt cosine >= 0.80 -> Xác thực thành công, chuyển sang Lớp 3.
- CƠ CHẾ XÁC THỰC KÉP: Nếu Voice Auth từ chối (cosine < 0.80 do môi trường ồn), KHÔNG vội hủy bỏ mà chuyển lên Lớp 3.
  Nếu Google STT xác nhận đúng "Tèo đâu" VÀ RMS > 500 -> VẪN KÍCH HOẠT AN TOÀN!

--- 3. LỚP 3: GOOGLE STT (TRỌNG TÀI CHÍNH - ONLINE) ---
- Gửi đoạn âm thanh lên Google STT (speech_recognition với language='vi-VN', timeout 2.0s).
- Nếu kết quả trả về (sau khi bỏ dấu) có chứa "teo dau" -> KÍCH HOẠT WAKE WORD (hiện Glow Orb, bíp).
- Nếu kết quả là chữ khác -> Bỏ qua, quay lại Lớp 1.

--- 4. LỚP 4: VOSK (PHAO CỨU SINH OFFLINE) ---
- Kích hoạt Lớp 4 CHỈ KHI Google STT bị lỗi mạng (Timeout, Connection Error, hoặc mất kết nối).
- Nạp đoạn âm thanh vào Vosk với Grammar cứng: ["tèo đâu", "teo dau", "[unk]"].
- Dùng RapidFuzz kiểm tra độ tương đồng với "teo dau". Nếu >= 70% -> Kích hoạt!
"""

import collections
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import sounddevice as sd
import vosk
from rapidfuzz import fuzz

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    nr = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

from utils import (
    MIC_LOCK,
    get_configured_microphone,
    get_exe_dir,
    get_vosk_model_path,
    resource_path,
)

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so sánh chuỗi không dấu."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()


class BoolCallable:
    """Đối tượng vừa có giá trị boolean vừa có thể gọi hàm () để tương thích ngược."""
    def __init__(self, val: bool):
        self.val = bool(val)
    def __bool__(self):
        return self.val
    def __call__(self):
        return self.val
    def __repr__(self):
        return str(self.val)
    def __eq__(self, other):
        return bool(self.val) == bool(other)


class WakeWordEngine:
    """
    Bộ lắng nghe và phát hiện Wake Word 'Tèo đâu' theo Kiến trúc 4 Lớp (Tối ưu phản hồi):
    - Lớp 1: WebRTC VAD + RMS > 350 -> Thu âm động (dừng khi im lặng 0.6s, tối đa 2.5s).
    - Lớp 2: Voice Auth (Resemblyzer Cosine >= 0.80 + noisereduce). Chuyển sang Lớp 3 xác nhận từ khóa.
    - Lớp 3: Google STT (Trọng tài chính Online, timeout cứng 1.5s, Blacklist, adaptive noisereduce). Xác nhận 'teo dau' -> Kích hoạt Glow Orb & Bíp.
    - Lớp 4: Vosk Offline (Phao cứu sinh khi mất mạng, Grammar cứng ['tèo đâu', 'teo dau', '[unk]'] + Fuzzy >= 70%). Xác nhận 'teo dau' -> Kích hoạt Glow Orb & Bíp.
    """

    SAMPLE_RATE: int = 16000
    FRAME_MS: int = 30  # WebRTC VAD hỗ trợ 10, 20 hoặc 30ms
    BLOCK_SIZE: int = 480  # 480 samples ở 16kHz = đúng 30ms
    RECORD_DURATION_SEC: float = 2.5  # Thu âm tối đa 2.5s nếu người dùng nói liên tục
    SILENCE_LIMIT_SEC: float = 0.8    # Ngưỡng khoảng lặng (0.8s) để dừng thu âm động sớm
    MIN_RECORD_SEC: float = 0.8       # Thời gian thu âm tối thiểu (0.8s) cho từ khóa 2 âm tiết

    TARGET_KEYWORD: str = "teo dau"
    DEFAULT_MIN_RMS: float = 350.0
    DUAL_AUTH_MIN_RMS: float = 500.0  # Ngưỡng RMS cho xác thực kép khi nói âm lượng bình thường
    DEFAULT_FUZZY_THRESHOLD: float = 70.0
    DEFAULT_VOICE_AUTH_THRESHOLD: float = 0.80
    DEFAULT_MIN_COSINE: float = 0.80
    DEFAULT_COOLDOWN_SEC: float = 1.0
    DEFAULT_VAD_MODE: int = 2  # 0: Chất lượng thấp, 3: Lọc nhiễu mạnh nhất
    DEFAULT_GRAMMAR_WORDS: List[str] = ["tèo đâu", "teo dau", "[unk]"]

    # Danh sách Blacklist chặn triệt để các từ gây nhầm lẫn với "Tèo đâu"
    BLACKLIST_PHRASES: List[str] = [
        "tèo đâu rồi", "teo dau roi",
        "mày đi đâu", "may di dau",
        "đi đâu", "di dau",
        "ở đâu", "o dau",
        "sao đâu", "sao dau",
        "có đâu", "co dau",
        "biết đâu", "biet dau",
        "thế đâu", "the dau",
        "đâu rồi", "dau roi",
        "đâu đấy", "dau day",
        "đâu nè", "dau ne",
        "đâu thế", "dau the",
        "đâu vậy", "dau vay",
    ]
    BLACKLIST_WORDS: List[str] = [
        "mèo", "meo",
        "heo",
        "bèo", "beo",
        "kèo", "keo",
        "nèo", "neo",
    ]

    def __init__(
        self,
        vosk_model_path: Optional[str] = None,
        wake_keyword: str = "tèo đâu",
        threshold: float = 0.5,
        on_detected: Optional[Callable[[str, float], None]] = None,
        cooldown_sec: float = 1.0,
        grammar_words: Optional[List[str]] = None,
        input_device: Optional[int] = None,
        min_rms: float = 350.0,
        vad_mode: int = 2,
        min_conf: float = 0.65,
        fuzzy_threshold: float = 70.0,
        voice_auth_threshold: float = 0.80,
        min_cosine: Optional[float] = None,
        stt_verifier: Optional[Callable[..., str]] = None,
        voice_verifier: Optional[Callable[[bytes], Union[bool, Tuple[bool, float]]]] = None,
        on_early_wake: Optional[Callable[[], None]] = None,
        on_early_cancel: Optional[Callable[[], None]] = None,
        silence_limit_sec: float = 0.8,
        **kwargs,
    ) -> None:
        """
        Khởi tạo WakeWordEngine với kiến trúc 4 lớp mới.

        Args:
            vosk_model_path: Đường dẫn mô hình Vosk Offline (làm phao cứu sinh Lớp 4).
            wake_keyword: Từ khóa đánh thức (mặc định: 'tèo đâu').
            threshold: Tham số tương thích ngược.
            on_detected: Callback gọi khi phát hiện Wake Word: callback(keyword, score).
            cooldown_sec: Thời gian chờ tối thiểu giữa 2 lần kích hoạt (mặc định 1.0s).
            input_device: Chỉ số thiết bị Microphone (sounddevice index). None = mặc định.
            min_rms: Ngưỡng âm lượng RMS tối thiểu để kích hoạt thu âm (tăng lên 350.0 để bỏ qua tiếng ồn nhỏ).
            vad_mode: Mức độ lọc nhiễu WebRTC VAD (0 đến 3, mặc định 2).
            fuzzy_threshold: Ngưỡng tương đồng mờ (mặc định 70.0%).
            voice_auth_threshold: Ngưỡng Cosine xác thực giọng chủ nhân (mặc định 0.80).
            min_cosine: Tùy chọn đặt ngưỡng Cosine (mặc định 0.80).
            stt_verifier: Hàm gọi Google STT tùy biến nhận (audio_bytes, timeout=1.5).
            voice_verifier: Hàm xác thực giọng nói tùy biến.
            on_early_wake: [Đã bỏ Optimistic UI] Giữ để tương thích ngược.
            on_early_cancel: [Đã bỏ Optimistic UI] Giữ để tương thích ngược.
            silence_limit_sec: Ngưỡng khoảng lặng (giây) để dừng thu âm động sớm (mặc định 0.6s).
        """
        self.wake_keyword = wake_keyword.lower()
        self.target_keyword = self.TARGET_KEYWORD
        self.threshold = threshold
        self.on_detected = on_detected
        self.on_early_wake = on_early_wake
        self.on_early_cancel = on_early_cancel
        self.silence_limit_sec = silence_limit_sec
        self.cooldown_sec = cooldown_sec
        self.min_rms = min_rms
        self.dual_auth_min_rms = kwargs.get("dual_auth_min_rms", self.DUAL_AUTH_MIN_RMS)
        self.vad_mode = vad_mode
        self.min_conf = min_conf
        self.fuzzy_threshold = fuzzy_threshold
        self.voice_auth_threshold = min_cosine if min_cosine is not None else voice_auth_threshold
        self.min_cosine = self.voice_auth_threshold
        self.stt_verifier = stt_verifier
        self.voice_verifier = voice_verifier
        self.grammar_words = grammar_words or list(self.DEFAULT_GRAMMAR_WORDS)
        self._skip_noisereduce_stt: bool = False

        # 1. Khởi tạo bộ phát hiện tiếng người WebRTC VAD (Lớp 1)
        self.vad = None
        if webrtcvad is not None:
            try:
                self.vad = webrtcvad.Vad(self.vad_mode)
                print(f"[WakeWord] ✓ Đã kích hoạt WebRTC VAD (mode={self.vad_mode}) cho Lớp 1.")
            except Exception as vad_err:
                print(f"[WakeWord Cảnh báo] Không thể khởi tạo WebRTC VAD: {vad_err}")
                self.vad = None
        else:
            print("[WakeWord Cảnh báo] Chưa cài đặt 'webrtcvad'!")

        # 2. Cấu hình thiết bị Microphone đầu vào
        if input_device is not None:
            self.input_device = input_device
            self.input_device_name = f"Device {input_device}"
        else:
            cfg_idx, cfg_name = get_configured_microphone()
            self.input_device = cfg_idx
            self.input_device_name = cfg_name

        # 3. Khởi tạo Vosk Offline Model (LỚP 4: PHAO CỨU SINH KHI MẤT MẠNG)
        # BẬT LẠI GRAMMAR CỨNG: ["tèo đâu", "teo dau", "[unk]"] + FUZZY MATCH >= 70%
        vosk.SetLogLevel(-1)  # Tắt log Kaldi rườm rà
        self.model_path = get_vosk_model_path(vosk_model_path)
        model_name = Path(self.model_path).name
        print(f"[WakeWord] Đang nạp mô hình Vosk Offline từ '{model_name}' ({self.model_path})...")
        try:
            self.model = vosk.Model(self.model_path)
            grammar_list = self.grammar_words or list(self.DEFAULT_GRAMMAR_WORDS)
            grammar_json = json.dumps(grammar_list, ensure_ascii=False)
            try:
                self.recognizer = vosk.KaldiRecognizer(self.model, float(self.SAMPLE_RATE), grammar_json)
                try:
                    self.recognizer.SetWords(True)
                except Exception:
                    pass
                print(f"[WakeWord] ✓ Nạp Vosk thành công (Bật Grammar Lớp 4: {grammar_list}).")
            except Exception as gr_err:
                self.recognizer = vosk.KaldiRecognizer(self.model, float(self.SAMPLE_RATE))
                try:
                    self.recognizer.SetWords(True)
                except Exception:
                    pass
                print(f"[WakeWord Cảnh báo] Không thể nạp Grammar Vosk ({gr_err}), dùng recognizer mặc định.")
        except Exception as exc:
            print(f"[WakeWord Cảnh báo] Không thể nạp mô hình Vosk: {exc}")
            self.model = None
            self.recognizer = None

        self._running = False
        self._paused = False
        self._is_listening = False
        self.stream: Optional[sd.RawInputStream] = None
        self.last_audio_time: float = time.time()
        self._thread: Optional[threading.Thread] = None
        self.pre_buffer: collections.deque = collections.deque(maxlen=10)
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog: threading.Event = threading.Event()
        self._lock = threading.Lock()
        self._last_trigger_time = 0.0

        self._log_debug(
            f"[WakeWord Khởi tạo 4 Lớp] Từ khóa: '{self.wake_keyword}', "
            f"VAD mode={self.vad_mode}, RMS > {self.min_rms:.0f}, "
            f"Voice Auth thresh={self.voice_auth_threshold}, Cooldown={self.cooldown_sec}s."
        )

    @property
    def listen_thread(self) -> Optional[threading.Thread]:
        """Luồng đọc microphone của Wake Word."""
        return self._thread

    @listen_thread.setter
    def listen_thread(self, val: Optional[threading.Thread]):
        self._thread = val

    @property
    def is_listening(self):
        """Trạng thái lắng nghe của engine."""
        return BoolCallable(self._is_listening)

    @is_listening.setter
    def is_listening(self, val: bool):
        self._is_listening = bool(val)

    def is_stream_alive(self) -> bool:
        """Kiểm tra stream microphone đang hoạt động bình thường."""
        try:
            return (
                self.stream is not None
                and getattr(self.stream, "active", False)
                and not getattr(self.stream, "stopped", True)
            )
        except Exception:
            return False

    def _log_debug(self, message: str) -> None:
        """Ghi log có timestamp ra console và file teo_debug.log."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} {message}"
        print(log_line, flush=True)
        try:
            log_path = get_exe_dir() / "teo_debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            pass

    def _signal_early_wake(self) -> None:
        """[Đã gỡ bỏ Optimistic UI] Không còn phát tín hiệu mở Glow Orb sớm."""
        pass

    def _signal_early_cancel(self) -> None:
        """[Đã gỡ bỏ Optimistic UI] Không còn phát tín hiệu hủy sớm."""
        pass

    # =========================================================================
    # LỚP 2: VOICE AUTH (XÁC THỰC GIỌNG NÓI SINH TRẮC HỌC)
    # =========================================================================
    def _verify_voice_auth(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """
        LỚP 2: Xác thực sinh trắc học giọng nói ngoại tuyến (Offline 100%).
        - Nếu file my_voice.pkl chưa có -> Bỏ qua xác thực, trả về (True, 1.0).
        - Nếu file có nhưng sai giọng (cosine < 0.80) -> Trả về (False, similarity).
        - Nếu đúng giọng chủ nhân (cosine >= 0.80) -> Trả về (True, similarity).
        """
        if not audio_bytes:
            return False, 0.0

        if self.voice_verifier is not None:
            try:
                res = self.voice_verifier(audio_bytes)
                if isinstance(res, tuple):
                    return bool(res[0]), float(res[1])
                return bool(res), 1.0 if res else 0.0
            except Exception as exc:
                self._log_debug(f"[WakeWord 🛡️ VoiceAuth Cảnh báo] Lỗi voice_verifier: {exc}")
                return False, 0.0

        try:
            from voice_auth import verify_voice
            return verify_voice(
                audio_bytes,
                sample_rate=self.SAMPLE_RATE,
                threshold=self.voice_auth_threshold,
            )
        except Exception as exc:
            self._log_debug(f"[WakeWord 🛡️ VoiceAuth Cảnh báo]: {exc}")
            # Nếu module chưa sẵn sàng, bỏ qua để không chặn người dùng
            return True, 1.0

    # =========================================================================
    # LỚP 3: GOOGLE STT (TRỌNG TÀI CHÍNH - ONLINE)
    # =========================================================================
    def _preprocess_for_stt(self, audio_bytes: bytes) -> bytes:
        """
        Tiền xử lý âm thanh trước khi gửi lên Google STT:
        1. Khử tạp âm nền bằng noisereduce (reduce_noise).
           Đo thời gian thực thi: Nếu noisereduce mất hơn 0.2s, tự động bỏ qua bước lọc nhiễu
           cho Lớp 3 ở các lần sau (vì Google STT đã có khả năng tự lọc nhiễu tốt),
           chỉ giữ lại chuẩn hóa âm lượng (normalize).
        2. Chuẩn hóa âm lượng (Peak normalization về 0.90) để đảm bảo âm thanh có độ lớn đồng đều.
        """
        if not audio_bytes or len(audio_bytes) < 1600:
            return audio_bytes

        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Nếu âm thanh là khoảng lặng hoàn toàn, bỏ qua để tránh chia cho 0 trong STFT
            if float(np.max(np.abs(samples))) <= 1e-4:
                return audio_bytes

            # 1. Lọc bớt nhiễu nền môi trường (nếu chưa bị tắt do quá chậm)
            if not getattr(self, "_skip_noisereduce_stt", False):
                if HAS_NOISEREDUCE and nr is not None:
                    t_start = time.time()
                    try:
                        samples = nr.reduce_noise(y=samples, sr=self.SAMPLE_RATE)
                        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
                        elapsed = time.time() - t_start
                        if elapsed > 0.20:
                            self._skip_noisereduce_stt = True
                            self._log_debug(
                                f"[WakeWord ⚡ Tăng tốc] noisereduce mất {elapsed:.3f}s (> 0.2s) -> "
                                f"Tự động bỏ qua lọc nhiễu cho Lớp 3 ở các lần sau (chỉ giữ chuẩn hóa âm lượng normalize)."
                            )
                    except Exception:
                        pass

            # 2. Chuẩn hóa âm lượng (Peak normalization về 0.90)
            max_val = float(np.max(np.abs(samples)))
            if max_val > 1e-4:
                samples = samples * (0.90 / max_val)

            samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
            return samples_int16.tobytes()
        except Exception as exc:
            self._log_debug(f"[WakeWord Cảnh báo] Lỗi tiền xử lý âm thanh STT: {exc}")
            return audio_bytes

    def _recognize_google(self, audio_bytes: bytes, timeout: float = 2.5) -> Tuple[str, bool, str]:
        """
        Gửi đoạn âm thanh lên Google STT (language='vi-VN', timeout cứng 2.5s).
        Ép timeout cứng bằng ThreadPoolExecutor: Nếu quá 2.5s không phản hồi,
        lập tức hủy luồng đó, ghi log '[Google STT] Timeout 2.5s -> Hủy và quay lại Lớp 1',
        trả về is_network_error=True để chuyển ngay sang Lớp 4 (Vosk) hoặc bỏ qua,
        và TUYỆT ĐỐI KHÔNG RETRY nếu bị timeout.

        Returns:
            (recognized_text: str, is_network_error: bool, error_message: str)
            - Nếu nhận diện thành công: (text, False, "")
            - Nếu lỗi mạng / timeout / connection error: ("", True, err_msg)
            - Nếu không nhận diện được chữ (UnknownValueError): ("", False, "")
        """
        if not audio_bytes:
            return "", False, "Dữ liệu âm thanh rỗng"

        # Tiền xử lý âm thanh: lọc nhiễu (nếu < 0.2s) + chuẩn hóa âm lượng (normalize)
        prep_audio = self._preprocess_for_stt(audio_bytes)

        def _do_raw_call(raw_bytes: bytes, req_timeout: float) -> Tuple[str, bool, str]:
            # 1. Thử dùng stt_verifier được truyền từ app_tray (SpeechToText.recognize_google)
            if self.stt_verifier is not None:
                try:
                    import inspect
                    sig = inspect.signature(self.stt_verifier)
                    if "timeout" in sig.parameters:
                        res = self.stt_verifier(raw_bytes, timeout=req_timeout)
                    else:
                        res = self.stt_verifier(raw_bytes)
                    return str(res or "").strip(), False, ""
                except (TimeoutError, concurrent.futures.TimeoutError) as te:
                    return "", True, f"Google STT Timeout {req_timeout:.1f}s: {te}"
                except (ConnectionError, OSError) as ce:
                    return "", True, f"Google STT mất kết nối mạng: {ce}"
                except Exception as exc:
                    err_str = str(exc).lower()
                    if any(k in err_str for k in ["timeout", "connection", "network", "getaddrinfo", "offline", "request"]):
                        return "", True, f"Lỗi mạng Google STT: {exc}"
                    return "", False, str(exc)

            # 2. Dự phòng gọi trực tiếp qua SpeechRecognition
            if sr is None:
                return "", True, "Thư viện speech_recognition chưa được nạp"

            try:
                recognizer = sr.Recognizer()
                audio_data = sr.AudioData(raw_bytes, self.SAMPLE_RATE, 2)
                res_text = recognizer.recognize_google(audio_data, language="vi-VN")
                return (res_text.strip() if res_text else ""), False, ""
            except sr.RequestError as req_err:
                return "", True, f"Google STT lỗi kết nối mạng: {req_err}"
            except sr.UnknownValueError:
                return "", False, "Không phát hiện giọng nói rõ ràng"
            except Exception as exc:
                err_str = str(exc).lower()
                if any(k in err_str for k in ["timeout", "connection", "network", "socket", "offline"]):
                    return "", True, f"Lỗi mạng: {exc}"
                return "", False, str(exc)

        # 1. BỌC HÀM GỌI GOOGLE STT VÀO ThreadPoolExecutor VỚI TIMEOUT CỨNG 1.5S
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_raw_call, prep_audio, timeout)
        try:
            text, is_net_err, err_msg = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._log_debug(f"[Google STT] Timeout {timeout:.1f}s -> Hủy và quay lại Lớp 1")
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            return "", True, f"Google STT Timeout {timeout:.1f}s"
        except Exception as ex:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
            return "", True, f"Lỗi thực thi Google STT: {ex}"
        finally:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass

        # NẾU LẦN 1 BỊ TIMEOUT HOẶC LỖI MẠNG: TUYỆT ĐỐI KHÔNG RETRY (chống treo thêm)
        if is_net_err:
            if "timeout" in err_msg.lower():
                self._log_debug(f"[Google STT] Timeout {timeout:.1f}s -> Hủy và quay lại Lớp 1")
            return "", True, err_msg

        # NẾU KẾT QUẢ TRẢ VỀ RỖNG "" (VÀ KHÔNG PHẢI DO TIMEOUT/LỖI MẠNG), THỬ LẠI TỐI ĐA 1 LẦN VỚI TIMEOUT 1.5S
        if not text:
            self._log_debug(f"[Google STT] Kết quả trả về rỗng -> Thử lại lần 2 (timeout cứng {timeout:.1f}s)...")
            exec_retry = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future_retry = exec_retry.submit(_do_raw_call, prep_audio, timeout)
            try:
                text_retry, is_net_err_retry, err_msg_retry = future_retry.result(timeout=timeout)
                if text_retry:
                    self._log_debug(f"[Google STT] ✓ Retry lần 2 thành công! Kết quả: \"{text_retry}\"")
                    return text_retry, False, ""
                if is_net_err_retry:
                    return "", True, err_msg_retry
                return "", False, err_msg_retry or "Không phát hiện giọng nói sau retry"
            except concurrent.futures.TimeoutError:
                self._log_debug(f"[Google STT] Timeout {timeout:.1f}s -> Hủy và quay lại Lớp 1")
                return "", True, f"Google STT Timeout {timeout:.1f}s"
            finally:
                try:
                    exec_retry.shutdown(wait=False)
                except Exception:
                    pass

        return text, is_net_err, err_msg

    @classmethod
    def is_blacklisted(cls, text: str) -> bool:
        """
        Kiểm tra blacklist chặn các từ nhầm lẫn:
        - Các cụm từ: 'tèo đâu rồi', 'mày đi đâu', 'ở đâu', 'đi đâu', 'sao đâu'...
        - Các từ đơn gây nhầm: 'mèo', 'heo', 'bèo', 'kèo', 'nèo' (và không dấu).
        """
        if not text:
            return False
        raw_lower = text.lower().strip()
        clean = strip_accents(raw_lower).lower()
        clean = re.sub(r"[^\w\s]", " ", clean).strip()
        clean = re.sub(r"\s+", " ", clean)

        for phrase in cls.BLACKLIST_PHRASES:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, raw_lower) or re.search(pattern, clean):
                return True

        raw_words = re.findall(r"\w+", raw_lower)
        clean_words = clean.split()
        all_words = set(raw_words + clean_words)
        for w in cls.BLACKLIST_WORDS:
            if w in all_words:
                return True

        return False

    @classmethod
    def _normalize_syllable(cls, s: str) -> str:
        s = s.lower().replace("đ", "d").replace("Đ", "D")
        nfkd = unicodedata.normalize("NFKD", s)
        s = "".join([c for c in nfkd if not unicodedata.combining(c)]).strip()
        return re.sub(r"[^\w]", "", s)

    @classmethod
    def _is_google_text_matched(cls, google_text: str, threshold: float = 80.0) -> bool:
        """
        Kiểm tra matching theo âm tiết (thay thế hoàn toàn substring thô):
        - Tách text gốc thành các âm tiết theo khoảng trắng.
        - Chuẩn hóa từng âm tiết: lower + bỏ dấu tiếng Việt bằng unicodedata.
        - Dùng rapidfuzz.ratio để so sánh.
        - ACCEPT nếu thỏa CẢ 3 điều kiện:
          + Tổng số âm tiết <= 3 và >= 2.
          + Âm tiết cuối có rapidfuzz.ratio với "dau" >= 80.
          + Có ít nhất 1 âm tiết trong các âm tiết còn lại có rapidfuzz.ratio với "teo" >= 80.
        - REJECT mọi trường hợp còn lại.
        """
        if not google_text or not google_text.strip():
            return False

        raw_syllables = google_text.strip().split()
        syllables = [cls._normalize_syllable(s) for s in raw_syllables if cls._normalize_syllable(s)]
        if not syllables:
            return False

        last_syl = syllables[-1]
        last_ratio = float(fuzz.ratio(last_syl, "dau"))

        other_syls = syllables[:-1]
        if other_syls:
            teo_ratios = [float(fuzz.ratio(s, "teo")) for s in other_syls]
            best_teo_ratio = max(teo_ratios)
        else:
            best_teo_ratio = 0.0

        is_match = (
            2 <= len(syllables) <= 3
            and last_ratio >= 80.0
            and best_teo_ratio >= 80.0
        )

        status = "ACCEPT" if is_match else "REJECT"
        log_line = f"[WakeWord][Match] {status} syllables={syllables}, last_ratio={last_ratio:.1f}, first_ratio={best_teo_ratio:.1f}"
        print(log_line, flush=True)
        try:
            log_path = get_exe_dir() / "teo_debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + log_line + "\n")
        except Exception:
            pass

        return is_match

    # Alias để tương thích ngược với các test suite
    _is_google_stt_matched = _is_google_text_matched

    def _handle_cross_verification(
        self,
        candidate_audio: bytes,
        phrase: str = "tèo đâu",
        vosk_conf: float = 0.90,
    ) -> bool:
        """
        Hàm xác thực chéo hỗ trợ tương thích ngược cho test suite:
        1. Xác thực giọng nói (Voice Auth).
        2. Nếu Voice Auth từ chối nhưng Google STT khớp 'teo dau' và RMS > 500 -> Xác thực kép thành công.
        3. Gửi Google STT xác thực chéo trực tuyến.
        4. Nếu mất mạng nhưng đã đúng giọng chủ nhân -> Cho phép kích hoạt (Lớp 4).
        """
        if not candidate_audio:
            return False

        chunk_samples = np.frombuffer(candidate_audio, dtype=np.int16).astype(np.float32)
        chunk_rms = float(np.sqrt(np.mean(chunk_samples ** 2))) if len(chunk_samples) > 0 else 0.0
        active_samples = []
        for i in range(0, len(chunk_samples), self.BLOCK_SIZE):
            blk = chunk_samples[i : i + self.BLOCK_SIZE]
            if len(blk) > 0:
                b_rms = float(np.sqrt(np.mean(blk ** 2)))
                if b_rms >= self.min_rms:
                    active_samples.extend(blk)
        active_rms = float(np.sqrt(np.mean(np.array(active_samples) ** 2))) if len(active_samples) > 0 else chunk_rms
        effective_rms = max(chunk_rms, active_rms)

        is_owner, sim_score = self._verify_voice_auth(candidate_audio)
        if not is_owner:
            if effective_rms <= 500.0:
                return False
            google_text, is_net_err, _ = self._recognize_google(candidate_audio, timeout=2.5)
            if not is_net_err and self._is_google_text_matched(google_text):
                self._log_debug(
                    f"[WakeWord][DualAuth] Voice Auth reject (cosine={sim_score:.2f}) "
                    f"nhưng Google STT xác nhận '{google_text}' + RMS={effective_rms:.0f} -> KÍCH HOẠT."
                )
                self._trigger_wake(f"DualAuth: Google STT ('{google_text}', RMS={effective_rms:.0f})")
                return True
            return False

        # Không phát tín hiệu hiển thị sớm (Đã gỡ bỏ Optimistic UI)
        google_text, is_net_err, _ = self._recognize_google(candidate_audio, timeout=2.5)
        if not is_net_err and self._is_google_text_matched(google_text):
            self._trigger_wake(f"Google STT ('{google_text}')")
            return True
        elif is_net_err or not google_text:
            # Mất mạng hoặc rỗng khi đã qua Voice Auth -> Chuyển sang Lớp 4 (Vosk Offline)
            if self.model is not None:
                vosk_matched = self._handle_layer4_vosk(candidate_audio)
                if vosk_matched:
                    return True
                return False
            else:
                self._trigger_wake(f"VoiceAuth ({sim_score:.2f}) [Mất mạng/Vosk Offline]")
                return True
        else:
            # Google STT từ chối từ khóa
            return False

    # =========================================================================
    # LỚP 4: VOSK OFFLINE (PHAO CỨU SINH - CHỈ KÍCH HOẠT KHI MẤT MẠNG)
    # =========================================================================
    def _check_wake_condition(
        self,
        raw_text: str,
        conf: float = 1.0,
        source: str = "Offline",
    ) -> Tuple[bool, str]:
        """
        Kiểm tra điều kiện khớp từ khóa cho Vosk (hỗ trợ tương thích ngược và kiểm thử):
        - Yêu cầu conf >= min_conf (0.60).
        - Giới hạn câu ngắn <= 3 từ.
        - Khớp chính xác hoặc Fuzzy Match >= fuzzy_threshold (75.0%).
        """
        if not raw_text:
            return False, ""

        clean_text = strip_accents(raw_text).lower()
        clean_text = re.sub(r"[^\w\s]", " ", clean_text).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
        words = clean_text.split()

        if not words or len(words) > 3:
            return False, ""

        if conf < self.min_conf:
            return False, ""

        if clean_text == self.target_keyword:
            return True, f"{raw_text} (teo dau, conf={conf:.2f})"

        ratio_score = fuzz.ratio(clean_text, self.target_keyword)
        partial_score = fuzz.partial_ratio(clean_text, self.target_keyword)
        fuzzy_score = float(max(ratio_score, partial_score))

        if fuzzy_score >= self.fuzzy_threshold:
            return True, f"{raw_text} (Fuzzy {fuzzy_score:.0f}% '{self.target_keyword}', conf={conf:.2f})"

        return False, ""

    def _check_text_for_wake_word(self, raw_text: str) -> bool:
        """Kiểm tra nhanh văn bản với conf=1.0 (hỗ trợ tương thích ngược)."""
        cond_matched, _ = self._check_wake_condition(raw_text, conf=1.0)
        return cond_matched

    @staticmethod
    def _check_vosk_match(clean_text: str) -> Tuple[bool, str]:
        """
        Kiểm tra chuỗi nhận diện từ Vosk Offline (đã bỏ dấu) bằng Regex và Fuzzy Match.
        Bảo đảm hỗ trợ các biến thể âm thanh: 'theo dau', 'theo do', 'teo do', 'teo dau'.
        Loại trừ từ gây nhiễu: 'the dau', 'di dau', 'o dau'.
        """
        if not clean_text:
            return False, "Chuỗi rỗng"

        # 1. Khớp chính xác cụm từ 'teo dau'
        if "teo dau" in clean_text:
            return True, "Khớp chính xác 'teo dau'"
        if re.search(r"\bteo\s*dau\b", clean_text):
            return True, "Regex khớp 'teo dau'"

        # 2. Fuzzy match cặp từ đánh thức (hạn chế cho câu ngắn <= 4 từ)
        words = clean_text.split()
        if len(words) <= 4:
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                # Chặn các từ phổ biến không phải tèo: 'the' (thế), 'di' (đi), 'o' (ở), 'keu' (kêu)
                if w1 in ["the", "di", "o", "keu", "ai", "dau"]:
                    continue
                w1_score = fuzz.ratio(w1, "teo")
                w2_score = fuzz.ratio(w2, "dau")
                w1_ok = (w1 in ["teo", "theo"]) or (w1_score >= 80)
                w2_ok = (w2 in ["dau", "do"]) or (w2_score >= 80)
                if w1_ok and w2_ok:
                    return True, f"Fuzzy khớp cặp từ ({w1}, {w2})"

        return False, "Không khớp 'teo dau'"

    def _handle_layer4_vosk(self, audio_bytes: bytes) -> bool:
        """
        Xử lý Lớp 4 (Phao cứu sinh Vosk Offline):
        - Kích hoạt khi Google STT gặp sự cố mạng / timeout.
        - Khởi tạo KaldiRecognizer với Grammar cứng: ["tèo đâu", "teo dau", "[unk]"] (chỉ áp dụng cho Lớp 4).
        - Sử dụng RapidFuzz so sánh chuỗi nhận diện với 'teo dau'.
        - Nếu độ khớp >= 70% -> LẬP TỨC KÍCH HOẠT (không cần quan tâm conf vì đã qua Voice Auth).
        """
        if not self.model:
            self._log_debug("[WakeWord 🛟 Lớp 4 Lỗi] Mô hình Vosk chưa sẵn sàng!")
            return False

        try:
            t0 = time.time()
            grammar_list = self.grammar_words or list(self.DEFAULT_GRAMMAR_WORDS)
            grammar_json = json.dumps(grammar_list, ensure_ascii=False)

            # Sử dụng recognizer đã nạp sẵn với Grammar Lớp 4 hoặc khởi tạo mới
            if self.recognizer is not None:
                rec = self.recognizer
                rec.Reset()
            else:
                rec = vosk.KaldiRecognizer(self.model, float(self.SAMPLE_RATE), grammar_json)

            rec.AcceptWaveform(audio_bytes)
            res_str = rec.FinalResult()
            elapsed = round(time.time() - t0, 3)

            res = json.loads(res_str) if res_str else {}
            vosk_text = res.get("text", "").strip()
            word_items = res.get("result", [])
            vosk_conf = (
                sum(float(w.get("conf", 1.0)) for w in word_items) / len(word_items)
                if word_items
                else 1.0
            )
            self._log_debug(f"[WakeWord 🛟 Lớp 4 - Vosk Offline ({elapsed}s)] Kết quả: '{vosk_text}' (conf={vosk_conf:.2f})")

            if not vosk_text:
                self._log_debug("[WakeWord 🛟 Lớp 4 ✗ BỎ QUA] Vosk không nhận diện được từ nào -> Bỏ qua, quay lại Lớp 1.")
                return False

            if word_items and vosk_conf < self.min_conf:
                self._log_debug(
                    f"[WakeWord 🛟 Lớp 4 ✗ BỎ QUA] Vosk nhận diện: '{vosk_text}' nhưng độ tin cậy quá thấp "
                    f"({vosk_conf:.2f} < {self.min_conf:.2f}) -> Bỏ qua tạp âm, quay lại Lớp 1."
                )
                return False

            clean_text = strip_accents(vosk_text).lower()
            clean_text = re.sub(r"[^\w\s]", " ", clean_text).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)

            # Bỏ qua từ đặc biệt [unk] hoặc rỗng
            if not clean_text or clean_text in ["unk", "[unk]"]:
                self._log_debug("[WakeWord 🛟 Lớp 4 ✗ BỎ QUA] Vosk trả về chuỗi rỗng hoặc '[unk]' -> Bỏ qua, quay lại Lớp 1.")
                return False

            words = clean_text.split()
            # Chặn các từ phổ biến không phải 'tèo' ở vị trí từ đầu tiên: 'thế', 'đi', 'ở', 'kêu', 'ai'
            if words and words[0] in ["the", "di", "o", "keu", "ai"]:
                self._log_debug(f"[WakeWord 🛟 Lớp 4 ✗ BỎ QUA] Từ đầu '{words[0]}' thuộc danh sách chặn -> Bỏ qua, quay lại Lớp 1.")
                return False

            # Dùng RapidFuzz so sánh trực tiếp với 'teo dau'
            ratio_score = fuzz.ratio(clean_text, self.target_keyword)
            partial_score = fuzz.partial_ratio(clean_text, self.target_keyword)
            if len(words) >= 2:
                fuzzy_score = float(max(ratio_score, partial_score))
            else:
                fuzzy_score = float(ratio_score)

            self._log_debug(
                f"[WakeWord 🛟 Lớp 4] So sánh RapidFuzz '{clean_text}' với '{self.target_keyword}': "
                f"{fuzzy_score:.1f}% (ngưỡng >= {self.fuzzy_threshold:.1f}%)"
            )

            # Nếu độ khớp >= 70% hoặc chứa chính xác 'teo dau' -> LẬP TỨC KÍCH HOẠT
            if fuzzy_score >= self.fuzzy_threshold or "teo dau" in clean_text:
                self._log_debug(
                    f"[WakeWord 🛟 Lớp 4 ✓ PHAO CỨU SINH KÍCH HOẠT] "
                    f"Vosk Offline nhận diện: '{vosk_text}' (Fuzzy {fuzzy_score:.1f}% >= {self.fuzzy_threshold:.1f}%) "
                    f"-> KÍCH HOẠT WAKE WORD (Glow Orb & Bíp)!"
                )
                self._trigger_wake(f"Vosk Offline ('{vosk_text}', khớp {fuzzy_score:.1f}%)")
                return True
            else:
                self._log_debug(
                    f"[WakeWord 🛟 Lớp 4 ✗ BỎ QUA] Vosk Offline nhận diện: '{vosk_text}' "
                    f"(Fuzzy {fuzzy_score:.1f}% < {self.fuzzy_threshold:.1f}%) -> Bỏ qua, quay lại Lớp 1."
                )
                return False
        except Exception as exc:
            self._log_debug(f"[WakeWord 🛟 Lớp 4 Lỗi nhận diện Vosk]: {exc}")
            return False

    # =========================================================================
    # ĐIỀU KHIỂN LUỒNG & MICROPHONE
    # =========================================================================
    def _trigger_wake(self, detected_phrase: str) -> None:
        """Kích hoạt sự kiện phát hiện từ khóa đánh thức."""
        self._last_trigger_time = time.time()
        self._log_debug(
            f"\n=======================================================\n"
            f"[WakeWord 🔔 KÍCH HOẠT THÀNH CÔNG] '{detected_phrase}'\n"
            f"-> Kích hoạt Tèo Assistant (Glow Orb & Bíp)!\n"
            f"======================================================="
        )
        self.pause()
        if self.on_detected:
            try:
                self.on_detected("teo", 1.0)
            except Exception as cb_err:
                self._log_debug(f"[WakeWord Cảnh báo] Lỗi callback on_detected: {cb_err}")

    def set_input_device(self, device_index: Optional[int], device_name: str = "") -> None:
        """Cập nhật thiết bị Microphone đầu vào và khởi động lại luồng lắng nghe an toàn."""
        with self._lock:
            if self.input_device == device_index:
                return
            self.input_device = device_index
            self.input_device_name = device_name or (
                f"Device {device_index}" if device_index is not None else "Mặc định hệ thống"
            )
            was_running = self._running and bool(self.is_listening)

        self._log_debug(
            f"[WakeWord] 🎙️ Đã chuyển Microphone sang: {self.input_device_name} (device={self.input_device})"
        )
        if was_running:
            self.pause()
            time.sleep(0.15)
            self.resume()

    def start(self) -> None:
        """Bắt đầu luồng lắng nghe Wake Word."""
        with self._lock:
            if self._running and self.is_stream_alive():
                return
            self._running = True
            self._paused = False
            self._stop_watchdog.clear()

            if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
                self._watchdog_thread = threading.Thread(
                    target=self._watchdog_loop,
                    name="AudioWatchdog",
                    daemon=True,
                )
                self._watchdog_thread.start()

            self._log_debug(f"[WakeWord] 🎙️ Bắt đầu lắng nghe từ khóa '{self.wake_keyword}' (Kiến trúc 4 Lớp)...")

        self.resume()

    def stop(self) -> None:
        """Dừng hoàn toàn luồng lắng nghe Wake Word."""
        self._stop_watchdog.set()
        self.pause()
        with self._lock:
            self._running = False
        self._log_debug("[WakeWord] ⏹️ Đã dừng luồng lắng nghe Wake Word.")

    def pause(self) -> None:
        """Tạm dừng lắng nghe và giải phóng micro để STT thu âm lệnh."""
        with self._lock:
            self.is_listening = False
            self._paused = True
            if self.stream is not None:
                try:
                    self.stream.abort()
                except Exception:
                    pass

        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=1.0)

        with self._lock:
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

        stream_alive = self.is_stream_alive()
        self._log_debug(f"[WakeWord][State] listening={bool(self.is_listening)}, stream_alive={stream_alive}")

    def resume(self) -> None:
        """
        Kích hoạt lại việc lắng nghe Wake Word sau khi thực thi lệnh xong.
        Đảm bảo stream đã sẵn sàng:
        - Nếu stream đã active và thread đang chạy thì chỉ cần đảm bảo listening=True.
        - Nếu chưa, mở stream mới (retry 3 lần) và khởi động listen_thread.
        """
        if (
            self.is_stream_alive()
            and bool(self.is_listening)
            and self.listen_thread is not None
            and self.listen_thread.is_alive()
        ):
            with self._lock:
                self._paused = False
                self._last_trigger_time = time.time()
                self.last_audio_time = time.time()
            self._log_debug("[WakeWord][Stream] resume OK: active=True, stopped=False, listening=True")
            return

        # 1. Đảm bảo đóng dọn dẹp stream cũ nếu còn
        with self._lock:
            if self.is_stream_alive():
                try:
                    if self.stream is not None:
                        self.stream.abort()
                        self.stream.close()
                except Exception:
                    pass
                self.stream = None

        # 2. Thử mở stream trước, retry tối đa 3 lần
        success = False
        for attempt in range(1, 4):
            with MIC_LOCK:
                try:
                    if self.stream is not None:
                        try:
                            self.stream.abort()
                            self.stream.close()
                        except Exception:
                            pass
                        self.stream = None

                    self.stream = sd.RawInputStream(
                        samplerate=self.SAMPLE_RATE,
                        blocksize=self.BLOCK_SIZE,
                        dtype="int16",
                        channels=1,
                        device=self.input_device,
                    )
                    self.stream.start()
                except Exception as exc:
                    self.stream = None

            if (
                self.stream is not None
                and getattr(self.stream, "active", False)
                and not getattr(self.stream, "stopped", True)
            ):
                success = True
                break

            if attempt < 3:
                time.sleep(0.1)

        # 3. Chỉ khi stream active mới set listening=True
        if success:
            with self._lock:
                self.is_listening = True
                self.last_audio_time = time.time()
                self._paused = False
                self._last_trigger_time = time.time()

                if self._running and (self.listen_thread is None or not self.listen_thread.is_alive()):
                    self.listen_thread = threading.Thread(
                        target=self._listen_loop,
                        name="WakeWord4LayerThread",
                        daemon=True,
                    )
                    self.listen_thread.start()

            self._log_debug("[WakeWord][Stream] resume OK: active=True, stopped=False, listening=True")
        else:
            with self._lock:
                self.is_listening = False
            active = getattr(self.stream, "active", False) if self.stream else False
            stopped = getattr(self.stream, "stopped", True) if self.stream else True
            self._log_debug(f"[WakeWord][Stream] resume FAILED sau 3 lần, active={active}, stopped={stopped}")

    def force_reset_stream(self) -> None:
        """
        Cưỡng chế reset stream sạch:
        - Set is_listening = False.
        - Nếu self.listen_thread và self.listen_thread.is_alive(): join(timeout=1.0).
          Nếu sau 1s vẫn alive, log cảnh báo và bỏ tham chiếu (thread cũ sẽ tự chết khi thấy is_listening=False ở vòng lặp kế).
        - Close stream cũ nếu còn.
        - Set self.stream = None.
        - Clear pre-buffer deque.
        - Set last_audio_time = time.time().
        - Mở stream mới.
        - Set is_listening = True.
        - Khởi động listen_thread mới.
        - Log [WakeWord][Stream] force_reset OK: active=True, thread_alive=True.
        """
        # 1. Set is_listening = False
        with self._lock:
            self.is_listening = False
            self._paused = True
            if self.stream is not None:
                try:
                    self.stream.abort()
                except Exception:
                    pass

        # 2. Nếu self.listen_thread và self.listen_thread.is_alive(): join(timeout=1.0)
        old_thread = self.listen_thread
        if old_thread and old_thread.is_alive() and threading.current_thread() != old_thread:
            old_thread.join(timeout=1.0)
            if old_thread.is_alive():
                self._log_debug("[WakeWord][Stream] Cảnh báo: listen_thread cũ vẫn alive sau 1s, bỏ tham chiếu.")
        self.listen_thread = None

        # 3. Close stream cũ nếu còn & Set self.stream = None
        with MIC_LOCK:
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

        # 4. Clear pre-buffer deque
        if hasattr(self, "pre_buffer") and self.pre_buffer is not None:
            self.pre_buffer.clear()

        # 5. Set last_audio_time = time.time()
        self.last_audio_time = time.time()

        # 6. Mở stream mới
        with MIC_LOCK:
            try:
                new_stream = sd.RawInputStream(
                    samplerate=self.SAMPLE_RATE,
                    blocksize=self.BLOCK_SIZE,
                    dtype="int16",
                    channels=1,
                    device=self.input_device,
                )
                new_stream.start()
                self.stream = new_stream
            except Exception as exc:
                self._log_debug(f"[WakeWord Lỗi force_reset_stream]: {exc}")
                self.stream = None

        # 7. Set is_listening = True
        with self._lock:
            self.is_listening = True
            self._paused = False
            self._last_trigger_time = time.time()

            # 8. Khởi động listen_thread mới
            if self._running:
                self.listen_thread = threading.Thread(
                    target=self._listen_loop,
                    name="WakeWord4LayerThread",
                    daemon=True,
                )
                self.listen_thread.start()

        # 9. Log [WakeWord][Stream] force_reset OK: active=True, thread_alive=True
        time.sleep(0.05)
        active = self.is_stream_alive()
        thread_alive = self.listen_thread.is_alive() if self.listen_thread else False
        self._log_debug(f"[WakeWord][Stream] force_reset OK: active={active}, thread_alive={thread_alive}")

    def _reopen_stream(self) -> None:
        """Khởi động lại stream microphone khi bị đơ/chết ngầm."""
        self._log_debug("[WakeWord] 🔄 _reopen_stream đang khởi động lại stream...")
        self.force_reset_stream()

    def _watchdog_loop(self) -> None:
        """Thread AudioWatchdog, chu kỳ 0.5s, chạy trong lúc is_listening=True."""
        while not self._stop_watchdog.wait(0.5):
            if not self._running:
                break
            if bool(self.is_listening):
                now = time.time()
                last_chunk_age = now - self.last_audio_time
                if last_chunk_age > 3.0:
                    self._log_debug(
                        f"[Watchdog][PANIC] Stream treo: {last_chunk_age:.1f}s không có audio chunk, tiến hành reset."
                    )
                    self.force_reset_stream()

    def reset_microphone(self) -> None:
        """Reset lại toàn bộ luồng Micro."""
        self._log_debug("[WakeWord] 🔄 Đang reset lại luồng Microphone...")
        self.force_reset_stream()

    # =========================================================================
    # VÒNG LẶP LẮNG NGHE CHÍNH (KIẾN TRÚC 4 LỚP)
    # =========================================================================
    def _listen_loop(self) -> None:
        """
        Vòng lặp thu âm micro và xử lý 4 Lớp:
        - Lớp 1: WebRTC VAD + RMS > 350 phát hiện tiếng người -> Thu âm 2.5s (bắt trọn âm cuối).
        - Lớp 2: Voice Auth (Resemblyzer Cosine >= 0.80 + noisereduce). Cơ chế xác thực kép nếu < 0.80 & RMS > 500.
        - Lớp 3: Google STT (Trọng tài chính Online, tiền xử lý khử nhiễu + normalize, retry 1.5s, rapidfuzz >= 80%).
        - Lớp 4: Vosk Offline (Phao cứu sinh khi mất mạng, Grammar ['tèo đâu', 'teo dau', '[unk]'] + Fuzzy >= 70%).
        """
        if not self._running or not bool(self.is_listening) or not self.is_stream_alive():
            return

        # Buffer trượt lưu 300ms âm thanh trước thời điểm phát hiện tiếng nói
        # (10 blocks x 30ms = 300ms) để không bị nuốt mất phụ âm đầu "Tèo..."
        self.pre_buffer.clear()
        pre_buffer = self.pre_buffer
        # Tăng thời gian thu âm lên 2.5s (83 blocks x 30ms = 2.49s ~ 2.5s)
        target_chunk_blocks = int(self.RECORD_DURATION_SEC / (self.FRAME_MS / 1000.0))

        self._log_debug(
            f"[WakeWord] 🎙️ Đang liên tục lắng nghe qua Micro ({self.input_device_name}) "
            f"[VAD mode={self.vad_mode}, RMS > {self.min_rms:.0f}]..."
        )

        try:
            while self._running and bool(self.is_listening) and self.is_stream_alive():
                try:
                    data, overflowed = self.stream.read(self.BLOCK_SIZE)
                except Exception as read_err:
                    if self._running and bool(self.is_listening):
                        self._log_debug(f"[WakeWord Lỗi Stream Read]: {read_err}")
                    break

                if not data:
                    continue

                # Cập nhật mốc thời gian nhận audio mới nhất mỗi chunk
                self.last_audio_time = time.time()

                # 1. Kiểm tra Cooldown sau lần kích hoạt gần nhất
                now = time.time()
                if (now - self._last_trigger_time) < self.cooldown_sec:
                    pre_buffer.append(bytes(data))
                    continue

                # 2. Tính năng lượng âm lượng RMS của block 30ms
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

                # 3. LỚP 1: Kiểm tra xem có tiếng người hay không (VAD + RMS > 350)
                is_speech_vad = False
                if self.vad is not None:
                    try:
                        is_speech_vad = self.vad.is_speech(bytes(data), self.SAMPLE_RATE)
                    except Exception:
                        is_speech_vad = False
                else:
                    is_speech_vad = True

                is_human_speech = is_speech_vad and (rms >= self.min_rms)

                if not is_human_speech:
                    # Là khoảng lặng hoặc tiếng ồn nhỏ -> Lưu vào pre_buffer và tiếp tục nghe
                    pre_buffer.append(bytes(data))
                    continue

                # === PHÁT HIỆN TIẾNG NGƯỜI NÓI (LỚP 1 TRIGGER) ===
                self._log_debug(
                    f"\n[WakeWord 🎙️ LỚP 1] Pre-roll 300ms + ghi động (RMS: {rms:.0f} >= {self.min_rms:.0f}, "
                    f"dừng khi im lặng {self.silence_limit_sec:.2f}s, tối đa {self.RECORD_DURATION_SEC:.1f}s)..."
                )

                # Thu âm động: gồm 300ms trước đó + block hiện tại + các block tiếp theo
                recorded_frames = list(pre_buffer)
                recorded_frames.append(bytes(data))
                consecutive_silence_blocks = 0
                silence_target_blocks = int(self.silence_limit_sec / (self.FRAME_MS / 1000.0))  # 0.6s im lặng = 20 blocks
                min_record_blocks = int(self.MIN_RECORD_SEC / (self.FRAME_MS / 1000.0))       # Thu tối thiểu 0.8s = ~26 blocks

                while self._running and bool(self.is_listening) and self.is_stream_alive() and len(recorded_frames) < target_chunk_blocks:
                    try:
                        chunk, ov = self.stream.read(self.BLOCK_SIZE)
                    except Exception:
                        break
                    if not chunk:
                        continue
                    self.last_audio_time = time.time()
                    chunk_bytes = bytes(chunk)
                    recorded_frames.append(chunk_bytes)

                    # Kiểm tra xem block 30ms này có tiếng người không
                    c_samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
                    c_rms = float(np.sqrt(np.mean(c_samples ** 2))) if len(c_samples) > 0 else 0.0

                    is_chunk_speech = False
                    if self.vad is not None:
                        try:
                            is_chunk_speech = self.vad.is_speech(chunk_bytes, self.SAMPLE_RATE)
                        except Exception:
                            is_chunk_speech = False
                        # Kết hợp cả điều kiện VAD và RMS (ngưỡng duy trì 65% min_rms để không bị nuốt âm đuôi khi nói nhỏ/nhanh)
                        is_chunk_speech = is_chunk_speech and (c_rms >= (self.min_rms * 0.65))
                    else:
                        is_chunk_speech = (c_rms >= (self.min_rms * 0.65))

                    if not is_chunk_speech:
                        consecutive_silence_blocks += 1
                        # Nếu im lặng kéo dài 0.6s VÀ đã thu tối thiểu 0.8s -> Dừng thu âm sớm!
                        if len(recorded_frames) >= min_record_blocks and consecutive_silence_blocks >= silence_target_blocks:
                            dur_so_far = len(recorded_frames) * (self.FRAME_MS / 1000.0)
                            self._log_debug(
                                f"[WakeWord 🎙️ LỚP 1 - Thu âm động] Phát hiện im lặng "
                                f"{consecutive_silence_blocks * self.FRAME_MS / 1000.0:.2f}s sau tiếng nói "
                                f"-> Dừng thu âm sớm ở {dur_so_far:.2f}s (thay vì chờ đủ {self.RECORD_DURATION_SEC:.1f}s)!"
                            )
                            break
                    else:
                        consecutive_silence_blocks = 0

                if not self._running or not bool(self.is_listening):
                    break

                audio_chunk = b"".join(recorded_frames)
                pre_buffer.clear()

                # Kiểm tra độ dài âm thanh tối thiểu (tối thiểu 0.8s)
                if len(audio_chunk) < int(self.SAMPLE_RATE * 2 * self.MIN_RECORD_SEC):
                    continue

                duration = len(audio_chunk) / (self.SAMPLE_RATE * 2)

                # Tính năng lượng RMS của toàn bộ đoạn audio 2.5s và phần tiếng nói thực tế
                chunk_samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                chunk_rms = float(np.sqrt(np.mean(chunk_samples ** 2))) if len(chunk_samples) > 0 else 0.0

                active_samples = []
                for i in range(0, len(chunk_samples), self.BLOCK_SIZE):
                    blk = chunk_samples[i : i + self.BLOCK_SIZE]
                    if len(blk) > 0:
                        b_rms = float(np.sqrt(np.mean(blk ** 2)))
                        if b_rms >= self.min_rms:
                            active_samples.extend(blk)
                active_rms = float(np.sqrt(np.mean(np.array(active_samples) ** 2))) if len(active_samples) > 0 else chunk_rms
                effective_rms = max(chunk_rms, active_rms)

                # === LỚP 2: VOICE AUTH (XÁC THỰC GIỌNG NÓI) ===
                self._log_debug(
                    f"[WakeWord 🛡️ LỚP 2] Đang xác thực sinh trắc học giọng nói "
                    f"({duration:.2f}s, RMS={chunk_rms:.0f}, effective_RMS={effective_rms:.0f})..."
                )
                is_owner, sim_score = self._verify_voice_auth(audio_chunk)

                if not is_owner:
                    # 1. Xác thực kép có điều kiện RMS:
                    # - Nếu RMS <= 500 và Voice Auth reject: reject luôn, không gọi Google, để tiết kiệm request.
                    if effective_rms <= 500.0:
                        self._log_debug(
                            f"[VoiceAuth] Từ chối (cosine={sim_score:.2f} < {self.voice_auth_threshold:.2f}, "
                            f"RMS={effective_rms:.0f} <= 500) -> Reject, không gọi Google."
                        )
                        time.sleep(0.1)
                        continue

                    # - Khi Voice Auth reject (cosine < 0.80) NHƯNG RMS > 500:
                    #   + Vẫn gửi audio lên Google STT như bình thường.
                    self._log_debug(
                        f"[VoiceAuth] Từ chối (cosine={sim_score:.2f} < {self.voice_auth_threshold:.2f}) "
                        f"NHƯNG RMS={effective_rms:.0f} > 500 -> Gửi Google STT để xác thực kép..."
                    )
                    google_text, is_net_err, err_msg = self._recognize_google(audio_chunk, timeout=2.5)
                    if is_net_err:
                        self._log_debug(
                            f"[VoiceAuth 🛡️ Xác thực kép] Google STT lỗi/timeout ({err_msg}) -> Hủy và quay lại Lớp 1."
                        )
                        time.sleep(0.1)
                        continue
                    elif self._is_google_text_matched(google_text):
                        self._log_debug(
                            f"[WakeWord][DualAuth] Voice Auth reject (cosine={sim_score:.2f}) "
                            f"nhưng Google STT xác nhận '{google_text}' + RMS={effective_rms:.0f} -> KÍCH HOẠT."
                        )
                        self._trigger_wake(f"DualAuth: Google STT ('{google_text}', RMS={effective_rms:.0f})")
                        break
                    else:
                        self._log_debug(
                            f"[VoiceAuth] Từ chối (cosine {sim_score:.2f} < {self.voice_auth_threshold:.2f}) "
                            f"và Google STT không xác nhận (kết quả: '{google_text}') -> HỦY BỎ, quay lại Lớp 1!"
                        )
                        time.sleep(0.1)
                        continue

                if sim_score == 1.0:
                    self._log_debug(
                        "[WakeWord 🛡️ LỚP 2 ℹ️ BỎ QUA] Chưa có file 'my_voice.pkl' (người dùng mới) "
                        "-> Bỏ qua xác thực giọng nói, chuyển sang Lớp 3."
                    )
                else:
                    self._log_debug(
                        f"[WakeWord 🛡️ LỚP 2 ✓ CHẤP NHẬN] Đúng giọng chủ nhân "
                        f"(cosine={sim_score:.2f} >= {self.voice_auth_threshold:.2f}) "
                        f"-> Chuyển sang Lớp 3 (Google STT) để xác nhận từ khóa."
                    )

                # === LỚP 3: GOOGLE STT (TRỌNG TÀI CHÍNH - ONLINE) ===
                self._log_debug(
                    f"[WakeWord 🌐 LỚP 3] Gửi audio {duration:.2f}s lên Google STT "
                    f"(language='vi-VN', timeout cứng 2.5s)..."
                )
                google_text, is_net_err, err_msg = self._recognize_google(audio_chunk, timeout=2.5)

                if is_net_err:
                    # Google STT bị lỗi mạng/timeout cứng 1.5s -> Chuyển sang LỚP 4: VOSK OFFLINE!
                    self._log_debug(
                        f"[WakeWord 🌐 LỚP 3 ⚠️ TIMEOUT/LỖI MẠNG] Google STT không phản hồi ({err_msg}) "
                        f"-> Chuyển ngay sang LỚP 4: VOSK (Phao cứu sinh Offline)!"
                    )
                    vosk_matched = self._handle_layer4_vosk(audio_chunk)
                    if vosk_matched:
                        break
                    else:
                        time.sleep(0.1)
                        continue

                # Google STT kết nối trực tuyến thành công
                self._log_debug(f"[WakeWord 🌐 LỚP 3] Google STT trả về: \"{google_text}\"")

                if self._is_google_text_matched(google_text):
                    self._log_debug(
                        f"[WakeWord 🌐 LỚP 3 ✓ TRỌNG TÀI CHÍNH XÁC NHẬN] "
                        f"Google STT xác nhận chứa 'teo dau': \"{google_text}\" "
                        f"-> KÍCH HOẠT WAKE WORD (Glow Orb & Bíp)!"
                    )
                    self._trigger_wake(f"Google STT ('{google_text}')")
                    break
                else:
                    self._log_debug(
                        f"[WakeWord 🌐 LỚP 3 ✗ BÁC BỎ] Google STT trả về \"{google_text}\" "
                        f"(không chứa 'teo dau') -> BỎ QUA, quay lại Lớp 1."
                    )
                    time.sleep(0.1)
                    continue

        except Exception as exc:
            if self._running and bool(self.is_listening):
                self._log_debug(f"[WakeWord Lỗi Stream Audio]: {exc}")
                if self.input_device is not None:
                    self._log_debug("[WakeWord] Tự động thử lại với Micro mặc định...")
                    self.input_device = None
                time.sleep(0.3)
        finally:
            with MIC_LOCK:
                if self.stream is not None:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None


# =============================================================================
# CHẠY THỬ NGHIỆM ĐỘC LẬP
# =============================================================================
if __name__ == "__main__":
    import winsound

    def test_callback(keyword: str, score: float) -> None:
        print(f"\n>>> [TEST CALLBACK] Phát hiện Wake Word '{keyword}'! Bíp...")
        winsound.Beep(1200, 200)
        time.sleep(1.0)
        print(">>> [TEST CALLBACK] Đang tiếp tục lắng nghe...\n")
        engine.resume()

    print("=" * 70)
    print("--- THỬ NGHIỆM WAKE WORD ENGINE 4 LỚP (GOOGLE STT + VOSK OFFLINE) ---")
    print("  • Lớp 1: WebRTC VAD + RMS > 350 (Phát hiện tiếng nói -> Thu âm động)")
    print("  • Lớp 2: Voice Auth + noisereduce (Cosine >= 0.80; Cơ chế xác thực kép nếu < 0.80 & RMS > 500)")
    print("  • Lớp 3: Google STT (Trọng tài chính Online, timeout 1.5s, tìm 'teo dau')")
    print("  • Lớp 4: Vosk Offline (Phao cứu sinh khi mất mạng, Grammar cứng ['tèo đâu', 'teo dau'])")
    print("=" * 70)

    engine = WakeWordEngine(
        wake_keyword="tèo đâu",
        on_detected=test_callback,
        min_rms=350.0,
        voice_auth_threshold=0.80,
        cooldown_sec=1.5,
    )

    engine.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nĐang dừng engine...")
        engine.stop()
