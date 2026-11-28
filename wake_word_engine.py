"""
Module: wake_word_engine.py
Mục đích: Bộ phát hiện từ khóa đánh thức (Wake Word Engine) cho Tèo Assistant.
Kiến trúc 4 Lớp Gọn Nhẹ (0 GB VRAM — Silero VAD + Resemblyzer + Google STT Chính + Vosk Lazy-Load):

--- 1. LỚP 1: SILERO VAD + DURATION GATE (CHẶN TIẾNG HÁT / NÓI CHUYỆN DÀI) ---
- Sử dụng Silero VAD (PyTorch onnx=False, 512 samples = 32ms ở 16kHz), pre-roll ~480ms (15 frames).
- Kích hoạt ghi động khi speech_prob >= 0.55 VÀ RMS >= 180.0; dừng khi speech_prob < 0.35 liên tiếp 0.65s (tối thiểu 0.75s).
- Trần thu tối đa 2.1s: nếu chạm 2.1s mà chưa ngắt im lặng (đang hát/nói dài) -> HỦY BỎ NGAY TẠI LỚP 1.

--- 2. LỚP 2: VOICE AUTH (XÁC THỰC GIỌNG CHỦ NHÂN - RESEMBLYZER 5 MẪU) ---
- Chạy verify_voice() từ voice_auth.py (Cosine >= 0.65), hỗ trợ bật/tắt Voice ID và DualAuth (>= threshold - 0.10 & RMS > 300).

--- 3. LỚP 3: GOOGLE STT (TRỌNG TÀI CHÍNH - ÂM THANH GỐC SẠCH) ---
- Tiền xử lý sạch: khử DC offset, khuếch đại nhẹ tối đa 3.0x khi RMS < 1000, đệm 200ms im lặng hai đầu (không noisereduce/high-pass/peak-norm).
- Gọi 1 lần timeout 2.0s (không retry khi rỗng), áp dụng Blacklist và khớp âm tiết `_is_google_text_matched`.

--- 4. LỚP 4: VOSK OFFLINE LAZY-LOAD (CHỈ NẠP KHI THẬT SỰ MẤT MẠNG / TIMEOUT) ---
- Không nạp lúc khởi động (0 MB RAM khi Online). Chỉ gọi `utils.get_vosk_model()` khi Lớp 3 gặp lỗi mạng.
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
from rapidfuzz import fuzz

try:
    import torch
    from silero_vad import load_silero_vad
    HAS_SILERO_VAD = True
except ImportError:
    torch = None
    load_silero_vad = None
    HAS_SILERO_VAD = False

try:
    from scipy.signal import butter, sosfiltfilt
    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False
    butter = None
    sosfiltfilt = None

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
    load_config,
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
    Bộ lắng nghe và phát hiện Wake Word 'Tèo đâu' (0 GB VRAM):
    - Lớp 1: Silero VAD (512 samples = 32ms, pre-roll 15 frames ~480ms, prob >= 0.55 & RMS >= 180) + Duration Gate 2.1s.
    - Lớp 2: Voice Auth (Resemblyzer Cosine >= 0.65, hỗ trợ bật/tắt và DualAuth >= threshold - 0.10 & RMS > 300).
    - Lớp 3: Google STT (âm thanh gốc sạch, timeout 2.0s, không retry khi rỗng) + Blacklist + khớp âm tiết.
    - Lớp 4: Vosk Offline Singleton Lazy-Load (chỉ nạp khi Lớp 3 lỗi mạng/timeout).
    """

    SAMPLE_RATE: int = 16000
    FRAME_MS: int = 32   # Silero VAD chuẩn 16kHz: 512 samples = 32ms
    BLOCK_SIZE: int = 512
    PRE_BUFFER_FRAMES: int = 15  # 15 x 32ms = 480ms pre-roll chống cắt phụ âm đầu "T"
    RECORD_DURATION_SEC: float = 2.1  # Trần 2.1s — vượt quá mà không im lặng sẽ hủy tại Lớp 1 (Duration Gate)
    SILENCE_LIMIT_SEC: float = 0.65   # Dừng thu động khi speech_prob < 0.35 liên tiếp 0.65s
    MIN_RECORD_SEC: float = 0.75      # Thời gian thu âm tối thiểu 0.75s

    SILERO_START_PROB: float = 0.55
    SILERO_CONT_PROB: float = 0.35

    TARGET_KEYWORD: str = "teo dau"
    DEFAULT_MIN_RMS: float = 180.0
    DUAL_AUTH_MIN_RMS: float = 300.0  # Ngưỡng RMS cho xác thực kép
    DEFAULT_FUZZY_THRESHOLD: float = 70.0
    DEFAULT_VOICE_AUTH_THRESHOLD: float = 0.65
    DEFAULT_MIN_COSINE: float = 0.65
    DEFAULT_COOLDOWN_SEC: float = 1.0
    DEFAULT_VAD_MODE: int = 1  # Dự phòng nếu fallback sang WebRTC VAD
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
    # Các cụm Google STT hay dịch nhầm từ "Tèo đâu" — CHỈ cứu khi chủ nhân đã xác thực rất chắc (cosine >= 0.75)
    OWNER_RESCUE_PHRASES: set = {
        "heo dat",
        "heo dau",
        "teo dat",
        "ta o dau",
        "tau o dau",
        "keo dau",
        "meo dau",
        "theo dau",
    }

    def __init__(
        self,
        vosk_model_path: Optional[str] = None,
        wake_keyword: str = "tèo đâu",
        threshold: float = 0.5,
        on_detected: Optional[Callable[[str, float], None]] = None,
        cooldown_sec: float = 1.0,
        grammar_words: Optional[List[str]] = None,
        input_device: Optional[int] = None,
        min_rms: float = 180.0,
        vad_mode: int = 1,
        min_conf: float = 0.65,
        fuzzy_threshold: float = 70.0,
        voice_auth_threshold: float = 0.65,
        min_cosine: Optional[float] = None,
        stt_verifier: Optional[Callable[..., str]] = None,
        voice_verifier: Optional[Callable[[bytes], Union[bool, Tuple[bool, float]]]] = None,
        on_early_wake: Optional[Callable[[], None]] = None,
        on_early_cancel: Optional[Callable[[], None]] = None,
        silence_limit_sec: float = 0.65,
        voice_auth_enabled: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Khởi tạo WakeWordEngine (Silero VAD + Voice Auth + Google STT + Vosk Lazy-Load).
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
        if voice_auth_enabled is None:
            self.voice_auth_enabled: bool = bool(load_config().get("voice_auth_enabled", True))
        else:
            self.voice_auth_enabled = bool(voice_auth_enabled)
        self.stt_verifier = stt_verifier
        self.voice_verifier = voice_verifier
        self.grammar_words = grammar_words or list(self.DEFAULT_GRAMMAR_WORDS)

        # 1. Khởi tạo bộ phát hiện tiếng người Silero VAD (PyTorch onnx=False; fallback WebRTC VAD nếu thiếu gói)
        self.silero_vad = None
        self.vad = None
        self._use_silero: bool = False
        self.block_size: int = self.BLOCK_SIZE
        self.frame_ms: int = self.FRAME_MS

        if HAS_SILERO_VAD and load_silero_vad is not None:
            try:
                self.silero_vad = load_silero_vad(onnx=False)
                self._use_silero = True
                self.block_size = 512
                self.frame_ms = 32
                print(
                    f"[WakeWord] ✓ Đã kích hoạt Silero VAD (PyTorch CPU, 512 samples/32ms, "
                    f"prob>={self.SILERO_START_PROB:.2f}, RMS>={self.min_rms:.0f}, trần {self.RECORD_DURATION_SEC:.1f}s) cho Lớp 1."
                )
            except Exception as silero_err:
                print(f"[WakeWord Cảnh báo] Không thể nạp Silero VAD ({silero_err}), fallback WebRTC VAD...")
                self.silero_vad = None
                self._use_silero = False

        if not self._use_silero:
            self.block_size = 480
            self.frame_ms = 30
            if webrtcvad is not None:
                try:
                    self.vad = webrtcvad.Vad(self.vad_mode)
                    print(f"[WakeWord] ✓ Fallback WebRTC VAD (mode={self.vad_mode}) cho Lớp 1.")
                except Exception as vad_err:
                    print(f"[WakeWord Cảnh báo] Không thể khởi tạo WebRTC VAD: {vad_err}")
                    self.vad = None
            else:
                print("[WakeWord Cảnh báo] Không tìm thấy Silero VAD lẫn WebRTC VAD!")

        # 2. Cấu hình thiết bị Microphone đầu vào
        if input_device is not None:
            self.input_device = input_device
            self.input_device_name = f"Device {input_device}"
        else:
            cfg_idx, cfg_name = get_configured_microphone()
            self.input_device = cfg_idx
            self.input_device_name = cfg_name

        # 3. KHÔNG nạp Vosk lúc khởi động (Lazy-Load qua utils.get_vosk_model() CHỈ khi mất mạng ở Lớp 4)
        self.model_path = vosk_model_path or ""
        self.model = None
        self.recognizer = None

        self._running = False
        self._paused = False
        self._is_listening = False
        self._in_verification = False
        self.stream: Optional[sd.RawInputStream] = None
        self.last_audio_time: float = time.time()
        self._thread: Optional[threading.Thread] = None
        self.pre_buffer: collections.deque = collections.deque(maxlen=self.PRE_BUFFER_FRAMES)
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog: threading.Event = threading.Event()
        self._lock = threading.Lock()
        self._last_trigger_time = 0.0

        vad_label = "Silero VAD" if self._use_silero else f"WebRTC VAD(mode={self.vad_mode})"
        self._log_debug(
            f"[WakeWord Khởi tạo 4 Lớp] Từ khóa: '{self.wake_keyword}', "
            f"Lớp 1={vad_label} (RMS>={self.min_rms:.0f}, trần {self.RECORD_DURATION_SEC:.1f}s), "
            f"Lớp 2=VoiceAuth({self.voice_auth_threshold}), Lớp 3=Google STT, Lớp 4=Vosk Lazy-Load."
        )

    def _reset_vad_states(self) -> None:
        """Đặt lại trạng thái nội bộ của Silero VAD để sạch cho lượt lắng nghe kế tiếp."""
        if self.silero_vad is not None:
            try:
                self.silero_vad.reset_states()
            except Exception:
                pass

    def _get_silero_prob(self, pcm_bytes: bytes) -> float:
        """Tính xác suất tiếng người (0.0 - 1.0) cho 1 frame 512 samples bằng Silero VAD."""
        if self.silero_vad is None or torch is None:
            return 0.0
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size != 512:
                return 0.0
            tensor = torch.from_numpy(samples)
            with torch.no_grad():
                return float(self.silero_vad(tensor, self.SAMPLE_RATE).item())
        except Exception:
            return 0.0

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
    # LỚP 3: GOOGLE STT (TRỌNG TÀI CHÍNH - ÂM THANH GỐC SẠCH)
    # =========================================================================
    def _preprocess_for_stt(self, audio_bytes: bytes) -> bytes:
        """
        Tiền xử lý âm thanh sạch cho Google STT (KHÔNG noisereduce, KHÔNG high-pass, KHÔNG peak-normalize 0.85):
        1. Khử DC offset (samples - mean).
        2. Tăng gain nhẹ NẾU RMS < 1000 với trần tối đa 3.0x: gain = min(3.0, 1500.0 / max(rms, 1.0)).
        3. Clip về int16 và đệm 200ms im lặng ở cả đầu và cuối để Google không cắt đầu/đuôi cụm ngắn "Tèo đâu".
        """
        if not audio_bytes or len(audio_bytes) < 1600:
            return audio_bytes

        try:
            usable_len = len(audio_bytes) - (len(audio_bytes) % 2)
            samples = np.frombuffer(audio_bytes[:usable_len], dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return audio_bytes

            # 1. Khử DC offset
            samples = samples - float(np.mean(samples))

            # 2. Tăng gain nhẹ nếu giọng nhỏ (RMS < 1000), trần tối đa 3.0x
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if 1.0 < rms < 1000.0:
                gain = min(3.0, 1500.0 / max(rms, 1.0))
                samples = samples * gain

            # 3. Clip về int16 và đệm 200ms im lặng ở đầu và cuối
            samples_int16 = np.clip(samples, -32768.0, 32767.0).astype(np.int16)
            pad_bytes = b"\x00" * (int(self.SAMPLE_RATE * 0.2) * 2)
            return pad_bytes + samples_int16.tobytes() + pad_bytes
        except Exception as exc:
            self._log_debug(f"[WakeWord Cảnh báo] Lỗi tiền xử lý âm thanh STT: {exc}")
            return audio_bytes

    def _preprocess_for_vosk(self, audio_bytes: bytes) -> bytes:
        """Tiền xử lý cho Vosk Offline: khử DC offset + khuếch đại nhẹ mục tiêu RMS ~ 3000 (trần 6.0x)."""
        if not audio_bytes or len(audio_bytes) < 1600:
            return audio_bytes
        try:
            usable_len = len(audio_bytes) - (len(audio_bytes) % 2)
            samples = np.frombuffer(audio_bytes[:usable_len], dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return audio_bytes
            samples = samples - float(np.mean(samples))
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if 1.0 < rms < 3000.0:
                gain = min(6.0, 3000.0 / max(rms, 1.0))
                samples = samples * gain
            return np.clip(samples, -32768.0, 32767.0).astype(np.int16).tobytes()
        except Exception:
            return audio_bytes

    def _recognize_google(self, audio_bytes: bytes, timeout: float = 2.8) -> Tuple[str, bool, str]:
        """
        Gửi đoạn âm thanh lên Google STT (language='vi-VN', timeout cứng 2.8s).
        Chỉ gọi 1 lần duy nhất (không retry khi rỗng để tránh đơ luồng).

        Returns:
            (recognized_text: str, is_network_error: bool, error_message: str)
            - Nếu nhận diện thành công: (text, False, "")
            - Nếu lỗi mạng / timeout / connection error: ("", True, err_msg)
            - Nếu không nhận diện được chữ (UnknownValueError): ("", False, "")
        """
        if not audio_bytes:
            return "", False, "Dữ liệu âm thanh rỗng"

        # Tiền xử lý âm thanh sạch: khử DC offset + khuếch đại nhẹ + đệm 200ms im lặng hai đầu
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

        # Bọc hàm gọi Google STT vào ThreadPoolExecutor với timeout cứng (mặc định 2.0s)
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

        if is_net_err and "timeout" in err_msg.lower():
            self._log_debug(f"[Google STT] Timeout {timeout:.1f}s -> Hủy và quay lại Lớp 1")

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
    def _is_google_text_matched(
        cls,
        google_text: str,
        owner_cosine: float = 0.0,
        threshold: float = 72.0,
    ) -> bool:
        """
        Kiểm tra matching theo âm tiết (có Owner-Verified Phonetic Rescue & Blacklist):
        - Owner-Verified Phonetic Rescue: CHỈ khi `owner_cosine >= 0.75`, nếu câu chuẩn hóa khớp chính xác
          1 trong các cụm Google STT hay dịch nhầm từ 'Tèo đâu' (OWNER_RESCUE_PHRASES) -> ACCEPT.
        - Chặn ngay nếu thuộc danh sách Blacklist ('mèo', 'heo', 'đi đâu', 'ở đâu', 'tèo đâu rồi'...).
        - Hỗ trợ:
          + 'tèo đâu' (kể cả prefix 'tèo đâu ...', hoặc câu 2–4 âm tiết có cặp (w1, w2) khớp 'teo' & 'dau').
          + 'tèo ơi' (cặp âm tiết w1 khớp 'teo' >= 80% + w2 == 'oi').
          + 'alo tèo' / 'a lô tèo' (cặp w1 in {'alo', 'lo'} + w2 khớp 'teo' >= 80%).
        - REJECT mọi trường hợp còn lại.
        """
        if not google_text or not google_text.strip():
            return False

        # 1. Owner-Verified Phonetic Rescue (CHỈ khi Voice ID đã xác thực chủ nhân rất chắc: cosine >= 0.75)
        if owner_cosine >= 0.75:
            clean_phrase = strip_accents(google_text.lower())
            clean_phrase = re.sub(r"[^\w\s]", " ", clean_phrase)
            clean_phrase = re.sub(r"\s+", " ", clean_phrase).strip()
            if clean_phrase in cls.OWNER_RESCUE_PHRASES:
                log_line = f"[WakeWord][Match] ACCEPT (Owner-Rescue cosine={owner_cosine:.2f}) text='{google_text}'"
                print(log_line, flush=True)
                try:
                    log_path = get_exe_dir() / "teo_debug.log"
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + log_line + "\n")
                except Exception:
                    pass
                return True

        # 2. Kiểm tra Blacklist (chặn tuyệt đối khi owner_cosine < 0.75 hoặc không nằm trong Rescue)
        if cls.is_blacklisted(google_text):
            log_line = f"[WakeWord][Match] REJECT (blacklisted) text='{google_text}'"
            print(log_line, flush=True)
            try:
                log_path = get_exe_dir() / "teo_debug.log"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + log_line + "\n")
            except Exception:
                pass
            return False

        raw_syllables = google_text.strip().split()
        syllables = [cls._normalize_syllable(s) for s in raw_syllables if cls._normalize_syllable(s)]
        if len(syllables) < 2:
            return False

        is_match = False
        best_w1_ratio = 0.0
        best_w2_ratio = 0.0

        # Nếu câu bắt đầu nguyên vẹn bằng 'teo dau', 'teo oi', 'alo teo', hoặc 'a lo teo' -> ACCEPT
        if (
            (syllables[0] == "teo" and syllables[1] in ("dau", "oi"))
            or (syllables[0] == "alo" and syllables[1] == "teo")
            or (len(syllables) >= 3 and syllables[0] == "a" and syllables[1] == "lo" and syllables[2] == "teo")
        ):
            is_match = True
            best_w1_ratio = 100.0
            best_w2_ratio = 100.0
        elif 2 <= len(syllables) <= 4:
            allowed_w1 = {"teo", "theo", "tien", "keo"}
            for i in range(len(syllables) - 1):
                w1, w2 = syllables[i], syllables[i + 1]
                w1_teo_ratio = float(fuzz.ratio(w1, "teo"))
                w2_dau_ratio = float(fuzz.ratio(w2, "dau"))
                w2_teo_ratio = float(fuzz.ratio(w2, "teo"))

                if (w1_teo_ratio + w2_dau_ratio) > (best_w1_ratio + best_w2_ratio):
                    best_w1_ratio = w1_teo_ratio
                    best_w2_ratio = w2_dau_ratio

                # Nhánh 1: "tèo đâu"
                w1_ok = (w1 in allowed_w1) or (w1_teo_ratio >= 72.0)
                w2_ok = (w2_dau_ratio >= 75.0) or (w1 == "teo" and w2 == "do")
                if w1_ok and w2_ok:
                    is_match = True
                    best_w1_ratio = w1_teo_ratio
                    best_w2_ratio = w2_dau_ratio
                    break

                # Nhánh 2: "tèo ơi" (w1 khớp "teo" >= 80% + w2 == "oi")
                if w1_teo_ratio >= 80.0 and w2 == "oi":
                    is_match = True
                    best_w1_ratio = w1_teo_ratio
                    best_w2_ratio = 100.0
                    break

                # Nhánh 3: "alo tèo" / "a lô tèo" (w1 in {"alo", "lo"} + w2 khớp "teo" >= 80%)
                if w1 in ("alo", "lo") and w2_teo_ratio >= 80.0:
                    is_match = True
                    best_w1_ratio = 100.0
                    best_w2_ratio = w2_teo_ratio
                    break

        status = "ACCEPT" if is_match else "REJECT"
        log_line = f"[WakeWord][Match] {status} syllables={syllables}, w1_ratio={best_w1_ratio:.1f}, w2_ratio={best_w2_ratio:.1f}"
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
        Hàm xác thực chéo (Lớp 2 Voice Auth + Lớp 3 Google STT + Lớp 4 Vosk Lazy-Load khi lỗi mạng):
        1. Xác thực giọng nói (Voice Auth) nếu voice_auth_enabled == True.
        2. Nếu Voice Auth từ chối nhưng sim_score >= (voice_auth_threshold - 0.10), RMS > dual_auth_min_rms
           và Google STT khớp 'teo dau' -> Xác thực kép thành công.
        3. Gọi Google STT 1 lần (timeout 2.8s). Nếu rỗng "" không lỗi mạng -> bỏ qua (KHÔNG nạp Vosk).
        4. CHỈ khi lỗi mạng/timeout (is_net_err=True) -> chuyển Lớp 4 Vosk Lazy-Load.
        """
        if not candidate_audio:
            return False

        chunk_samples = np.frombuffer(candidate_audio, dtype=np.int16).astype(np.float32)
        chunk_rms = float(np.sqrt(np.mean(chunk_samples ** 2))) if len(chunk_samples) > 0 else 0.0
        active_samples = []
        for i in range(0, len(chunk_samples), self.block_size):
            blk = chunk_samples[i : i + self.block_size]
            if len(blk) > 0:
                b_rms = float(np.sqrt(np.mean(blk ** 2)))
                if b_rms >= self.min_rms:
                    active_samples.extend(blk)
        active_rms = float(np.sqrt(np.mean(np.array(active_samples) ** 2))) if len(active_samples) > 0 else chunk_rms
        effective_rms = max(chunk_rms, active_rms)

        verified_owner_cosine = 0.0
        if self.voice_auth_enabled:
            is_owner, sim_score = self._verify_voice_auth(candidate_audio)
            if not is_owner:
                dual_auth_min_cosine = self.voice_auth_threshold - 0.10
                if effective_rms <= self.dual_auth_min_rms or sim_score < dual_auth_min_cosine:
                    return False
                google_text, is_net_err, _ = self._recognize_google(candidate_audio, timeout=2.8)
                if not is_net_err and self._is_google_text_matched(google_text, owner_cosine=0.0):
                    self._log_debug(
                        f"[WakeWord][DualAuth] Voice Auth reject (cosine={sim_score:.2f}) "
                        f"nhưng Google STT xác nhận '{google_text}' + RMS={effective_rms:.0f} -> KÍCH HOẠT."
                    )
                    self._trigger_wake(f"DualAuth: Google STT ('{google_text}', RMS={effective_rms:.0f})")
                    return True
                return False
            if sim_score < 1.0:
                verified_owner_cosine = sim_score

        google_text, is_net_err, _ = self._recognize_google(candidate_audio, timeout=2.8)
        if not is_net_err and self._is_google_text_matched(google_text, owner_cosine=verified_owner_cosine):
            self._trigger_wake(f"Google STT ('{google_text}')")
            return True
        elif is_net_err:
            # CHỈ khi Google STT bị lỗi mạng / timeout mới gọi Lớp 4 (Vosk Lazy-Load)
            vosk_matched = self._handle_layer4_vosk(candidate_audio)
            return bool(vosk_matched)
        else:
            # Google STT trả về rỗng "" hoặc sai từ khóa -> bỏ qua bình thường, KHÔNG nạp Vosk
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
        Xử lý Lớp 4 (Phao cứu sinh Vosk Offline Lazy-Load):
        - CHỈ kích hoạt khi Google STT gặp sự cố mạng / timeout.
        - Tự động gọi utils.get_vosk_model() đúng 1 lần duy nhất (dùng chung với STT).
        - Khởi tạo KaldiRecognizer với Grammar cứng: ["tèo đâu", "teo dau", "[unk]"].
        - Khuếch đại nhẹ mục tiêu RMS ~ 3000 (trần 6.0x, khử DC offset) trước khi nạp vào Vosk.
        """
        if self.model is None:
            from utils import get_vosk_model
            self.model = get_vosk_model(self.model_path or None)

        if not self.model:
            self._log_debug("[WakeWord 🛟 Lớp 4 Lỗi] Mô hình Vosk chưa sẵn sàng!")
            return False

        try:
            import vosk
            t0 = time.time()
            grammar_list = self.grammar_words or list(self.DEFAULT_GRAMMAR_WORDS)
            grammar_json = json.dumps(grammar_list, ensure_ascii=False)

            if self.recognizer is not None:
                rec = self.recognizer
                rec.Reset()
            else:
                rec = vosk.KaldiRecognizer(self.model, float(self.SAMPLE_RATE), grammar_json)
                try:
                    rec.SetWords(True)
                except Exception:
                    pass
                self.recognizer = rec

            vosk_audio = self._preprocess_for_vosk(audio_bytes)
            rec.AcceptWaveform(vosk_audio)
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
                        blocksize=self.block_size,
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
            self._in_verification = False
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

        # 4. Clear pre-buffer deque & reset Silero VAD states
        if hasattr(self, "pre_buffer") and self.pre_buffer is not None:
            self.pre_buffer.clear()
        self._reset_vad_states()

        # 5. Set last_audio_time = time.time()
        self.last_audio_time = time.time()

        # 6. Mở stream mới
        with MIC_LOCK:
            try:
                new_stream = sd.RawInputStream(
                    samplerate=self.SAMPLE_RATE,
                    blocksize=self.block_size,
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
        """Thread AudioWatchdog, chu kỳ 0.5s, chạy trong lúc is_listening=True và không đang xác thực."""
        while not self._stop_watchdog.wait(0.5):
            if not self._running:
                break
            if bool(self.is_listening) and not self._in_verification:
                now = time.time()
                last_chunk_age = now - self.last_audio_time
                if last_chunk_age > 4.0:
                    self._log_debug(
                        f"[Watchdog][PANIC] Stream treo: {last_chunk_age:.1f}s không có audio chunk, tiến hành reset."
                    )
                    self.force_reset_stream()

    def reset_microphone(self) -> None:
        """Reset lại toàn bộ luồng Micro."""
        self._log_debug("[WakeWord] 🔄 Đang reset lại luồng Microphone...")
        self.force_reset_stream()

    # =========================================================================
    # VÒNG LẶP LẮNG NGHE CHÍNH (SILERO VAD + DURATION GATE + VOICE AUTH + GOOGLE STT + VOSK LAZY)
    # =========================================================================
    def _listen_loop(self) -> None:
        """
        Vòng lặp thu âm micro và xử lý 4 Lớp (0 GB VRAM):
        - Lớp 1: Silero VAD (512 samples/32ms, pre-roll ~480ms, prob >= 0.55 & RMS >= 180)
                 -> Thu động (dừng khi prob < 0.35 liên tiếp 0.65s, tối thiểu 0.75s, trần 2.1s).
                 -> Duration Gate: nếu chạm trần 2.1s mà chưa ngắt im lặng (hát/nói dài) -> HỦY BỎ TẠI LỚP 1.
        - Lớp 2: Voice Auth (Resemblyzer Cosine >= 0.65; hỗ trợ tắt Voice ID và DualAuth >= threshold-0.10 & RMS>300).
        - Lớp 3: Google STT (âm thanh gốc sạch + đệm 200ms im lặng, timeout 2.0s, 1 lần duy nhất).
        - Lớp 4: Vosk Offline Lazy-Load (CHỈ nạp khi Lớp 3 lỗi mạng/timeout).
        """
        if not self._running or not bool(self.is_listening) or not self.is_stream_alive():
            return

        # Buffer trượt lưu ~480ms âm thanh trước thời điểm phát hiện tiếng nói
        # (15 blocks x 32ms = 480ms) để không bao giờ nuốt mất phụ âm đầu "T" của "Tèo..."
        self.pre_buffer.clear()
        self._reset_vad_states()
        pre_buffer = self.pre_buffer
        frame_sec = self.frame_ms / 1000.0
        target_chunk_blocks = int(round(self.RECORD_DURATION_SEC / frame_sec))
        silence_target_blocks = max(1, int(round(self.silence_limit_sec / frame_sec)))
        min_record_blocks = max(1, int(round(self.MIN_RECORD_SEC / frame_sec)))

        vad_info = (
            f"Silero VAD prob>={self.SILERO_START_PROB:.2f}, RMS>={self.min_rms:.0f}, trần {self.RECORD_DURATION_SEC:.1f}s"
            if self._use_silero
            else f"WebRTC VAD mode={self.vad_mode}, RMS>={self.min_rms:.0f}, trần {self.RECORD_DURATION_SEC:.1f}s"
        )
        self._log_debug(
            f"[WakeWord] 🎙️ Đang liên tục lắng nghe qua Micro ({self.input_device_name}) [{vad_info}]..."
        )

        try:
            while self._running and bool(self.is_listening) and self.is_stream_alive():
                try:
                    data, overflowed = self.stream.read(self.block_size)
                except Exception as read_err:
                    if self._running and bool(self.is_listening):
                        self._log_debug(f"[WakeWord Lỗi Stream Read]: {read_err}")
                    break

                if not data:
                    continue

                raw_bytes = bytes(data)

                # Cập nhật mốc thời gian nhận audio mới nhất mỗi chunk
                self.last_audio_time = time.time()

                # 1. Kiểm tra Cooldown sau lần kích hoạt gần nhất
                now = time.time()
                if (now - self._last_trigger_time) < self.cooldown_sec:
                    pre_buffer.append(raw_bytes)
                    continue

                # 2. Tính năng lượng âm lượng RMS của frame
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

                # 3. LỚP 1: Kiểm tra tiếng nói người (Silero VAD prob >= 0.55 & RMS >= 180.0)
                speech_prob = 0.0
                if self._use_silero:
                    speech_prob = self._get_silero_prob(raw_bytes)
                    is_human_speech = (speech_prob >= self.SILERO_START_PROB) and (rms >= self.min_rms)
                else:
                    is_speech_vad = True
                    if self.vad is not None:
                        try:
                            is_speech_vad = self.vad.is_speech(raw_bytes, self.SAMPLE_RATE)
                        except Exception:
                            is_speech_vad = False
                    is_human_speech = is_speech_vad and (rms >= self.min_rms)

                if not is_human_speech:
                    # Là khoảng lặng hoặc tiếng ồn nền -> Lưu vào pre_buffer và tiếp tục nghe
                    pre_buffer.append(raw_bytes)
                    continue

                # === PHÁT HIỆN TIẾNG NGƯỜI NÓI (LỚP 1 TRIGGER) ===
                preroll_ms = len(pre_buffer) * self.frame_ms
                prob_str = f", prob={speech_prob:.2f}" if self._use_silero else ""
                self._log_debug(
                    f"\n[WakeWord 🎙️ LỚP 1] Pre-roll {preroll_ms}ms + ghi động "
                    f"(RMS: {rms:.0f} >= {self.min_rms:.0f}{prob_str}, "
                    f"dừng khi im lặng {self.silence_limit_sec:.2f}s, trần {self.RECORD_DURATION_SEC:.1f}s)..."
                )

                # Thu âm động: gồm ~480ms pre-roll + frame hiện tại + các frame tiếp theo
                recorded_frames = list(pre_buffer)
                recorded_frames.append(raw_bytes)
                consecutive_silence_blocks = 0
                stopped_by_silence = False

                while self._running and bool(self.is_listening) and self.is_stream_alive() and len(recorded_frames) < target_chunk_blocks:
                    try:
                        chunk, ov = self.stream.read(self.block_size)
                    except Exception:
                        break
                    if not chunk:
                        continue
                    self.last_audio_time = time.time()
                    chunk_bytes = bytes(chunk)
                    recorded_frames.append(chunk_bytes)

                    if self._use_silero:
                        c_prob = self._get_silero_prob(chunk_bytes)
                        is_chunk_speech = c_prob >= self.SILERO_CONT_PROB
                    else:
                        c_samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
                        c_rms = float(np.sqrt(np.mean(c_samples ** 2))) if len(c_samples) > 0 else 0.0
                        is_chunk_speech = False
                        if self.vad is not None:
                            try:
                                is_chunk_speech = self.vad.is_speech(chunk_bytes, self.SAMPLE_RATE)
                            except Exception:
                                is_chunk_speech = False
                            is_chunk_speech = is_chunk_speech and (c_rms >= (self.min_rms * 0.55))
                        else:
                            is_chunk_speech = c_rms >= (self.min_rms * 0.55)

                    if not is_chunk_speech:
                        consecutive_silence_blocks += 1
                        if len(recorded_frames) >= min_record_blocks and consecutive_silence_blocks >= silence_target_blocks:
                            stopped_by_silence = True
                            dur_so_far = len(recorded_frames) * frame_sec
                            self._log_debug(
                                f"[WakeWord 🎙️ LỚP 1 - Thu âm động] Phát hiện im lặng "
                                f"{consecutive_silence_blocks * frame_sec:.2f}s sau tiếng nói "
                                f"-> Dừng thu âm sớm ở {dur_so_far:.2f}s!"
                            )
                            break
                    else:
                        consecutive_silence_blocks = 0

                # Luôn reset trạng thái Silero VAD sau khi kết thúc một lượt thu âm động
                self._reset_vad_states()
                pre_buffer.clear()

                if not self._running or not bool(self.is_listening):
                    break

                # DURATION GATE: Chặn tiếng hát / nói chuyện dài kéo dài >= 2.1s ngay tại Lớp 1
                if not stopped_by_silence:
                    self._log_debug(
                        f"[WakeWord 🎙️ LỚP 1 ✗ BỎ QUA] Tiếng nói/hát kéo dài >= {self.RECORD_DURATION_SEC:.1f}s -> Không phải 'Tèo đâu'"
                    )
                    continue

                audio_chunk = b"".join(recorded_frames)

                # Kiểm tra độ dài âm thanh tối thiểu (0.75s)
                if len(audio_chunk) < int(self.SAMPLE_RATE * 2 * self.MIN_RECORD_SEC):
                    continue

                duration = len(audio_chunk) / (self.SAMPLE_RATE * 2)

                # Tính năng lượng RMS của đoạn audio và phần tiếng nói thực tế
                chunk_samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                chunk_rms = float(np.sqrt(np.mean(chunk_samples ** 2))) if len(chunk_samples) > 0 else 0.0

                active_samples = []
                for i in range(0, len(chunk_samples), self.block_size):
                    blk = chunk_samples[i : i + self.block_size]
                    if len(blk) > 0:
                        b_rms = float(np.sqrt(np.mean(blk ** 2)))
                        if b_rms >= self.min_rms:
                            active_samples.extend(blk)
                active_rms = float(np.sqrt(np.mean(np.array(active_samples) ** 2))) if len(active_samples) > 0 else chunk_rms
                effective_rms = max(chunk_rms, active_rms)

                # Đánh dấu đang trong quá trình xác thực Lớp 2/3/4 để AudioWatchdog không PANIC reset stream
                self._in_verification = True
                try:
                    # === LỚP 2: VOICE AUTH (XÁC THỰC GIỌNG NÓI) ===
                    verified_owner_cosine = 0.0
                    if not self.voice_auth_enabled:
                        self._log_debug(
                            "[WakeWord 🛡️ LỚP 2 ℹ️ TẮT] Voice ID đang tắt -> "
                            "Bỏ qua xác thực giọng nói, đi thẳng sang Lớp 3 (Google STT)."
                        )
                    else:
                        self._log_debug(
                            f"[WakeWord 🛡️ LỚP 2] Đang xác thực sinh trắc học giọng nói "
                            f"({duration:.2f}s, RMS={chunk_rms:.0f}, effective_RMS={effective_rms:.0f})..."
                        )
                        is_owner, sim_score = self._verify_voice_auth(audio_chunk)

                        if not is_owner:
                            dual_auth_min_cosine = self.voice_auth_threshold - 0.10
                            # Xác thực kép: chỉ cho phép khi RMS > dual_auth_min_rms (300) VÀ sim_score >= (threshold - 0.10)
                            if effective_rms <= self.dual_auth_min_rms or sim_score < dual_auth_min_cosine:
                                self._log_debug(
                                    f"[VoiceAuth] Từ chối (cosine={sim_score:.2f} < {self.voice_auth_threshold:.2f}, "
                                    f"dual_min={dual_auth_min_cosine:.2f}, RMS={effective_rms:.0f}, "
                                    f"dual_rms>{self.dual_auth_min_rms:.0f}) -> Reject, bỏ qua Lớp 3."
                                )
                                time.sleep(0.05)
                                continue

                            self._log_debug(
                                f"[VoiceAuth] Từ chối (cosine={sim_score:.2f} >= {dual_auth_min_cosine:.2f}) "
                                f"NHƯNG RMS={effective_rms:.0f} > {self.dual_auth_min_rms:.0f} -> Gửi Google STT xác thực kép..."
                            )
                            google_text, is_net_err, err_msg = self._recognize_google(audio_chunk, timeout=2.8)
                            if is_net_err:
                                self._log_debug(
                                    f"[VoiceAuth 🛡️ Xác thực kép] Google STT lỗi/timeout ({err_msg}) -> Hủy và quay lại Lớp 1."
                                )
                                time.sleep(0.05)
                                continue
                            elif self._is_google_text_matched(google_text, owner_cosine=0.0):
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
                                time.sleep(0.05)
                                continue

                        if sim_score == 1.0:
                            self._log_debug(
                                "[WakeWord 🛡️ LỚP 2 ℹ️ BỎ QUA] Chưa có file 'my_voice.pkl' (người dùng mới) "
                                "-> Bỏ qua xác thực giọng nói, chuyển sang Lớp 3."
                            )
                        else:
                            verified_owner_cosine = sim_score
                            self._log_debug(
                                f"[WakeWord 🛡️ LỚP 2 ✓ CHẤP NHẬN] Đúng giọng chủ nhân "
                                f"(cosine={sim_score:.2f} >= {self.voice_auth_threshold:.2f}) "
                                f"-> Chuyển sang Lớp 3 (Google STT) để xác nhận từ khóa."
                            )

                    # === LỚP 3: GOOGLE STT (TRỌNG TÀI CHÍNH - ONLINE, 1 LẦN TIMEOUT 2.8s) ===
                    self._log_debug(
                        f"[WakeWord 🌐 LỚP 3] Gửi audio {duration:.2f}s lên Google STT (timeout 2.8s)..."
                    )
                    google_text, is_net_err, err_msg = self._recognize_google(audio_chunk, timeout=2.8)

                    if is_net_err:
                        # CHỈ KHI MẤT MẠNG / TIMEOUT -> Nạp Vosk Lazy-Load (Lớp 4)
                        self._log_debug(
                            f"[WakeWord 🌐 LỚP 3 ⚠️ LỖI MẠNG/TIMEOUT] ({err_msg}) "
                            f"-> Chuyển sang LỚP 4: Vosk Offline Lazy-Load!"
                        )
                        vosk_matched = self._handle_layer4_vosk(audio_chunk)
                        if vosk_matched:
                            break
                        else:
                            time.sleep(0.05)
                            continue

                    # Google STT kết nối thành công
                    self._log_debug(f'[WakeWord 🌐 LỚP 3] Google STT trả về: "{google_text}"')

                    if not google_text:
                        # Trả rỗng "" (không lỗi mạng) -> bỏ qua bình thường, KHÔNG nạp Vosk
                        time.sleep(0.05)
                        continue

                    if self._is_google_text_matched(google_text, owner_cosine=verified_owner_cosine):
                        self._log_debug(
                            f"[WakeWord 🌐 LỚP 3 ✓ XÁC NHẬN] "
                            f"Google STT xác nhận từ khóa: \"{google_text}\" "
                            f"-> KÍCH HOẠT WAKE WORD (Glow Orb & Dạ)!"
                        )
                        self._trigger_wake(f"Google STT ('{google_text}')")
                        break
                    else:
                        self._log_debug(
                            f"[WakeWord 🌐 LỚP 3 ✗ BÁC BỎ] Google STT trả về \"{google_text}\" "
                            f"(không khớp từ khóa đánh thức) -> BỎ QUA, quay lại Lớp 1."
                        )
                        time.sleep(0.05)
                        continue
                finally:
                    self._in_verification = False
                    self.last_audio_time = time.time()

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
    print("--- THỬ NGHIỆM WAKE WORD ENGINE 4 LỚP (SILERO VAD + GOOGLE STT + VOSK LAZY) ---")
    print("  • Lớp 1: Silero VAD (512 samples/32ms, prob >= 0.55 & RMS >= 180, Duration Gate 2.1s)")
    print("  • Lớp 2: Voice Auth 5 mẫu (Cosine >= 0.65; DualAuth khi >= 0.55 & RMS > 300)")
    print("  • Lớp 3: Google STT (âm thanh gốc sạch + đệm 200ms, timeout 2.0s)")
    print("  • Lớp 4: Vosk Offline Lazy-Load (chỉ nạp khi mất mạng)")
    print("=" * 70)

    engine = WakeWordEngine(
        wake_keyword="tèo đâu",
        on_detected=test_callback,
        min_rms=180.0,
        voice_auth_threshold=0.65,
        cooldown_sec=1.5,
    )

    engine.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nĐang dừng engine...")
        engine.stop()
