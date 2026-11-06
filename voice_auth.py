"""
Module: voice_auth.py
Mục đích: Xác thực sinh trắc học giọng nói ngoại tuyến (Offline 100%) sử dụng Resemblyzer (VoiceEncoder).
Chức năng:
    - Trích xuất đặc trưng giọng nói (256-dimensional embedding vector).
    - So sánh giọng nói hiện tại với hồ sơ chủ nhân lưu trong `my_voice.pkl` qua độ tương đồng Cosine (Cosine Similarity).
    - Chặn đứng 99% hiện tượng kích hoạt ảo từ tiếng ồn, tiếng TV hoặc giọng người lạ.
"""

import os
from pathlib import Path
import pickle
import sys
import time
from typing import Optional, Tuple, Union

import numpy as np

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo nạp đường dẫn utils
from utils import get_data_path, resource_path

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False
    VoiceEncoder = None
    preprocess_wav = None

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    nr = None


class VoiceAuthenticator:
    """
    Bộ xác thực giọng nói ngoại tuyến sử dụng VoiceEncoder của Resemblyzer.
    Khởi tạo mô hình một lần duy nhất vào bộ nhớ để tái sử dụng tức thì (<10ms).
    """

    DEFAULT_THRESHOLD: float = 0.80
    min_cosine: float = 0.80
    PROFILE_FILENAME: str = "my_voice.pkl"

    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path or self._resolve_profile_path()
        self.encoder: Optional[VoiceEncoder] = None
        self.reference_embedding: Optional[np.ndarray] = None
        self.min_cosine: float = self.DEFAULT_THRESHOLD
        self._load_encoder()
        self.load_profile()

    def _resolve_profile_path(self) -> str:
        """Tìm đường dẫn tệp my_voice.pkl trong thư mục ứng dụng hoặc get_data_path."""
        local_file = Path(self.PROFILE_FILENAME)
        if local_file.exists():
            return str(local_file.resolve())
        return get_data_path(self.PROFILE_FILENAME, default_content="")

    def _load_encoder(self) -> None:
        """Khởi tạo mô hình VoiceEncoder trên CPU."""
        if not HAS_RESEMBLYZER:
            print("[VoiceAuth Cảnh báo] Chưa cài đặt 'resemblyzer' hoặc không thể import!")
            return
        try:
            t0 = time.time()
            self.encoder = VoiceEncoder(device="cpu", verbose=False)
            elapsed = round(time.time() - t0, 3)
            print(f"[VoiceAuth] ✓ Đã nạp VoiceEncoder ({elapsed}s, sẵn sàng xác thực chủ nhân).")
        except Exception as exc:
            print(f"[VoiceAuth Lỗi] Không thể nạp VoiceEncoder: {exc}")
            self.encoder = None

    def load_profile(self) -> bool:
        """Tải vector đặc trưng giọng chủ nhân từ tệp my_voice.pkl."""
        path = Path(self.profile_path)
        if not path.exists() or path.stat().st_size == 0:
            self.reference_embedding = None
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "embedding" in data:
                emb = np.array(data["embedding"], dtype=np.float32)
            elif isinstance(data, np.ndarray):
                emb = data.astype(np.float32)
            else:
                emb = np.array(data, dtype=np.float32)

            # Chuẩn hóa vector đơn vị L2
            norm = np.linalg.norm(emb)
            if norm > 0:
                self.reference_embedding = emb / norm
                print(f"[VoiceAuth] ✓ Đã tải hồ sơ giọng chủ nhân từ '{path.name}' (vector {self.reference_embedding.shape}).")
                return True
            else:
                self.reference_embedding = None
                return False
        except Exception as exc:
            print(f"[VoiceAuth Lỗi] Không thể đọc tệp hồ sơ giọng nói '{path}': {exc}")
            self.reference_embedding = None
            return False

    def is_enrolled(self) -> bool:
        """Kiểm tra người dùng đã có hồ sơ giọng nói mẫu chưa."""
        return self.reference_embedding is not None

    def compute_embedding(
        self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Trích xuất vector đặc trưng 256 chiều từ dữ liệu âm thanh.
        Hỗ trợ:
        - bytes: Dữ liệu PCM 16-bit mono 16000Hz (từ microphone hoặc file).
        - np.ndarray: Mảng số nguyên int16 hoặc số thực float32 [-1, 1].
        """
        if self.encoder is None:
            return None

        try:
            # 1. Chuyển đổi về float32 waveform chuẩn [-1.0, 1.0]
            if isinstance(audio_data, bytes):
                if len(audio_data) < 1600:  # Ít hơn 0.05s
                    return None
                wav = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                if audio_data.dtype == np.int16:
                    wav = audio_data.astype(np.float32) / 32768.0
                else:
                    wav = audio_data.astype(np.float32)
            else:
                return None

            if len(wav) == 0:
                return None

            # 1.5. Lọc bớt tiếng ồn nền bằng noisereduce trước khi đưa vào Resemblyzer
            if HAS_NOISEREDUCE and nr is not None and float(np.max(np.abs(wav))) > 1e-4:
                try:
                    reduced_noise = nr.reduce_noise(y=wav, sr=sample_rate)
                    wav = np.nan_to_num(reduced_noise, nan=0.0, posinf=1.0, neginf=-1.0)
                except Exception as nr_exc:
                    pass

            # 2. Tiền xử lý waveform qua Resemblyzer (chuẩn hóa âm lượng, cắt khoảng lặng)
            preprocessed = preprocess_wav(wav, source_sr=sample_rate)
            if len(preprocessed) < 1600:
                # Nếu tiền xử lý cắt quá ngắn, dùng trực tiếp wav ban đầu
                preprocessed = wav

            # 3. Trích xuất embedding
            embedding = self.encoder.embed_utterance(preprocessed)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as exc:
            print(f"[VoiceAuth Lỗi] Lỗi trích xuất embedding: {exc}")
            return None

    def verify_voice(
        self,
        audio_data: Union[bytes, np.ndarray],
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Xác thực xem âm thanh có phải là giọng nói của chủ nhân hay không.
        
        Args:
            audio_data: Dữ liệu âm thanh thô (bytes PCM 16-bit hoặc numpy array).
            sample_rate: Tần số lấy mẫu (mặc định 16000Hz).
            threshold: Ngưỡng Cosine Similarity tối thiểu (mặc định 0.80).

        Returns:
            (is_match: bool, similarity: float)
        """
        target_thresh = threshold if threshold is not None else self.min_cosine

        # Nếu chưa có profile giọng chủ nhân (người dùng mới / first-run: Tùy chọn)
        if not self.is_enrolled():
            # Thử nạp lại phòng trường hợp vừa ghi âm xong
            if not self.load_profile():
                print("[VoiceAuth] Chưa đăng ký giọng nói, bỏ qua xác thực.")
                return True, 1.0

        current_embedding = self.compute_embedding(audio_data, sample_rate=sample_rate)
        if current_embedding is None:
            return False, 0.0

        # Tính Cosine Similarity giữa 2 vector đơn vị
        similarity = float(np.dot(current_embedding, self.reference_embedding))
        is_match = similarity >= target_thresh

        if not is_match:
            print(f"[VoiceAuth] Từ chối (cosine < {target_thresh:.2f})")

        return is_match, similarity

    def save_profile(self, embedding: np.ndarray, samples_count: int = 10) -> str:
        """Lưu vector giọng mẫu vào file my_voice.pkl."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        profile_data = {
            "version": "1.0",
            "embedding": embedding,
            "samples_count": samples_count,
            "created_at": time.time(),
        }

        # Lưu cả ở thư mục hiện tại lẫn get_data_path nếu khác nhau
        save_path = Path(self.PROFILE_FILENAME)
        with open(save_path, "wb") as f:
            pickle.dump(profile_data, f)

        data_path = Path(get_data_path(self.PROFILE_FILENAME, default_content=""))
        if data_path.resolve() != save_path.resolve():
            try:
                with open(data_path, "wb") as f:
                    pickle.dump(profile_data, f)
            except Exception:
                pass

        self.reference_embedding = embedding
        print(f"[VoiceAuth] ✓ Đã lưu hồ sơ giọng nói vào '{save_path.resolve()}'.")
        return str(save_path.resolve())


# Singleton instance toàn cục để dùng chung không tốn RAM
_GLOBAL_AUTHENTICATOR: Optional[VoiceAuthenticator] = None


def get_authenticator() -> VoiceAuthenticator:
    """Lấy thể hiện VoiceAuthenticator duy nhất."""
    global _GLOBAL_AUTHENTICATOR
    if _GLOBAL_AUTHENTICATOR is None:
        _GLOBAL_AUTHENTICATOR = VoiceAuthenticator()
    return _GLOBAL_AUTHENTICATOR


def verify_voice(
    audio_data: Union[bytes, np.ndarray],
    sample_rate: int = 16000,
    threshold: float = 0.80,
) -> Tuple[bool, float]:
    """
    Hàm tiện ích cấp module để gọi nhanh từ wake_word_engine.py.
    Trả về (is_match: bool, similarity: float).
    """
    auth = get_authenticator()
    return auth.verify_voice(audio_data, sample_rate=sample_rate, threshold=threshold)


def is_voice_profile_enrolled() -> bool:
    """Kiểm tra đã có tệp my_voice.pkl hợp lệ hay chưa."""
    auth = get_authenticator()
    return auth.is_enrolled()
