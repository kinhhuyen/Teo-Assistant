"""
Kiểm thử tự động các cải tiến cho bản v1.1.0:
1. Watchdog thay vì timeout tuyệt đối trong app_tray.py.
2. Timeout HTTP 5s cho Ollama trong brain.py.
3. Tách câu ghép với liên từ ' và ' trong Fast Path của brain.py.
4. Ngưỡng Wake Word: min_conf=0.90, min_rms=800.0, cooldown_sec=3.0.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
from brain import Brain
from wake_word_engine import WakeWordEngine


def test_wake_word_thresholds():
    """Kiểm tra các tham số mặc định của WakeWordEngine (Silero VAD + Duration Gate 2.1s + Google STT + Vosk Lazy)."""
    assert not (ROOT_DIR / "whisper_engine.py").exists(), "whisper_engine.py phải đã được gỡ bỏ hoàn toàn!"

    engine = WakeWordEngine(vosk_model_path="tests")
    assert engine.min_conf == 0.65, f"Expected min_conf 0.65, got {engine.min_conf}"
    assert engine.fuzzy_threshold == 70.0, f"Expected fuzzy_threshold 70.0, got {engine.fuzzy_threshold}"
    assert engine.min_rms == 180.0, f"Expected min_rms 180.0, got {engine.min_rms}"
    assert engine.dual_auth_min_rms == 300.0, f"Expected dual_auth_min_rms 300.0, got {engine.dual_auth_min_rms}"
    assert engine.RECORD_DURATION_SEC == 2.1, f"Expected RECORD_DURATION_SEC 2.1, got {engine.RECORD_DURATION_SEC}"
    assert engine.SILERO_START_PROB == 0.55, f"Expected SILERO_START_PROB 0.55, got {engine.SILERO_START_PROB}"
    assert engine.vad_mode == 1, f"Expected vad_mode 1, got {engine.vad_mode}"
    assert engine.silence_limit_sec == 0.65, f"Expected silence_limit_sec 0.65, got {engine.silence_limit_sec}"
    assert engine.cooldown_sec == 1.0, f"Expected cooldown_sec 1.0, got {engine.cooldown_sec}"
    assert engine.voice_auth_threshold == 0.65, f"Expected voice_auth_threshold 0.65, got {engine.voice_auth_threshold}"
    assert engine.min_cosine == 0.65, f"Expected min_cosine 0.65, got {engine.min_cosine}"
    assert engine.model is None, "Không được nạp Vosk Model lúc khởi động (phải Lazy-Load)!"
    assert engine._use_silero is True and engine.block_size == 512, "Phải ưu tiên Silero VAD (512 samples)"


def test_wake_word_condition_filtering():
    """Kiểm tra logic lọc điều kiện conf >= 0.65 và Fuzzy matching >= 70% cho từ khóa 'tèo đâu'."""
    engine = WakeWordEngine(vosk_model_path="tests")

    # 1. Khớp từ khóa chuẩn nhưng conf thấp (< 0.65) -> Phải TỪ CHỐI
    matched, _ = engine._check_wake_condition("tèo đâu", conf=0.55)
    assert matched is False, "Conf 0.55 phải bị từ chối vì ngưỡng tối thiểu là 0.65"

    # 2. Khớp từ khóa chuẩn với conf cao (>= 0.65) -> Phải CHẤP NHẬN
    matched, phrase = engine._check_wake_condition("tèo đâu", conf=0.65)
    assert matched is True, "Conf 0.65 với từ khóa 'tèo đâu' phải được chấp nhận"
    assert "teo" in phrase or "tèo" in phrase

    # 3. Fuzzy Matching: 'theo đó' (conf 0.66 >= 0.65, fuzzy >= 70%) -> Phải CHẤP NHẬN ứng viên
    matched_fuzzy, phrase_fuzzy = engine._check_wake_condition("theo đó", conf=0.66)
    assert matched_fuzzy is True, "'theo đó' (conf 0.66) phải được chấp nhận qua RapidFuzz >= 70%"
    assert "Fuzzy" in phrase_fuzzy or "theo" in phrase_fuzzy

    # 4. Tạp âm hoặc câu dài (> 3 từ) -> Phải TỪ CHỐI (Cấm tuyệt đối fuzzy match cho câu dài)
    matched, _ = engine._check_wake_condition("đi tìm cái gì đó", conf=0.99)
    assert matched is False, "Câu dài > 3 từ phải bị từ chối ngay cả khi conf cao"

    # 5. Từ ngắn khác hẳn (< 70% fuzzy) -> Phải TỪ CHỐI
    matched_diff, _ = engine._check_wake_condition("đi chơi đi", conf=0.90)
    assert matched_diff is False, "'đi chơi đi' không đủ 70% tương đồng với 'teo dau'"


def test_brain_fast_path_split_and():
    """Kiểm tra Fast Path tự động tách các câu lệnh ghép bằng liên từ ' và ' hoặc lặp động từ 'mở'."""
    brain = Brain()

    # 1. Ghép mở app và mở app với liên từ "và"
    actions = brain._fast_path("mở chrome và mở notepad")
    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == {"action": "open_app", "name": "chrome"}
    assert actions[1] == {"action": "open_app", "name": "notepad"}

    # 2. Ghép mở app lặp động từ không có liên từ: "mở powerpoint mở excel"
    actions2 = brain._fast_path("mở powerpoint mở excel")
    assert actions2 is not None
    assert len(actions2) == 2
    assert actions2[0] == {"action": "open_app", "name": "powerpoint"}
    assert actions2[1] == {"action": "open_app", "name": "excel"}

    # 3. Ghép 3 app: "mở chrome mở excel mở powerpoint"
    actions3 = brain._fast_path("mở chrome mở excel mở powerpoint")
    assert actions3 is not None
    assert len(actions3) == 3
    assert actions3[0] == {"action": "open_app", "name": "chrome"}
    assert actions3[1] == {"action": "open_app", "name": "excel"}
    assert actions3[2] == {"action": "open_app", "name": "powerpoint"}

    # 4. Ghép hệ thống: Wifi và Độ sáng
    actions = brain._fast_path("bật wifi và tăng độ sáng 20%")
    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == {"action": "wifi", "state": "on"}
    assert actions[1] == {"action": "brightness", "direction": "up", "step": 20}

    # 5. Ghép Chụp màn hình và Tắt máy
    actions = brain._fast_path("chụp màn hình và tắt máy")
    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == {"action": "screenshot"}
    assert actions[1] == {"action": "shutdown"}

    # 6. Ghép Mở app và Tìm kiếm web
    actions = brain._fast_path("mở máy tính và tìm kiếm giá vàng hôm nay")
    assert actions is not None
    assert len(actions) == 2
    assert actions[0] == {"action": "open_app", "name": "máy tính"}
    assert actions[1] == {"action": "search_web", "query": "giá vàng hôm nay"}


def test_brain_ollama_timeout_fallback(monkeypatch):
    """Kiểm tra xử lý Ollama tắt (socket check < 0.5s) & Timeout HTTP trong brain.py."""
    import requests

    def mock_post_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timed out after 5.0 seconds")

    monkeypatch.setattr(requests, "post", mock_post_timeout)

    # Kiểm tra khi Ollama tắt (port không chạy) phải thoát cực nhanh (< 0.5s)
    brain_closed = Brain(ollama_url="http://localhost:19999")
    t0 = time.time()
    res_closed = brain_closed._slow_path("một câu lệnh rất lạ không có trong danh sách chuẩn")
    elapsed_closed = time.time() - t0
    assert res_closed == [{"action": "unknown"}], f"Kỳ vọng fallback về unknown, nhận được: {res_closed}"
    assert elapsed_closed < 0.5, f"Slow Path khi Ollama tắt phải thoát < 0.5s, thực tế: {elapsed_closed:.3f}s"

    brain = Brain(ollama_url="http://localhost:11434")
    t1 = time.time()
    res = brain._slow_path("một câu lệnh rất lạ không có trong danh sách chuẩn")
    elapsed = time.time() - t1
    assert res == [{"action": "unknown"}], f"Kỳ vọng fallback về unknown, nhận được: {res}"
    assert elapsed < 0.5, f"Slow Path phải thoát < 0.5s khi Ollama chưa bật, thực tế: {elapsed:.3f}s"


def test_google_stt_cross_verification():
    """Kiểm tra logic Lớp 2/Lớp 3: Xác thực chéo bằng Voice Auth & Google STT."""
    # 1. Kiểm tra bộ lọc chuỗi _is_google_stt_matched
    assert WakeWordEngine._is_google_stt_matched("Tèo đâu") is True
    assert WakeWordEngine._is_google_stt_matched("tèo đâu") is True
    assert WakeWordEngine._is_google_stt_matched("teo dau") is True
    assert WakeWordEngine._is_google_stt_matched("tiền đâu") is True
    assert WakeWordEngine._is_google_stt_matched("theo đâu") is True
    assert WakeWordEngine._is_google_stt_matched("tèo đó") is True
    assert WakeWordEngine._is_google_stt_matched("tèo đâu mở chrome") is True
    assert WakeWordEngine._is_google_stt_matched("ê này tèo đâu") is True
    # Khẩu lệnh tự nhiên mới: "Tèo ơi", "Alo Tèo" / "A lô Tèo"
    assert WakeWordEngine._is_google_stt_matched("Tèo ơi") is True
    assert WakeWordEngine._is_google_stt_matched("tèo ơi mở chrome") is True
    assert WakeWordEngine._is_google_stt_matched("Alo Tèo") is True
    assert WakeWordEngine._is_google_stt_matched("a lô tèo") is True

    # 2. Blacklist và các từ không khớp (khi owner_cosine=0.0) -> PHẢI BÁC BỎ 100%
    assert WakeWordEngine._is_google_stt_matched("ơi tèo đâu rồi") is False  # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("mèo đâu") is False        # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("kèo đâu") is False        # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("heo đất") is False        # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("Heo Đất") is False        # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("ta ở đâu") is False       # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("kêu đâu") is False
    assert WakeWordEngine._is_google_stt_matched("mày đi đâu đấy") is False  # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("chào bạn hôm nay thế nào") is False
    assert WakeWordEngine._is_google_stt_matched("mở chrome lên đi") is False
    assert WakeWordEngine._is_google_stt_matched("") is False

    # 2b. Owner-Verified Phonetic Rescue (CHỈ kích hoạt khi owner_cosine >= 0.75)
    assert WakeWordEngine._is_google_stt_matched("Heo Đất", owner_cosine=0.82) is True
    assert WakeWordEngine._is_google_stt_matched("ta ở đâu", owner_cosine=0.82) is True
    assert WakeWordEngine._is_google_stt_matched("mèo đâu", owner_cosine=0.82) is True
    assert WakeWordEngine._is_google_stt_matched("Tèo ơi", owner_cosine=0.82) is True
    # Khi owner_cosine < 0.75 -> vẫn chặn tuyệt đối theo blacklist
    assert WakeWordEngine._is_google_stt_matched("Heo Đất", owner_cosine=0.70) is False
    assert WakeWordEngine._is_google_stt_matched("ta ở đâu", owner_cosine=0.60) is False
    assert WakeWordEngine._is_google_stt_matched("mày đi đâu đấy", owner_cosine=0.90) is False

    # 3. Kiểm tra Voice Auth (Lớp 2) & Google STT (Lớp 3)
    triggered = []
    engine = WakeWordEngine(
        vosk_model_path="tests",
        on_detected=lambda kw, score: triggered.append(kw),
        voice_auth_enabled=True,
        voice_verifier=lambda audio: (False, 0.40),  # Giả lập: Người lạ hoặc tiếng ồn (cosine 0.40 < 0.65)
        stt_verifier=lambda audio: "Tèo đâu",
    )
    mock_audio = b"\x00" * 32000

    # 3.1. Voice Auth từ chối (không phải chủ nhân) -> Phải BÁC BỎ NGAY LẬP TỨC
    res_stranger = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_stranger is False, "Khi Voice Auth từ chối, phải dừng ngay và không kích hoạt!"
    assert len(triggered) == 0, "Không được gọi on_detected khi Voice Auth từ chối"

    # 3.2. Đúng giọng chủ nhân (cosine 0.85 >= 0.65) nhưng Google STT nghe ra 'kêu đâu'
    engine.voice_verifier = lambda audio: (True, 0.85)
    engine.stt_verifier = lambda audio: "kêu đâu"
    res_wrong_word = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_wrong_word is False, "Khi Google STT trả về 'kêu đâu', phải bác bỏ"
    assert len(triggered) == 0

    # 3.3. Đúng giọng chủ nhân + Google STT xác nhận 'Tèo đâu' -> KÍCH HOẠT!
    engine.stt_verifier = lambda audio: "Tèo đâu"
    res_success = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_success is True, "Xác thực đúng giọng + Google STT xác nhận -> Phải kích hoạt!"
    assert len(triggered) == 1

    # 3.4. Đúng giọng chủ nhân + Google STT trả về rỗng '' (không lỗi mạng) -> Bỏ qua, KHÔNG nạp Vosk!
    engine.stt_verifier = lambda audio: ""
    res_empty = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_empty is False, "Google STT trả rỗng '' (không lỗi mạng) phải bỏ qua, không kích hoạt!"
    assert engine.model is None, "Không được nạp Vosk khi Google STT chỉ trả rỗng (không lỗi mạng)!"
    assert len(triggered) == 1

    # 3.5. Cơ chế Xác thực kép: Voice Auth từ chối (cosine 0.55 >= 0.65 - 0.10) nhưng Google STT xác nhận "Tèo đâu" và RMS > 300
    import numpy as np
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    loud_audio_samples = (1500.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    loud_audio_bytes = loud_audio_samples.tobytes()
    engine.voice_verifier = lambda audio: (False, 0.55)
    engine.stt_verifier = lambda audio: "Tèo đâu"
    res_dual = engine._handle_cross_verification(loud_audio_bytes, "tèo đâu", vosk_conf=0.90)
    assert res_dual is True, "Cơ chế Xác thực kép: Google STT xác nhận 'Tèo đâu' + RMS > 300 + cosine >= 0.55 phải kích hoạt!"
    assert len(triggered) == 2

    # 3.6. Người lạ hét to (cosine 0.40 < 0.55, RMS > 300) -> PHẢI TỪ CHỐI DualAuth
    engine.voice_verifier = lambda audio: (False, 0.40)
    res_stranger_loud = engine._handle_cross_verification(loud_audio_bytes, "tèo đâu", vosk_conf=0.90)
    assert res_stranger_loud is False, "Người lạ hét to (cosine < threshold - 0.10) phải bị từ chối!"
    assert len(triggered) == 2

    # 3.7. Tắt Voice Auth (voice_auth_enabled=False) -> Bỏ qua Lớp 2 đi thẳng sang Lớp 3
    engine.voice_auth_enabled = False
    res_no_va = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_no_va is True, "Khi tắt voice_auth_enabled, phải bỏ qua Lớp 2 và chấp nhận qua Google STT!"
    assert len(triggered) == 3

    # 3.8. Kiểm tra Duration Gate 2.1s chặn tiếng hát / nói chuyện dài ngay tại Lớp 1
    va_calls = []
    stt_calls = []
    gate_engine = WakeWordEngine(
        vosk_model_path="tests",
        cooldown_sec=0.0,
        voice_auth_enabled=True,
        voice_verifier=lambda audio: (va_calls.append(1) or True, 0.90),
        stt_verifier=lambda audio: (stt_calls.append(1) or "Tèo đâu"),
    )
    loud_block = (2000.0 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.032, 512, dtype=np.float32))).astype(np.int16).tobytes()

    class _FakeLongSingingStream:
        active = True
        stopped = False
        def __init__(self, eng_ref):
            self.eng_ref = eng_ref
            self.reads = 0
        def read(self, n):
            self.reads += 1
            if self.reads > 75:  # > 2.4s liên tục -> dừng vòng lặp sau khi Duration Gate đã chạy
                self.eng_ref._running = False
                self.eng_ref.is_listening = False
            return loud_block, False
        def stop(self):
            pass
        def close(self):
            pass

    gate_engine._get_silero_prob = lambda pcm: 0.90  # Giả lập tiếng hát/nói liên tục không ngắt
    gate_engine._running = True
    gate_engine.is_listening = True
    gate_engine.stream = _FakeLongSingingStream(gate_engine)
    gate_engine._listen_loop()
    assert len(va_calls) == 0, "Duration Gate 2.1s phải chặn tại Lớp 1, KHÔNG gọi Voice Auth!"
    assert len(stt_calls) == 0, "Duration Gate 2.1s phải chặn tại Lớp 1, KHÔNG gọi Google STT!"


def test_memory_cache_verb_compatibility():
    """Kiểm tra Memory Cache chặn đứng xung đột động từ Mở vs Tắt và Tăng vs Giảm."""
    brain = Brain()

    # 1. Câu lệnh 'tắt' nhưng cache chứa 'open_app' -> Phải BÁC BỎ
    compatible = brain._is_cache_action_compatible(
        "tắt powerpoint",
        [{"action": "open_app", "name": "powerpoint"}],
        matched_key="mở powerpoint"
    )
    assert compatible is False, "Xung đột 'tắt' và 'open_app' phải bị từ chối"

    # 2. Câu lệnh 'mở' nhưng cache chứa 'close_app' -> Phải BÁC BỎ
    compatible2 = brain._is_cache_action_compatible(
        "mở powerpoint",
        [{"action": "close_app", "name": "powerpoint"}],
        matched_key="tắt powerpoint"
    )
    assert compatible2 is False, "Xung đột 'mở' và 'close_app' phải bị từ chối"

    # 3. Đúng động từ -> Phải CHẤP NHẬN
    compatible3 = brain._is_cache_action_compatible(
        "tắt powerpoint",
        [{"action": "close_app", "name": "powerpoint"}],
        matched_key="tắt powerpoint"
    )
    assert compatible3 is True, "Cùng động từ tắt phải được chấp nhận"

    # 4. Tăng vs Giảm -> Phải BÁC BỎ
    comp_vol = brain._is_cache_action_compatible(
        "tăng âm lượng",
        [{"action": "volume", "direction": "down", "step": 10}],
        matched_key="giảm âm lượng"
    )
    assert comp_vol is False, "Xung đột 'tăng' và direction 'down' phải bị từ chối"

    comp_bright = brain._is_cache_action_compatible(
        "giảm độ sáng",
        [{"action": "brightness", "direction": "up", "step": 10}],
        matched_key="tăng độ sáng"
    )
    assert comp_bright is False, "Xung đột 'giảm' và direction 'up' phải bị từ chối"

    # 5. Xung đột loại: Độ sáng vs Âm lượng -> Phải BÁC BỎ
    comp_type = brain._is_cache_action_compatible(
        "tăng độ sáng",
        [{"action": "volume", "direction": "up", "step": 10}],
        matched_key="tăng âm lượng"
    )
    assert comp_type is False, "Xung đột độ sáng và volume phải bị từ chối"


def test_memory_cache_number_bypass():
    """Kiểm tra câu lệnh chứa số đo/phần trăm KHÔNG được lấy từ Cache và KHÔNG được ghi vào Cache."""
    brain = Brain()

    # Chạy lệnh âm lượng có số
    res1 = brain.process("đặt âm lượng 50%")
    assert res1 == [{"action": "volume", "percentage": 50}]
    assert brain.memory.get("đặt âm lượng 50%") is None, "Lệnh có số đo không được ghi vào cache"

    # Chạy lệnh âm lượng có số khác
    res2 = brain.process("đặt âm lượng 80%")
    assert res2 == [{"action": "volume", "percentage": 80}]
    assert brain.memory.get("đặt âm lượng 80%") is None

    # Chạy lệnh độ sáng có số
    res3 = brain.process("tăng độ sáng 20%")
    assert res3 == [{"action": "brightness", "direction": "up", "step": 20}]
    assert brain.memory.get("tăng độ sáng 20%") is None


def test_stt_timeout_fallback():
    """Kiểm tra STT không nạp Vosk khi khởi động/Online và timeout Google STT."""
    from stt_engine import SpeechToText
    stt = SpeechToText()
    assert stt.vosk_model is None, "SpeechToText KHÔNG được nạp Vosk lúc khởi động!"

    # Kiểm tra khi Online + Google trả về kết quả -> KHÔNG nạp Vosk
    orig_online = stt.is_online
    orig_google = stt.recognize_google
    try:
        stt.is_online = lambda: True
        stt.recognize_google = lambda audio, timeout=2.5: "mở chrome"
        out = stt.process_audio(b"\x01\x00" * 4000)
        assert out == "mở chrome"
        assert stt.vosk_model is None, "Không được nạp Vosk khi Online thành công!"

        # Google trả rỗng '' -> trả '' và KHÔNG nạp Vosk
        stt.recognize_google = lambda audio, timeout=2.5: ""
        out_empty = stt.process_audio(b"\x01\x00" * 4000)
        assert out_empty == ""
        assert stt.vosk_model is None, "Không được nạp Vosk khi Google trả rỗng ''!"
    finally:
        stt.is_online = orig_online
        stt.recognize_google = orig_google

    # Giả lập Google STT bị delay > 3s (mất mạng/lag)
    def mock_hang_recognize_google(*args, **kwargs):
        time.sleep(3.5)
        return "câu lệnh bị lag"

    stt.recognizer.recognize_google = mock_hang_recognize_google
    try:
        # Gọi recognize_google với timeout=1.0s phải ném TimeoutError
        stt.recognize_google(b"\x00" * 3200, timeout=1.0)
        assert False, "Kỳ vọng TimeoutError sau khi quá thời gian timeout!"
    except TimeoutError as te:
        assert "Google STT phản hồi quá" in str(te)


def test_close_app_improvements():
    """Kiểm tra cải tiến đóng/tắt app thông minh bằng psutil, Fast Path và Blacklist an toàn."""
    from actions import ActionExecutor
    brain = Brain()
    executor = ActionExecutor()

    # 1. Kiểm tra từ điển UWP có đầy đủ
    assert "microsoft store" in executor.PROCESS_NAME_MAP
    assert "calculator" in executor.PROCESS_NAME_MAP

    # 2. Kiểm tra Fast Path lệnh tắt/đóng app
    actions = brain._fast_path("tắt powerpoint")
    assert actions == [{"action": "close_app", "name": "powerpoint"}]

    actions = brain._fast_path("đóng word")
    assert actions == [{"action": "close_app", "name": "word"}]

    actions = brain._fast_path("thoát excel")
    assert actions == [{"action": "close_app", "name": "excel"}]

    actions = brain._fast_path("tắt microsoft store")
    assert actions == [{"action": "close_app", "name": "microsoft store"}]

    # 3. Kiểm tra tách đa lệnh tắt
    actions = brain._fast_path("tắt chrome tắt notepad")
    assert actions == [{"action": "close_app", "name": "chrome"}, {"action": "close_app", "name": "notepad"}]

    # 4. Kiểm tra bảo vệ hệ thống: Blacklist các tiến trình nguy hiểm
    for danger in ["explorer", "explorer.exe", "svchost", "svchost.exe", "system", "registry", "csrss", "winlogon"]:
        act_res = brain.process(f"tắt {danger}")
        assert act_res == [{"action": "unknown"}], f"Tiến trình nguy hiểm '{danger}' phải bị chặn!"

    # 5. Kiểm tra ActionExecutor.close_app không quăng lỗi khi app không chạy
    res = executor.close_app("non_existent_fake_app_xyz")
    assert res == "App không chạy"


def test_voice_auth_offline():
    """Kiểm tra mô hình sinh trắc học Resemblyzer và so khớp Cosine Similarity (Offline 100%)."""
    import numpy as np
    from voice_auth import VoiceAuthenticator

    # 1. Kiểm tra cơ chế First-Run (người dùng mới, chưa có profile) -> Tự động trả về True (bỏ qua xác thực)
    auth_first_run = VoiceAuthenticator(profile_path="tests/non_existent_profile_xyz.pkl")
    dummy_wav = np.zeros(16000, dtype=np.float32)
    matched_new, sim_new = auth_first_run.verify_voice(dummy_wav)
    assert matched_new is True, "Người dùng mới chưa có my_voice.pkl phải tự động được chấp nhận"
    assert sim_new == 1.0

    # 2. Kiểm tra khi đã đăng ký profile giọng nói
    auth = VoiceAuthenticator(profile_path="tests/mock_voice.pkl")
    assert auth.encoder is not None, "VoiceEncoder phải được nạp thành công"

    # Tạo 2 mẫu âm thanh giả lập với phổ tần số khác biệt
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    sample_a = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    sample_b = (0.5 * np.sin(2 * np.pi * 800 * t) + 0.3 * np.sin(2 * np.pi * 1600 * t)).astype(np.float32)

    emb_a = auth.compute_embedding(sample_a, sample_rate=16000)
    assert emb_a is not None
    assert emb_a.shape == (256,)
    assert abs(np.linalg.norm(emb_a) - 1.0) < 1e-4

    # Gán emb_a làm reference_embedding của chủ nhân
    auth.reference_embedding = emb_a

    # So sánh chính mẫu của chủ nhân (self-similarity) -> Phải >= 0.65
    matched, sim = auth.verify_voice(sample_a, sample_rate=16000, threshold=0.65)
    assert matched is True
    assert sim >= 0.95, f"Self-similarity phải gần 1.0, nhận được: {sim}"

    # So sánh với mẫu âm thanh của người lạ (sample_b) -> Phải < 0.65
    matched_diff, sim_diff = auth.verify_voice(sample_b, sample_rate=16000, threshold=0.65)
    assert matched_diff is False, f"Giọng người lạ không được vượt qua ngưỡng 0.65 (nhận {sim_diff})"
    assert sim_diff < 0.65, f"Similarity của giọng lạ phải < 0.65, nhận được: {sim_diff}"

    # 3. Kiểm tra lưu & tải đa mẫu (sample_embeddings)
    tmp_profile = Path("tests/tmp_multi_sample_voice.pkl")
    try:
        auth_multi = VoiceAuthenticator(profile_path=str(tmp_profile))
        auth_multi.save_profile(emb_a, samples_count=5, sample_embeddings=[emb_a, emb_a])
        auth_reload = VoiceAuthenticator(profile_path=str(tmp_profile))
        assert auth_reload.is_enrolled() is True
        assert len(auth_reload.sample_embeddings) == 2
        m_ok, s_ok = auth_reload.verify_voice(sample_a, sample_rate=16000)
        assert m_ok is True and s_ok >= 0.95
    finally:
        if tmp_profile.exists():
            tmp_profile.unlink()


if __name__ == "__main__":
    print("--- [TEST 1] Wake Word Thresholds ---")
    test_wake_word_thresholds()
    print("✓ Passed!")

    print("--- [TEST 2] Wake Word Condition Filtering ---")
    test_wake_word_condition_filtering()
    print("✓ Passed!")

    print("--- [TEST 3] Google STT & Voice Auth Cross-Verification ---")
    test_google_stt_cross_verification()
    print("✓ Passed!")

    print("--- [TEST 4] Brain Fast Path Split 'và' ---")
    test_brain_fast_path_split_and()
    print("✓ Passed!")

    print("--- [TEST 5] Brain Ollama HTTP Timeout Fallback ---")
    class MonkeyPatch:
        def setattr(self, obj, attr, val):
            setattr(obj, attr, val)
    test_brain_ollama_timeout_fallback(MonkeyPatch())
    print("✓ Passed!")

    print("--- [TEST 6] Smart Close App & System Blacklist Guard ---")
    test_close_app_improvements()
    print("✓ Passed!")

    print("--- [TEST 7] Memory Cache Verb Compatibility (Mở vs Tắt & Tăng vs Giảm) ---")
    test_memory_cache_verb_compatibility()
    print("✓ Passed!")

    print("--- [TEST 8] Memory Cache Bypass For Number & Measurements ---")
    test_memory_cache_number_bypass()
    print("✓ Passed!")

    print("--- [TEST 9] STT Google Timeout (2s) & Fallback ---")
    test_stt_timeout_fallback()
    print("✓ Passed!")

    print("--- [TEST 10] Voice Auth Resemblyzer (Offline 100%) ---")
    test_voice_auth_offline()
    print("✓ Passed!")

    print("\n🎉 TOÀN BỘ 10 BÀI TEST ĐỀU THÀNH CÔNG 100%!")
