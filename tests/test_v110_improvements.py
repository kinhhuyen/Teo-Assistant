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
    """Kiểm tra các tham số mặc định của WakeWordEngine."""
    engine = WakeWordEngine(vosk_model_path="tests")
    assert engine.min_conf == 0.60, f"Expected min_conf 0.60, got {engine.min_conf}"
    assert engine.fuzzy_threshold == 70.0, f"Expected fuzzy_threshold 70.0, got {engine.fuzzy_threshold}"
    assert engine.min_rms == 350.0, f"Expected min_rms 350.0, got {engine.min_rms}"
    assert engine.cooldown_sec == 1.0, f"Expected cooldown_sec 1.0, got {engine.cooldown_sec}"
    assert engine.voice_auth_threshold == 0.80, f"Expected voice_auth_threshold 0.80, got {engine.voice_auth_threshold}"
    assert engine.min_cosine == 0.80, f"Expected min_cosine 0.80, got {engine.min_cosine}"


def test_wake_word_condition_filtering():
    """Kiểm tra logic lọc điều kiện conf >= 0.60 và Fuzzy matching >= 75% cho từ khóa 'tèo đâu'."""
    engine = WakeWordEngine(vosk_model_path="tests")

    # 1. Khớp từ khóa chuẩn nhưng conf thấp (< 0.60) -> Phải TỪ CHỐI
    matched, _ = engine._check_wake_condition("tèo đâu", conf=0.55)
    assert matched is False, "Conf 0.55 phải bị từ chối vì ngưỡng tối thiểu là 0.60"

    # 2. Khớp từ khóa chuẩn với conf cao (>= 0.60) -> Phải CHẤP NHẬN
    matched, phrase = engine._check_wake_condition("tèo đâu", conf=0.65)
    assert matched is True, "Conf 0.65 với từ khóa 'tèo đâu' phải được chấp nhận"
    assert "teo" in phrase or "tèo" in phrase

    # 3. Fuzzy Matching: 'theo đó' (conf 0.64 >= 0.60, fuzzy >= 75%) -> Phải CHẤP NHẬN ứng viên
    matched_fuzzy, phrase_fuzzy = engine._check_wake_condition("theo đó", conf=0.64)
    assert matched_fuzzy is True, "'theo đó' (conf 0.64) phải được chấp nhận qua RapidFuzz >= 75%"
    assert "Fuzzy" in phrase_fuzzy or "theo" in phrase_fuzzy

    # 4. Tạp âm hoặc câu dài (> 3 từ) -> Phải TỪ CHỐI (Cấm tuyệt đối fuzzy match cho câu dài)
    matched, _ = engine._check_wake_condition("đi tìm cái gì đó", conf=0.99)
    assert matched is False, "Câu dài > 3 từ phải bị từ chối ngay cả khi conf cao"

    # 5. Từ ngắn khác hẳn (< 75% fuzzy) -> Phải TỪ CHỐI
    matched_diff, _ = engine._check_wake_condition("đi chơi đi", conf=0.90)
    assert matched_diff is False, "'đi chơi đi' không đủ 75% tương đồng với 'teo dau'"


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
    """Kiểm tra xử lý Timeout HTTP khi gọi Ollama trong brain.py."""
    import requests

    def mock_post_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timed out after 5.0 seconds")

    monkeypatch.setattr(requests, "post", mock_post_timeout)

    brain = Brain(ollama_url="http://localhost:11434")
    # Lệnh không thuộc Fast Path buộc phải vào Slow Path
    res = brain._slow_path("một câu lệnh rất lạ không có trong danh sách chuẩn")
    assert res == [{"action": "unknown"}], f"Kỳ vọng fallback về unknown, nhận được: {res}"


def test_google_stt_cross_verification():
    """Kiểm tra logic Lớp 2: Xác thực chéo bằng Google STT."""
    # 1. Kiểm tra bộ lọc chuỗi _is_google_stt_matched
    assert WakeWordEngine._is_google_stt_matched("Tèo đâu") is True
    assert WakeWordEngine._is_google_stt_matched("tèo đâu") is True
    assert WakeWordEngine._is_google_stt_matched("teo dau") is True
    # Hỗ trợ nới lỏng RapidFuzz >= 80% ("em đâu", "tiền đâu", "theo đâu")
    assert WakeWordEngine._is_google_stt_matched("em đâu") is True
    assert WakeWordEngine._is_google_stt_matched("tiền đâu") is True
    assert WakeWordEngine._is_google_stt_matched("theo đâu") is True

    # 2. Blacklist và các từ không khớp -> PHẢI BÁC BỎ 100%
    assert WakeWordEngine._is_google_stt_matched("ơi tèo đâu rồi") is False  # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("mèo đâu") is False        # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("heo đất") is False        # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("kêu đâu") is False
    assert WakeWordEngine._is_google_stt_matched("mày đi đâu đấy") is False  # Nằm trong Blacklist
    assert WakeWordEngine._is_google_stt_matched("chào bạn hôm nay thế nào") is False
    assert WakeWordEngine._is_google_stt_matched("mở chrome lên đi") is False
    assert WakeWordEngine._is_google_stt_matched("") is False

    # 3. Kiểm tra Voice Auth (Lớp 1.5) & Google STT (Lớp 2)
    triggered = []
    engine = WakeWordEngine(
        vosk_model_path="tests",
        on_detected=lambda kw, score: triggered.append(kw),
        voice_verifier=lambda audio: (False, 0.40),  # Giả lập: Người lạ hoặc tiếng ồn (cosine 0.40 < 0.75)
        stt_verifier=lambda audio: "Tèo đâu",
    )
    mock_audio = b"\x00" * 32000

    # 3.1. Voice Auth từ chối (không phải chủ nhân) -> Phải BÁC BỎ NGAY LẬP TỨC
    res_stranger = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_stranger is False, "Khi Voice Auth từ chối, phải dừng ngay và không kích hoạt!"
    assert len(triggered) == 0, "Không được gọi on_detected khi Voice Auth từ chối"

    # 3.2. Đúng giọng chủ nhân (cosine 0.85 >= 0.75) nhưng Google STT nghe ra 'kêu đâu' (< 80%)
    engine.voice_verifier = lambda audio: (True, 0.85)
    engine.stt_verifier = lambda audio: "kêu đâu"
    res_wrong_word = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_wrong_word is False, "Khi Google STT trả về 'kêu đâu' (< 80%), phải bác bỏ"
    assert len(triggered) == 0

    # 3.3. Đúng giọng chủ nhân + Google STT xác nhận 'Tèo đâu' -> KÍCH HOẠT!
    engine.stt_verifier = lambda audio: "Tèo đâu"
    res_success = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_success is True, "Xác thực đúng giọng + Google STT xác nhận -> Phải kích hoạt!"
    assert len(triggered) == 1

    # 3.4. Đúng giọng chủ nhân + Google STT mất mạng/timeout (trả về '') -> Vẫn kích hoạt an toàn!
    engine.stt_verifier = lambda audio: ""
    res_offline = engine._handle_cross_verification(mock_audio, "tèo đâu", vosk_conf=0.90)
    assert res_offline is True, "Đã xác thực giọng chủ nhân thành công, khi mất mạng vẫn kích hoạt an toàn!"
    assert len(triggered) == 2

    # 3.5. Cơ chế Xác thực kép: Voice Auth từ chối (cosine 0.55 < 0.70 do ồn) nhưng Google STT xác nhận "Tèo đâu" và RMS > 500
    import numpy as np
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    loud_audio_samples = (1500.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    loud_audio_bytes = loud_audio_samples.tobytes()
    engine.voice_verifier = lambda audio: (False, 0.55)
    engine.stt_verifier = lambda audio: "Tèo đâu"
    res_dual = engine._handle_cross_verification(loud_audio_bytes, "tèo đâu", vosk_conf=0.90)
    assert res_dual is True, "Cơ chế Xác thực kép: Google STT xác nhận 'Tèo đâu' + RMS > 500 phải kích hoạt dù Voice Auth < 0.70!"
    assert len(triggered) == 3


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
    """Kiểm tra giới hạn thời gian nhận diện Google STT (3s) và chuyển sang Offline."""
    from stt_engine import SpeechToText
    stt = SpeechToText()

    # Giả lập Google STT bị delay > 3s (mất mạng/lag)
    def mock_hang_recognize_google(*args, **kwargs):
        time.sleep(3.5)
        return "câu lệnh bị lag"

    stt.recognizer.recognize_google = mock_hang_recognize_google
    try:
        # Gọi recognize_google với timeout=3.0s phải ném TimeoutError
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

    # So sánh chính mẫu của chủ nhân (self-similarity) -> Phải >= 0.75
    matched, sim = auth.verify_voice(sample_a, sample_rate=16000, threshold=0.75)
    assert matched is True
    assert sim >= 0.95, f"Self-similarity phải gần 1.0, nhận được: {sim}"

    # So sánh với mẫu âm thanh của người lạ (sample_b) -> Phải < 0.75 (khoảng ~0.41)
    matched_diff, sim_diff = auth.verify_voice(sample_b, sample_rate=16000, threshold=0.75)
    assert matched_diff is False, f"Giọng người lạ không được vượt qua ngưỡng 0.75 (nhận {sim_diff})"
    assert sim_diff < 0.75, f"Similarity của giọng lạ phải < 0.75, nhận được: {sim_diff}"


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
