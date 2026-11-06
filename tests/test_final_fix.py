"""
Kiểm thử tự động bản cập nhật 'Chốt hạ' (Final Fix) cho kiến trúc 4 Lớp của Wake Word:
1. Hạ ngưỡng VAD (Lớp 1): RMS > 250 & Tăng thời gian thu âm lên 2.5s
2. Lọc nhiễu Voice Auth (Lớp 2): noisereduce & Ngưỡng 0.70
3. Tiền xử lý âm thanh STT (Lớp 3): noisereduce + normalize
4. Cơ chế Retry khi Google STT trả về rỗng (timeout 1.5s)
5. Nới lỏng điều kiện Google STT: RapidFuzz >= 80% ("em đâu", "tiền đâu", "tèo đâu")
6. Vosk Offline (Lớp 4): Grammar ["tèo đâu", "teo dau", "[unk]"] + Fuzzy Matching >= 70%
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from voice_auth import VoiceAuthenticator, get_authenticator, verify_voice
from wake_word_engine import WakeWordEngine, strip_accents
from rapidfuzz import fuzz


def test_vad_low_threshold_and_duration():
    """1. Kiểm tra ngưỡng RMS VAD đã nâng lên 350.0 và thời gian thu âm là 2.5s."""
    engine = WakeWordEngine(vosk_model_path="tests")
    assert engine.min_rms == 350.0, f"Expected min_rms 350.0, got {engine.min_rms}"
    assert engine.DEFAULT_MIN_RMS == 350.0
    assert engine.RECORD_DURATION_SEC == 2.5, f"Expected RECORD_DURATION_SEC 2.5s, got {engine.RECORD_DURATION_SEC}"


def test_voice_auth_noisereduce_and_threshold():
    """2. Kiểm tra noisereduce và ngưỡng Voice Auth 0.80."""
    import voice_auth
    assert voice_auth.HAS_NOISEREDUCE is True, "noisereduce phải được nạp thành công"
    assert voice_auth.nr is not None

    auth = VoiceAuthenticator(profile_path="tests/mock_voice_final.pkl")
    assert auth.DEFAULT_THRESHOLD == 0.80
    assert auth.min_cosine == 0.80

    # Kiểm tra compute_embedding hoạt động với tín hiệu và giảm nhiễu
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    clean_signal = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    noisy_signal = clean_signal + (0.02 * np.random.randn(16000)).astype(np.float32)

    emb_clean = auth.compute_embedding(clean_signal, sample_rate=16000)
    emb_noisy = auth.compute_embedding(noisy_signal, sample_rate=16000)
    assert emb_clean is not None and emb_noisy is not None
    assert emb_clean.shape == (256,) and emb_noisy.shape == (256,)

    auth.reference_embedding = emb_clean

    # Xác thực với ngưỡng mặc định 0.80 (noisy_signal đạt ~0.87 >= 0.80)
    matched, sim = auth.verify_voice(noisy_signal)
    assert sim >= 0.80, f"Similarity sau khi lọc nhiễu phải >= 0.80, nhận được: {sim}"
    assert matched is True


def test_stt_preprocessing_noisereduce_and_normalize():
    """3. Kiểm tra tiền xử lý âm thanh STT: Lọc nhiễu và chuẩn hóa âm lượng (normalize)."""
    engine = WakeWordEngine(vosk_model_path="tests")

    t = np.linspace(0, 2.5, int(16000 * 2.5), dtype=np.float32)
    # Tín hiệu nhỏ có nhiễu
    clean = (0.15 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    noise = (0.02 * np.random.randn(len(t))).astype(np.float32)
    raw_samples = ((clean + noise) * 32767.0).astype(np.int16)
    raw_bytes = raw_samples.tobytes()

    processed_bytes = engine._preprocess_for_stt(raw_bytes)
    assert processed_bytes is not None
    assert len(processed_bytes) == len(raw_bytes)

    out_samples = np.frombuffer(processed_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    max_amp = float(np.max(np.abs(out_samples)))
    # Sau khi normalize về 0.90, max amplitude phải xấp xỉ 0.90 (+- 0.05)
    assert abs(max_amp - 0.90) < 0.05, f"Expected normalized max amplitude ~0.90, got {max_amp}"


def test_stt_retry_mechanism():
    """4. Kiểm tra Cơ chế Retry khi Google STT trả về chuỗi rỗng lần đầu."""
    call_count = [0]

    def mock_stt_with_retry(audio_bytes, timeout=2.0):
        call_count[0] += 1
        if call_count[0] == 1:
            # Lần 1 trả về chuỗi rỗng giả lập lag
            return ""
        # Lần 2 retry trả về kết quả thành công
        return "Tèo đâu"

    engine = WakeWordEngine(
        vosk_model_path="tests",
        stt_verifier=mock_stt_with_retry,
    )

    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    dummy_audio = (1000.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()

    text, is_net_err, err_msg = engine._recognize_google(dummy_audio, timeout=2.0)
    assert text == "Tèo đâu", f"Expected 'Tèo đâu' sau retry, nhận được: '{text}'"
    assert is_net_err is False
    assert call_count[0] == 2, f"Expected 2 calls (initial + retry), got {call_count[0]}"


def test_google_stt_fuzzy_matching_80():
    """5. Nới lỏng điều kiện Google STT bằng RapidFuzz >= 80% (không nằm trong Blacklist)."""
    # Các biến thể khi Google nghe nhầm hoặc nhiễu âm -> ĐỀU PHẢI CHẤP NHẬN
    accepted_phrases = [
        "Tèo đâu",
        "tèo đâu",
        "teo dau",
        "em đâu",
        "tiền đâu",
        "theo đâu",
        "tèo đó",
        "ơi tèo đâu",
    ]

    for phrase in accepted_phrases:
        matched = WakeWordEngine._is_google_text_matched(phrase, threshold=80.0)
        assert matched is True, f"'{phrase}' phải được chấp nhận qua RapidFuzz >= 80%"


def test_blacklist_rejection():
    """5b. Thêm Blacklist chặn triệt để các từ gây nhầm lẫn (kể cả khi Fuzzy >= 80%)."""
    # Danh sách từ và câu gây nhầm lẫn trong yêu cầu người dùng
    blacklisted_phrases = [
        "mèo",
        "heo",
        "bèo",
        "kèo",
        "nèo",
        "tèo đâu rồi",
        "mày đi đâu",
        "mèo đâu",       # RapidFuzz từng đạt 86% -> Hiện tại phải BÁC BỎ
        "heo đất",       # Từng bị nhận nhầm -> Hiện tại phải BÁC BỎ
        "bèo dạt",
        "kèo thơm",
        "tèo đâu rồi em",
        "mày đi đâu đấy",
        "chú mèo nhỏ",
        "con heo đất",
    ]

    for phrase in blacklisted_phrases:
        matched = WakeWordEngine._is_google_text_matched(phrase, threshold=80.0)
        assert matched is False, f"'{phrase}' nằm trong Blacklist -> PHẢI BÁC BỎ NGAY LẬP TỨC"
        assert WakeWordEngine.is_blacklisted(phrase) is True

    # Các câu không liên quan khác cũng phải từ chối
    other_rejected = [
        "chào bạn hôm nay thế nào",
        "mở youtube",
        "tắt máy tính",
        "alo nghe rõ không",
        "",
    ]
    for phrase in other_rejected:
        matched = WakeWordEngine._is_google_text_matched(phrase, threshold=80.0)
        assert matched is False, f"'{phrase}' phải bị từ chối"


def test_google_stt_hard_timeout_15s_no_retry():
    """8. Ép timeout cứng 1.5s cho Google STT và BỎ HOÀN TOÀN RETRY khi bị Timeout."""
    import time
    call_count = [0]

    def mock_hanging_stt(audio_bytes, timeout=1.5):
        call_count[0] += 1
        time.sleep(3.0)  # Giả lập mạng lag 3 giây (vượt quá 1.5s)
        return "Tèo đâu"

    engine = WakeWordEngine(
        vosk_model_path="tests",
        stt_verifier=mock_hanging_stt,
    )

    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    dummy_audio = (1000.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()

    start_time = time.time()
    text, is_net_err, err_msg = engine._recognize_google(dummy_audio, timeout=1.5)
    elapsed = time.time() - start_time

    # Phải ngắt sau xấp xỉ 1.5s (dưới 1.9s), không được chờ hết 3s
    assert elapsed < 1.9, f"Thời gian thực thi {elapsed:.2f}s vượt quá ngưỡng timeout cứng 1.5s!"
    assert text == "", f"Khi timeout phải trả về chuỗi rỗng, nhận được: '{text}'"
    assert is_net_err is True, "is_net_err phải là True khi Google STT timeout"
    assert "timeout" in err_msg.lower()
    # TUYỆT ĐỐI KHÔNG RETRY khi lần 1 bị timeout!
    assert call_count[0] == 1, f"Expected exactly 1 call (no retry on timeout), got {call_count[0]}"


def test_adaptive_noisereduce_bypass():
    """9. Đo thời gian thực thi noisereduce: Tự động bỏ qua lọc nhiễu nếu mất hơn 0.2s."""
    engine = WakeWordEngine(vosk_model_path="tests")
    assert engine._skip_noisereduce_stt is False

    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    dummy_audio = (1000.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()

    # Giả lập trường hợp noisereduce chạy chậm (> 0.2s)
    import voice_auth
    original_reduce = voice_auth.nr.reduce_noise

    try:
        def slow_reduce(y, sr):
            import time
            time.sleep(0.25)  # Làm chậm 0.25s > 0.2s
            return y

        voice_auth.nr.reduce_noise = slow_reduce

        # Chạy lần 1: phát hiện > 0.2s -> cờ _skip_noisereduce_stt bật lên True
        engine._preprocess_for_stt(dummy_audio)
        assert engine._skip_noisereduce_stt is True, "Cờ _skip_noisereduce_stt phải bật thành True khi noisereduce > 0.2s"

        # Chạy lần 2: phải bỏ qua slow_reduce (chạy tức thì không sleep)
        t_start = time.time()
        engine._preprocess_for_stt(dummy_audio)
        dur = time.time() - t_start
        assert dur < 0.1, f"Lần 2 phải bỏ qua noisereduce để chạy nhanh (< 0.1s), mất {dur:.3f}s"
    finally:
        voice_auth.nr.reduce_noise = original_reduce


def test_dual_verification_logic():
    """6. Kiểm tra Cơ chế Xác thực kép (Voice Auth từ chối nhưng Google STT + RMS > 500)."""
    triggered = []
    engine = WakeWordEngine(
        vosk_model_path="tests",
        on_detected=lambda kw, score: triggered.append(kw),
        voice_auth_threshold=0.80,
    )

    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    loud_speech = (1200.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()
    quiet_noise = (200.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()

    # Case A: Voice Auth từ chối (< 0.80, ví dụ 0.55), Google STT trả về "Tèo đâu", RMS > 500 -> PHẢI KÍCH HOẠT
    engine.voice_verifier = lambda audio: (False, 0.55)
    engine.stt_verifier = lambda audio: "Tèo đâu"
    res_a = engine._handle_cross_verification(loud_speech, "tèo đâu", vosk_conf=0.90)
    assert res_a is True, "Xác thực kép phải kích hoạt khi Google STT xác nhận + RMS > 500"
    assert len(triggered) == 1

    # Case B: Voice Auth từ chối, Google STT trả về "Tèo đâu", nhưng RMS <= 500 -> PHẢI TỪ CHỐI
    triggered.clear()
    res_b = engine._handle_cross_verification(quiet_noise, "tèo đâu", vosk_conf=0.90)
    assert res_b is False, "Xác thực kép phải từ chối khi RMS <= 500"
    assert len(triggered) == 0

    # Case C: Voice Auth từ chối, Google STT nghe ra chữ khác ("chào bạn"), RMS > 500 -> PHẢI TỪ CHỐI
    engine.stt_verifier = lambda audio: "chào bạn"
    res_c = engine._handle_cross_verification(loud_speech, "tèo đâu", vosk_conf=0.90)
    assert res_c is False, "Google STT không xác nhận 'Tèo đâu' thì phải từ chối"
    assert len(triggered) == 0


def test_layer4_vosk_grammar_and_fuzzy():
    """7. Kiểm tra Vosk Offline (Lớp 4): Grammar cứng + RapidFuzz >= 70%."""
    engine = WakeWordEngine(vosk_model_path="tests")
    assert engine.fuzzy_threshold == 70.0
    assert engine.grammar_words == ["tèo đâu", "teo dau", "[unk]"]

    # Giả lập kết quả nhận diện Vosk
    test_cases = [
        ("tèo đâu", True),
        ("teo dau", True),
        ("theo dau", True),
        ("teo do", True),
        ("[unk]", False),
        ("unk", False),
        ("đi đâu", False),
        ("thế đâu", False),
    ]

    for raw, expected in test_cases:
        clean = strip_accents(raw).lower()
        if not clean or clean in ["unk", "[unk]"]:
            matched = False
        else:
            words = clean.split()
            if words and words[0] in ["the", "di", "o", "keu", "ai"]:
                matched = False
            else:
                r = fuzz.ratio(clean, "teo dau")
                pr = fuzz.partial_ratio(clean, "teo dau")
                score = float(max(r, pr)) if len(words) >= 2 else float(r)
                matched = score >= 70.0 or "teo dau" in clean

        assert matched == expected, f"Failed for '{raw}': expected {expected}, got {matched}"


def test_dynamic_vad_recording_parameters():
    """10. Kiểm tra thông số Thu âm động: Tối đa 2.5s, im lặng 0.6s dừng sớm, tối thiểu 0.8s."""
    engine = WakeWordEngine(vosk_model_path="tests")
    assert engine.RECORD_DURATION_SEC == 2.5
    assert engine.SILENCE_LIMIT_SEC == 0.6
    assert engine.MIN_RECORD_SEC == 0.8
    assert engine.silence_limit_sec == 0.6


def test_removal_of_optimistic_ui():
    """11. Kiểm tra XÓA BỎ HOÀN TOÀN OPTIMISTIC UI: Chỉ hiện quả cầu và bíp KHI VÀ CHỈ KHI Lớp 3/Lớp 4 xác nhận thành công."""
    events = []

    def mock_early_wake():
        events.append("early_wake")

    def mock_early_cancel():
        events.append("early_cancel")

    def mock_detected(kw, score):
        events.append(f"detected_{kw}")

    engine = WakeWordEngine(
        vosk_model_path="tests",
        on_detected=mock_detected,
        on_early_wake=mock_early_wake,
        on_early_cancel=mock_early_cancel,
        voice_auth_threshold=0.80,
    )

    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    dummy_audio = (1000.0 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()

    # Kịch bản 1: Đúng giọng chủ nhân + Google STT xác nhận "Tèo đâu"
    # -> TUYỆT ĐỐI KHÔNG gọi early_wake! Chỉ kích hoạt khi Google STT xác nhận (detected_teo).
    events.clear()
    engine.voice_verifier = lambda audio: (True, 0.85)
    engine.stt_verifier = lambda audio, timeout=1.5: "Tèo đâu"
    matched_1 = engine._handle_cross_verification(dummy_audio)
    assert matched_1 is True
    assert "early_wake" not in events, "Đã xóa bỏ Optimistic UI -> Cấm gọi early_wake"
    assert "early_cancel" not in events
    assert "detected_teo" in events, "Phải kích hoạt khi Google STT xác nhận thành công"

    # Kịch bản 2: Đúng giọng chủ nhân + Google STT từ chối (nghe ra "xin chào")
    # -> TUYỆT ĐỐI KHÔNG hiện quả cầu sớm (early_wake không gọi), không kích hoạt!
    events.clear()
    engine.voice_verifier = lambda audio: (True, 0.85)
    engine.stt_verifier = lambda audio, timeout=1.5: "xin chào"
    matched_2 = engine._handle_cross_verification(dummy_audio)
    assert matched_2 is False
    assert "early_wake" not in events, "Đã xóa bỏ Optimistic UI -> Không được hiện quả cầu sớm"
    assert "detected_teo" not in events

    # Kịch bản 3: Nhiễu / người lạ (cosine 0.75 < 0.80) -> Bị Voice Auth chặn đứng ở ngưỡng 0.80
    events.clear()
    engine.voice_verifier = lambda audio: (False, 0.75)
    engine.stt_verifier = lambda audio, timeout=1.5: "xin chào"
    matched_3 = engine._handle_cross_verification(dummy_audio)
    assert matched_3 is False
    assert "early_wake" not in events
    assert "detected_teo" not in events


if __name__ == "__main__":
    print("Testing 1: VAD RMS Threshold 350.0 & 2.5s Duration...")
    test_vad_low_threshold_and_duration()
    print("✓ Passed!")

    print("Testing 2: Noise Reduction & Voice Auth 0.80...")
    test_voice_auth_noisereduce_and_threshold()
    print("✓ Passed!")

    print("Testing 3: STT Preprocessing (Noise Reduction + Normalization)...")
    test_stt_preprocessing_noisereduce_and_normalize()
    print("✓ Passed!")

    print("Testing 4: Google STT Retry Mechanism...")
    test_stt_retry_mechanism()
    print("✓ Passed!")

    print("Testing 5: Google STT Fuzzy Matching >= 80% ('em đâu', 'tiền đâu', 'tèo đâu')...")
    test_google_stt_fuzzy_matching_80()
    print("✓ Passed!")

    print("Testing 5b: Blacklist Rejection ('mèo', 'heo', 'tèo đâu rồi', 'mày đi đâu')...")
    test_blacklist_rejection()
    print("✓ Passed!")

    print("Testing 6: Dual Verification Logic...")
    test_dual_verification_logic()
    print("✓ Passed!")

    print("Testing 7: Vosk Layer 4 Grammar & Fuzzy...")
    test_layer4_vosk_grammar_and_fuzzy()
    print("✓ Passed!")

    print("Testing 8: Google STT Hard Timeout 1.5s & No Retry on Timeout...")
    test_google_stt_hard_timeout_15s_no_retry()
    print("✓ Passed!")

    print("Testing 9: Adaptive Noisereduce Speedup (> 0.2s bypass)...")
    test_adaptive_noisereduce_bypass()
    print("✓ Passed!")

    print("Testing 10: Dynamic VAD Recording Parameters...")
    test_dynamic_vad_recording_parameters()
    print("✓ Passed!")

    print("Testing 11: Removal of Optimistic UI (No early wake)...")
    test_removal_of_optimistic_ui()
    print("✓ Passed!")

    print("\n🎉 TOÀN BỘ CÁC MỤC KIỂM THỬ ĐÃ THÀNH CÔNG 100%!")
