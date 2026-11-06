"""
Module: app_tray.py
Mục đích: Chương trình điều phối chính của Trợ lý ảo Tèo Assistant (v1.2.0).
Chức năng:
    - Quản lý giao diện PyQt6 trên luồng chính (Main Thread):
        + Cửa sổ Glow Orb Overlay trong suốt, mờ ảo chính giữa màn hình.
        + Khay hệ thống (System Tray Icon) chạy ngầm trên thanh tác vụ Windows Taskbar.
        + Tự động khởi động cùng Windows (Autostart qua Registry HKCU Run).
        + Lựa chọn và thay đổi Microphone trực tiếp từ menu Tray, lưu cấu hình vào `config.json`.
        + Ghi log xoay vòng ra tệp `teo_debug.log` để bắt bệnh và debug từ xa khi ẩn CMD console.
    - Chạy bộ nhận diện Wake Word "Tèo đâu" (Vosk tiếng Việt) và pipeline xử lý lệnh trên các luồng phụ.
"""

import concurrent.futures
import ctypes
import os
import socket
import sys
import threading
import time
import winsound
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

# Đảm bảo tiếng Việt hiển thị chính xác trên Windows Console
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Khởi động hệ thống ghi log xoay vòng ra teo_debug.log ngay từ đầu
from utils import setup_file_logging
setup_file_logging("teo_debug.log")

# =============================================================================
# SINGLE INSTANCE GUARD (Kiểm tra tức thì TRƯỚC KHI nạp bất kỳ thư viện nặng nào)
# =============================================================================
_SINGLE_INSTANCE_MUTEX = None
_SINGLE_INSTANCE_SOCKET = None
_IS_ALREADY_ACQUIRED = False
SINGLE_INSTANCE_PORT = 65432


def ensure_single_instance() -> bool:
    """
    Kiểm tra và đảm bảo chỉ có duy nhất 1 tiến trình Tèo Assistant chạy trên hệ thống.
    Sử dụng cơ chế Dual-Lock (Windows Named Mutex qua ctypes + Socket bind localhost:65432):
        1. Named Mutex: Cơ chế Windows API giải phóng tức thì khi tiến trình bị đóng (kể cả crash).
        2. Localhost Socket: Lớp bảo vệ bổ trợ ngăn chặn xung đột trên mọi môi trường.

    Trả về:
        True: Nếu đây là tiến trình duy nhất (được phép khởi động).
        False: Nếu đã có một tiến trình khác đang chạy ngầm.
    """
    global _SINGLE_INSTANCE_MUTEX, _SINGLE_INSTANCE_SOCKET, _IS_ALREADY_ACQUIRED

    if _IS_ALREADY_ACQUIRED:
        return True

    # 1. Kiểm tra qua Windows Named Mutex (ctypes)
    if sys.platform.startswith("win"):
        try:
            ERROR_ALREADY_EXISTS = 183
            kernel32 = ctypes.windll.kernel32
            mutex_name = "Local\\TeoAssistant_SingleInstance_Mutex"
            mutex = kernel32.CreateMutexW(None, False, mutex_name)
            last_error = kernel32.GetLastError()
            if last_error == ERROR_ALREADY_EXISTS:
                if mutex:
                    kernel32.CloseHandle(mutex)
                return False
            _SINGLE_INSTANCE_MUTEX = mutex
        except Exception:
            pass

    # 2. Kiểm tra qua Socket Binding (localhost:65432)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
        sock.listen(1)
        _SINGLE_INSTANCE_SOCKET = sock
    except (socket.error, OSError):
        # Nếu socket bind thất bại -> đã có tiến trình khác giữ port 65432
        if _SINGLE_INSTANCE_MUTEX and sys.platform.startswith("win"):
            try:
                ctypes.windll.kernel32.CloseHandle(_SINGLE_INSTANCE_MUTEX)
            except Exception:
                pass
            _SINGLE_INSTANCE_MUTEX = None
        return False

    _IS_ALREADY_ACQUIRED = True
    return True


# Kiểm tra nhanh tại entry point TRƯỚC KHI import PyQt6 hay Vosk
if __name__ == "__main__":
    if "--record-voice" in sys.argv:
        if sys.platform.startswith("win"):
            try:
                import ctypes
                if not ctypes.windll.kernel32.AttachConsole(-1):
                    ctypes.windll.kernel32.AllocConsole()
                sys.stdout = open("CONOUT$", "w", encoding="utf-8")
                sys.stderr = open("CONOUT$", "w", encoding="utf-8")
                sys.stdin = open("CONIN$", "r", encoding="utf-8")
            except Exception:
                pass
        from record_my_voice import main as record_voice_main
        record_voice_main()
        sys.exit(0)

    if not ensure_single_instance():
        try:
            print("Tèo Assistant đang chạy ngầm!", flush=True)
        except Exception:
            pass
        sys.exit(0)

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QGradient,
    QIcon,
    QLinearGradient,
    QPainter,
    QPixmap,
)
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from actions import ActionExecutor
from brain import Brain
from stt_engine import SpeechToText
from ui_overlay import GlowOrbOverlay
from utils import (
    get_available_microphones,
    get_configured_microphone,
    get_exe_dir,
    is_autostart_enabled,
    set_autostart,
    set_configured_microphone,
)
from wake_word_engine import WakeWordEngine


def create_tray_icon() -> QIcon:
    """Tạo biểu tượng System Tray phong cách Tèo Orb rực rỡ dạng vector bằng QPainter."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Vẽ vòng tròn gradient màu xanh cyan - tím neon
    grad = QLinearGradient(0, 0, 64, 64)
    grad.setColorAt(0.0, QColor(0, 230, 255))     # Cyan
    grad.setColorAt(0.5, QColor(138, 43, 226))    # Violet
    grad.setColorAt(1.0, QColor(255, 0, 128))     # Deep Pink

    painter.setBrush(grad)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 56, 56)

    # Vẽ vòng viền phát quang
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QColor(255, 255, 255, 220))
    painter.drawEllipse(10, 10, 44, 44)

    # Chữ "T" viết tắt của Tèo ở giữa
    painter.setPen(QColor(255, 255, 255))
    font = QFont("Arial", 24, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "T")

    painter.end()
    return QIcon(pixmap)


class CommandProcessingWorker(QThread):
    """
    Luồng nền chuyên thu âm và thực thi lệnh khi phát hiện Wake Word.
    """

    show_overlay_sig = pyqtSignal(str)
    update_status_sig = pyqtSignal(str)
    hide_overlay_sig = pyqtSignal()
    command_finished_sig = pyqtSignal()

    def __init__(
        self,
        stt: SpeechToText,
        brain: Brain,
        executor: ActionExecutor,
        wake_engine: WakeWordEngine,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.stt = stt
        self.brain = brain
        self.executor = executor
        self.wake_engine = wake_engine
        self._is_busy = False
        self.last_activity_time = time.time()

    def ping_activity(self, message: str = "") -> None:
        """Cập nhật mốc thời gian hoạt động gần nhất để Watchdog biết hệ thống vẫn đang xử lý bình thường."""
        self.last_activity_time = time.time()
        if message:
            print(f"[Worker Activity] {message}", flush=True)

    def trigger_command(self, wake_word_name: str = "teo") -> None:
        """Kích hoạt chạy luồng xử lý lệnh nếu chưa bận."""
        if self.isRunning() or self._is_busy:
            print("[Worker] Đang bận xử lý lệnh trước, bỏ qua trigger mới.")
            return
        self.start()

    def run(self) -> None:
        """
        Thực thi toàn bộ pipeline nhận diện và chạy hành động.
        Cơ chế giám sát thông minh Watchdog 15 giây (chống treo cứng):
        - Cập nhật liên tục self.last_activity_time mỗi khi STT, Brain hoặc Action có tiến triển.
        - Luồng Watchdog kiểm tra định kỳ 1s: chỉ khi hệ thống bị treo hoàn toàn > 15s không có hoạt động
          mới can thiệp kill tiến trình và reset micro.
        - Nếu đang xử lý bình thường (có log/tiến triển), Watchdog để tác vụ chạy đến khi xong.
        """
        self._is_busy = True
        self.ping_activity("Khởi động chu trình xử lý lệnh")

        stop_watchdog = threading.Event()
        pipeline_finished = threading.Event()
        is_hung = threading.Event()

        def _watchdog_loop():
            while not stop_watchdog.wait(1.0):
                if pipeline_finished.is_set():
                    break
                elapsed = time.time() - self.last_activity_time
                if elapsed > 15.0:
                    is_hung.set()
                    print(
                        f"\n[Worker Watchdog] ⚠️ Phát hiện hệ thống không có tiến triển trong {elapsed:.1f}s! "
                        f"Tiến hành reset lại Micro (KHÔNG kill tiến trình Brain để tránh ngắt dịch)..."
                    )
                    self.update_status_sig.emit("Hệ thống phản hồi chậm (Đang reset Micro)...")

                    # Chỉ reset lại Microphone, TUYỆT ĐỐI KHÔNG kill tiến trình Brain/subprocess
                    try:
                        self.wake_engine.reset_microphone()
                    except Exception as mic_err:
                        print(f"[Worker Watchdog Cảnh báo reset micro]: {mic_err}")
                    break

        watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            name="TeoWatchdogThread",
            daemon=True,
        )
        watchdog_thread.start()

        try:
            # 1. Mở Overlay và phát tiếng Bíp (chỉ khi Lớp 3 hoặc Lớp 4 đã xác thực thành công từ khóa)
            self.show_overlay_sig.emit("Tèo đang lắng nghe...")
            try:
                winsound.Beep(1200, 180)  # Tần số 1200Hz, 180ms
            except Exception:
                pass

            # 2. Thu âm câu lệnh qua STT Engine (silence_limit=1.5s, max_duration=6.0s)
            print("\n[Tèo Assistant] 🎙️ Đang lắng nghe câu lệnh người dùng...")
            self.ping_activity("Bắt đầu thu âm giọng nói...")
            audio_bytes = self.stt.record_wake_command(max_duration=6.0, silence_limit=1.5)
            self.ping_activity("Thu âm giọng nói hoàn tất")

            if not audio_bytes:
                self.update_status_sig.emit("Không nghe thấy lệnh...")
                time.sleep(0.8)
                return

            def _pipeline_task():
                # 3. Nhận diện giọng nói
                self.update_status_sig.emit("Đang nhận diện giọng nói...")
                self.ping_activity("Bắt đầu nhận diện STT")
                user_text = self.stt.process_audio(audio_bytes)
                self.ping_activity("STT nhận diện hoàn tất")

                if not user_text or not user_text.strip():
                    self.update_status_sig.emit("Không nghe rõ câu lệnh...")
                    time.sleep(0.8)
                    return

                print(f"[Tèo Assistant 👤] Lệnh nhận diện: \"{user_text}\"")
                self.update_status_sig.emit(f'"{user_text}"')
                time.sleep(0.3)

                # 4. Phân tích ý định qua Bộ não
                self.update_status_sig.emit("Đang thực thi lệnh...")
                self.ping_activity(f"Bắt đầu phân tích ý định Brain: '{user_text}'")
                t0 = time.time()
                actions_list = self.brain.process(user_text)
                elapsed = round(time.time() - t0, 3)
                self.ping_activity(f"Brain phân tích xong ({elapsed}s)")
                print(f"[Tèo Assistant 🧠] Ý định giải mã xong ({elapsed}s): {actions_list}")

                # 5. Thực thi các hành động
                if actions_list:
                    self.ping_activity("Bắt đầu thực thi các hành động Action")
                    self.executor.execute(actions_list)
                    self.ping_activity("Thực thi các hành động hoàn tất")
                    time.sleep(0.5)

                print("[Tèo Assistant] ✓ Đã hoàn thành lệnh. Sẵn sàng lắng nghe từ khóa tiếp theo.")

            # Thực thi pipeline và giám sát linh hoạt bằng Watchdog (Không dùng timeout 5s tuyệt đối)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_pipeline_task)
                while not future.done():
                    try:
                        future.result(timeout=0.5)
                    except concurrent.futures.TimeoutError:
                        if is_hung.is_set():
                            print("[Worker] ⚠️ Watchdog báo treo cứng > 15s. Hủy chờ pipeline.")
                            break

        except Exception as exc:
            print(f"[Tèo Assistant Lỗi Pipeline]: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            pipeline_finished.set()
            stop_watchdog.set()
            self.hide_overlay_sig.emit()
            self._is_busy = False
            # Mở lại luồng lắng nghe Wake Word sau khi hoàn tất
            try:
                self.wake_engine.resume()
            except Exception as res_err:
                print(f"[Worker Cảnh báo resume]: {res_err}")
            self.command_finished_sig.emit()


class TeoAssistantApp:
    """
    Lớp điều phối toàn diện ứng dụng Tèo Assistant (v1.3):
    - Quản lý ứng dụng Qt và System Tray.
    - Autostart Windows qua HKCU Run.
    - Cấu hình Microphone linh hoạt từ Tray.
    - Mở file log debug thuận tiện.
    - Kết nối tín hiệu giữa WakeWordEngine (Vosk), Worker và Overlay.
    """

    def __init__(self) -> None:
        # 1. Khởi tạo QApplication
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)  # Giữ app chạy nền khi đóng overlay

        print("\n" + "=" * 68)
        print("     KHỞI ĐỘNG TÈO ASSISTANT v1.2.0 (VOSK WAKE WORD & TRAY OVERLAY)")
        print("=" * 68)

        # 2. Khởi tạo Giao diện Overlay
        print("[Tèo] [1/4] Đang khởi tạo giao diện Glow Orb Overlay...")
        self.overlay = GlowOrbOverlay(size=300)

        # 3. Khởi tạo các module bộ não và hành động
        print("[Tèo] [2/4] Đang khởi tạo Bộ nhận diện STT, Brain và ActionExecutor...")
        self.stt = SpeechToText()
        self.brain = Brain()
        self.executor = ActionExecutor()

        # 4. Khởi tạo Wake Word Engine bằng Vosk kết hợp Voice Auth (Resemblyzer) & Google STT xác thực chéo
        print("[Tèo] [3/4] Đang khởi tạo Wake Word Engine ('Tèo đâu') với Voice Auth sinh trắc học...")
        self.wake_engine = WakeWordEngine(
            wake_keyword="tèo đâu",
            on_detected=self._on_wake_word_triggered,
            min_conf=0.60,
            fuzzy_threshold=70.0,
            voice_auth_threshold=0.80,
            min_rms=350.0,
            cooldown_sec=1.0,
            stt_verifier=self.stt.recognize_google,
        )

        # 5. Khởi tạo Worker xử lý giọng nói (QThread)
        print("[Tèo] [4/4] Đang kết nối luồng xử lý giọng nói CommandProcessingWorker...")
        self.worker = CommandProcessingWorker(
            stt=self.stt,
            brain=self.brain,
            executor=self.executor,
            wake_engine=self.wake_engine,
        )

        # Kết nối tín hiệu từ Worker sang Overlay an toàn đa luồng
        self.worker.show_overlay_sig.connect(self.overlay.show_overlay)
        self.worker.update_status_sig.connect(self.overlay.set_status)
        self.worker.hide_overlay_sig.connect(self.overlay.hide_overlay)

        # 6. Cấu hình Khay hệ thống (System Tray Icon)
        self._setup_system_tray()

    def _setup_system_tray(self) -> None:
        """Thiết lập biểu tượng và menu ngữ cảnh cho Khay hệ thống Windows."""
        self.tray_icon = QSystemTrayIcon(create_tray_icon(), self.app)
        self.tray_icon.setToolTip("Tèo Assistant v1.2.0 - Sẵn sàng (Nói 'Tèo đâu')")

        self.tray_menu = QMenu()

        # 1. Tiêu đề thông tin
        title_action = QAction("Tèo Assistant v1.2.0 (Đang hoạt động)", self.tray_menu)
        title_action.setEnabled(False)
        self.tray_menu.addAction(title_action)
        self.tray_menu.addSeparator()

        # 2. Bật/Tắt Lắng nghe Wake Word
        self.toggle_wake_action = QAction("Lắng nghe Wake Word (Tèo đâu)", self.tray_menu)
        self.toggle_wake_action.setCheckable(True)
        self.toggle_wake_action.setChecked(True)
        self.toggle_wake_action.triggered.connect(self._toggle_wake_word)
        self.tray_menu.addAction(self.toggle_wake_action)

        # 3. Kiểm tra thử nghiệm Overlay
        test_overlay_action = QAction("Xem thử Overlay (3s)", self.tray_menu)
        test_overlay_action.triggered.connect(self._test_show_overlay)
        self.tray_menu.addAction(test_overlay_action)

        self.tray_menu.addSeparator()

        # 4. Tự động khởi động cùng Windows (Registry HKCU Run)
        self.autostart_action = QAction("Khởi động cùng Windows", self.tray_menu)
        self.autostart_action.setCheckable(True)
        self.autostart_action.setChecked(is_autostart_enabled())
        self.autostart_action.triggered.connect(self._toggle_autostart)
        self.tray_menu.addAction(self.autostart_action)

        # 5. Đăng ký / Cập nhật Giọng nói (Voice ID)
        enroll_voice_action = QAction("🎙️ Đăng ký / Cập nhật Giọng nói (Voice ID)", self.tray_menu)
        enroll_voice_action.triggered.connect(self._enroll_voice_id)
        self.tray_menu.addAction(enroll_voice_action)

        # 6. Cấu hình Microphone (Submenu chọn Micro)
        self.mic_submenu = self.tray_menu.addMenu("🎧 Cấu hình Micro")
        self._build_mic_menu(self.mic_submenu)

        # 7. Mở file log debug (Bắt bệnh từ xa)
        open_log_action = QAction("📄 Mở file Log (teo_debug.log)", self.tray_menu)
        open_log_action.triggered.connect(self._open_log_file)
        self.tray_menu.addAction(open_log_action)

        self.tray_menu.addSeparator()

        # 7. Nút Thoát
        exit_action = QAction("Thoát", self.tray_menu)
        exit_action.triggered.connect(self.quit)
        self.tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.show()

        # Thông báo chào mừng
        self.tray_icon.showMessage(
            "Tèo Assistant v1.3.1",
            "Đã khởi động ngầm trên khay hệ thống.\nHãy gọi 'Tèo đâu' để ra lệnh!",
            QSystemTrayIcon.MessageIcon.Information,
            3500,
        )

    def _build_mic_menu(self, menu: QMenu) -> None:
        """Xây dựng danh sách các thiết bị Microphone khả dụng trong submenu."""
        menu.clear()

        curr_idx, curr_name = get_configured_microphone()
        mics = get_available_microphones(filter_compatible=True)

        self.mic_action_group = QActionGroup(menu)
        self.mic_action_group.setExclusive(True)

        # Lựa chọn 1: Mặc định theo hệ thống
        default_act = QAction("Mặc định theo hệ thống (Khuyên dùng)", menu)
        default_act.setCheckable(True)
        default_act.setChecked(curr_idx is None)
        default_act.triggered.connect(lambda: self._on_select_microphone(None, "Mặc định hệ thống"))
        self.mic_action_group.addAction(default_act)
        menu.addAction(default_act)

        menu.addSeparator()

        # Danh sách các micro vật lý kết nối
        for m in mics:
            api_info = f" [{m['hostapi_name']}]" if m.get("hostapi_name") else ""
            label = f"{m['name']}{api_info} (ID {m['index']})"
            act = QAction(label, menu)
            act.setCheckable(True)
            act.setChecked(curr_idx == m["index"])
            act.triggered.connect(
                lambda checked, idx=m["index"], name=m["name"]: self._on_select_microphone(idx, name)
            )
            self.mic_action_group.addAction(act)
            menu.addAction(act)

        menu.addSeparator()
        refresh_act = QAction("🔄 Làm mới danh sách Micro", menu)
        refresh_act.triggered.connect(lambda: self._build_mic_menu(menu))
        menu.addAction(refresh_act)

    def _toggle_autostart(self, checked: bool) -> None:
        """Bật hoặc tắt tính năng khởi động cùng Windows."""
        success = set_autostart(checked)
        if success:
            state = "BẬT" if checked else "TẮT"
            msg = f"Đã {state} tính năng tự động khởi động cùng Windows."
            print(f"[App] 🚀 {msg}")
            self.tray_icon.showMessage(
                "Khởi động cùng Windows",
                msg,
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )
        else:
            self.autostart_action.setChecked(not checked)

    def _on_select_microphone(self, device_index: Optional[int], device_name: str) -> None:
        """Xử lý khi người dùng chọn Microphone từ menu."""
        set_configured_microphone(device_index, device_name)
        self.wake_engine.set_input_device(device_index, device_name)
        self.stt.set_input_device(device_index, device_name)
        msg = f"Đã chuyển sang Microphone: {device_name}"
        print(f"[App] 🎙️ {msg}")
        self.tray_icon.showMessage(
            "Cấu hình Microphone",
            msg,
            QSystemTrayIcon.MessageIcon.Information,
            2500,
        )

    def _enroll_voice_id(self) -> None:
        """Kích hoạt chương trình ghi âm 10 lần khẩu lệnh 'Tèo đâu' để đăng ký Voice ID."""
        # Tạm dừng Wake Word để nhường micro cho việc ghi âm
        self.wake_engine.pause()

        # Hiển thị thông báo hướng dẫn
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Đăng ký Giọng nói Chủ nhân (Voice ID)")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(
            "<h3>🎙️ Đăng ký Hồ sơ Giọng nói (Voice ID)</h3>"
            "<p>Tính năng này giúp <b>Tèo Assistant chỉ phản hồi duy nhất với giọng nói của bạn</b>, "
            "triệt tiêu hoàn toàn kích hoạt ảo do tiếng ồn hoặc người khác nói.</p>"
            "<p><b>Quy trình thu âm:</b></p>"
            "<ol>"
            "<li>Một cửa sổ dòng lệnh (Console) sẽ hiện lên.</li>"
            "<li>Hệ thống sẽ đếm ngược <b>3, 2, 1</b> và phát tiếng bíp.</li>"
            "<li>Bạn nói rõ ràng: <b>\"Tèo đâu\"</b> (lặp lại <b>10 lần</b>).</li>"
            "</ol>"
            "<p>Nhấn <b>OK</b> để bắt đầu ghi âm ngay bây giờ!</p>"
        )
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        ret = msg_box.exec()

        if ret != QMessageBox.StandardButton.Ok:
            self.wake_engine.resume()
            return

        import subprocess
        from utils import is_frozen

        if is_frozen():
            cmd = [sys.executable, "--record-voice"]
        else:
            script_path = Path(__file__).resolve().parent / "record_my_voice.py"
            cmd = [sys.executable, str(script_path)]

        print(f"[App] 🎙️ Bắt đầu tiến trình ghi âm giọng nói: {cmd}")

        try:
            if sys.platform.startswith("win"):
                proc = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                proc = subprocess.Popen(cmd)

            # Luồng giám sát hoàn thành
            def _wait_and_reload():
                proc.wait()
                try:
                    from voice_auth import get_authenticator
                    auth = get_authenticator()
                    auth.load_profile()
                except Exception as exc:
                    print(f"[App Lỗi] Không thể reload Voice profile: {exc}")
                self.wake_engine.resume()
                self.tray_icon.showMessage(
                    "Đăng ký Giọng nói thành công",
                    "Đã lưu thành công hồ sơ Voice ID! Giờ đây chỉ có bạn mới có thể đánh thức Tèo.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3500,
                )

            threading.Thread(target=_wait_and_reload, daemon=True, name="VoiceEnrollWatcher").start()

        except Exception as exc:
            print(f"[App Lỗi] Không thể khởi chạy tiến trình đăng ký giọng nói: {exc}")
            self.wake_engine.resume()

    def _open_log_file(self) -> None:
        """Mở tệp teo_debug.log để người dùng xem trực tiếp."""
        log_path = get_exe_dir() / "teo_debug.log"
        if not log_path.exists():
            try:
                log_path.touch()
            except Exception:
                pass
        print(f"[App] 📄 Đang mở file log: '{log_path}'")
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(log_path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as exc:
            print(f"[App Lỗi] Không thể mở file log: {exc}")

    def _on_wake_word_triggered(self, model_name: str, score: float) -> None:
        """Hàm callback được gọi từ luồng Vosk Wake Word khi phát hiện từ khóa."""
        print(f"\n[App] 🔔 Nhận tín hiệu Wake Word: '{model_name}' -> Kích hoạt Worker!")
        self.worker.trigger_command(model_name)

    def _toggle_wake_word(self, checked: bool) -> None:
        """Bật hoặc tắt luồng lắng nghe từ khóa từ menu Tray."""
        if checked:
            self.wake_engine.resume()
            print("[App] Đã bật lại tính năng lắng nghe Wake Word.")
        else:
            self.wake_engine.pause()
            print("[App] Đã tạm dừng tính năng lắng nghe Wake Word.")

    def _test_show_overlay(self) -> None:
        """Hiển thị overlay thử nghiệm trong 3 giây."""
        self.overlay.show_overlay("Tèo đang thử nghiệm...")
        QTimer.singleShot(3000, self.overlay.hide_overlay)

    def start(self) -> None:
        """Bắt đầu chạy ứng dụng."""
        # Khởi động luồng WakeWordEngine
        self.wake_engine.start()

        print("\n" + "=" * 68)
        print("✓ TÈO ASSISTANT v1.2.0 ĐÃ HOẠT ĐỘNG HOÀN TOÀN TRÊN KHAY HỆ THỐNG!")
        print("• Hãy nói vào micro: 'Tèo đâu'")
        print("• Nhấn chuột phải vào icon ở góc màn hình (System Tray) để cấu hình:")
        print("    + Bật/Tắt Khởi động cùng Windows")
        print("    + Chọn thiết bị Microphone đang dùng")
        print("    + Mở file Log teo_debug.log để bắt bệnh từ xa")
        print("• Nhấn Ctrl + C trên terminal để đóng chương trình.")
        print("=" * 68 + "\n")

        # Chạy vòng lặp sự kiện chính của Qt (Main Thread)
        try:
            sys.exit(self.app.exec())
        except KeyboardInterrupt:
            self.quit()

    def quit(self) -> None:
        """Đóng an toàn toàn bộ ứng dụng và các tiến trình nền."""
        print("\n[App] Đang đóng Tèo Assistant...")
        try:
            self.wake_engine.stop()
        except Exception:
            pass

        try:
            if self.worker.isRunning():
                self.worker.quit()
                self.worker.wait(1000)
        except Exception:
            pass

        try:
            self.overlay.hide_overlay()
        except Exception:
            pass

        try:
            self.tray_icon.hide()
        except Exception:
            pass

        print("[App] ✓ Đã dừng toàn bộ luồng. Tạm biệt bạn!")

        # Giải phóng Mutex và Socket Single Instance
        global _SINGLE_INSTANCE_MUTEX, _SINGLE_INSTANCE_SOCKET, _IS_ALREADY_ACQUIRED
        if _SINGLE_INSTANCE_SOCKET:
            try:
                _SINGLE_INSTANCE_SOCKET.close()
            except Exception:
                pass
            _SINGLE_INSTANCE_SOCKET = None

        if _SINGLE_INSTANCE_MUTEX and sys.platform.startswith("win"):
            try:
                ctypes.windll.kernel32.CloseHandle(_SINGLE_INSTANCE_MUTEX)
            except Exception:
                pass
            _SINGLE_INSTANCE_MUTEX = None

        _IS_ALREADY_ACQUIRED = False
        sys.exit(0)


# Tương thích ngược tên lớp cũ
EchoAssistantApp = TeoAssistantApp


if __name__ == "__main__":
    app_instance = TeoAssistantApp()
    app_instance.start()
