"""
Script: download_vosk_medium.py
Mục đích: Tự động tải, giải nén và cấu hình mô hình Vosk tiếng Việt lớn hơn (Medium/Server) cho Tèo Assistant.
Chức năng:
    - Kiểm tra xem mô hình Medium đã tồn tại chưa (nếu có rồi -> bỏ qua).
    - Tải file nén .zip từ máy chủ Alphacephei chính thức kèm thanh tiến trình trực quan (tqdm).
    - Tự động giải nén bằng zipfile và xóa file nén tạm.
    - Xác thực cấu trúc mô hình (am, conf, graph, ivector) và cấu hình sẵn sàng cho hệ thống.
"""

import os
from pathlib import Path
import shutil
import sys
import zipfile

import requests
from tqdm import tqdm

# Đảm bảo UTF-8 trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# URL chính thức của mô hình
URL_VN_022 = "https://alphacephei.com/vosk/models/vosk-model-vn-0.22.zip"
URL_VN_04 = "https://alphacephei.com/vosk/models/vosk-model-vn-0.4.zip"

ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIR_022 = ROOT_DIR / "vosk-model-vn-0.22"
TARGET_DIR_04 = ROOT_DIR / "vosk-model-vn-0.4"


def is_model_valid(model_path: Path) -> bool:
    """Kiểm tra mô hình có đầy đủ các thư mục thành phần cơ bản của Kaldi/Vosk."""
    if not model_path.exists() or not model_path.is_dir():
        return False
    required_parts = ["am", "conf", "graph", "ivector"]
    return all((model_path / part).exists() for part in required_parts)


def download_with_progress(url: str, output_path: Path) -> bool:
    """Tải tệp từ URL với thanh tiến trình trực quan bằng tqdm."""
    print(f"[Download] Đang kết nối tới: {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
    except Exception as exc:
        print(f"[Download Lỗi] Không thể kết nối tới máy chủ: {exc}")
        return False

    if response.status_code != 200:
        print(f"[Download Lỗi] Máy chủ phản hồi mã lỗi HTTP {response.status_code}")
        return False

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunk

    desc = output_path.name
    print(f"[Download] Bắt đầu tải '{desc}' (Dung lượng: {total_size / (1024 * 1024):.1f} MB)...")

    try:
        with open(output_path, "wb") as f, tqdm(
            desc=desc,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=85,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"\n[Download Lỗi] Quá trình tải bị gián đoạn: {exc}")
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Giải nén tệp zip ra thư mục đích với thanh tiến trình."""
    print(f"\n[Giải nén] Đang giải nén '{zip_path.name}' vào '{extract_to}'...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            with tqdm(total=len(members), desc="Giải nén tệp", ncols=85) as bar:
                for member in members:
                    zf.extract(member, extract_to)
                    bar.update(1)
        print("[Giải nén] ✓ Giải nén thành công!")
        return True
    except Exception as exc:
        print(f"[Giải nén Lỗi] Tệp zip bị hỏng hoặc lỗi giải nén: {exc}")
        return False


def main():
    print("=" * 70)
    print("     TỰ ĐỘNG TẢI VÀ THIẾT LẬP MÔ HÌNH VOSK TIẾNG VIỆT LỚN (MEDIUM)")
    print("                      TÈO ASSISTANT")
    print("=" * 70)

    # 1. Kiểm tra xem mô hình đã tồn tại chưa
    if is_model_valid(TARGET_DIR_022):
        print(f"[Vosk] ✓ Đã có model Medium tại: '{TARGET_DIR_022.name}', bỏ qua.")
        return 0

    if is_model_valid(TARGET_DIR_04):
        print(f"[Vosk] ✓ Đã có model Medium tại: '{TARGET_DIR_04.name}', bỏ qua.")
        return 0

    # 2. Xác định URL tải về
    # Kiểm tra URL vosk-model-vn-0.22 trước theo yêu cầu người dùng
    download_url = URL_VN_022
    target_zip = ROOT_DIR / "vosk_model_temp.zip"

    print(f"\n[Bước 1/3] Kiểm tra URL: {download_url}")
    try:
        test_resp = requests.head(download_url, timeout=10, allow_redirects=True)
        if test_resp.status_code == 404:
            print(
                f"[LƯU Ý] Phiên bản 'vn-0.22' không tồn tại trên Alphacephei (Vosk chỉ có bản 0.22 cho tiếng Anh/Nga ~1.8GB).\n"
                f"-> Tự động chuyển sang mô hình tiếng Việt lớn nhất chính thức của Vosk: 'vosk-model-vn-0.4' (~78MB)."
            )
            download_url = URL_VN_04
    except Exception as exc:
        print(f"[Cảnh báo kết nối] Không thể kiểm tra HEAD, thử tải trực tiếp: {exc}")

    # 3. Tiến hành tải tệp
    success = download_with_progress(download_url, target_zip)
    if not success:
        # Nếu tải vn-0.22 lỗi khác, thử chuyển sang vn-0.4
        if download_url != URL_VN_04:
            print("\n[Thử lại] Đang chuyển sang tải mô hình chính thức 'vosk-model-vn-0.4.zip'...")
            download_url = URL_VN_04
            success = download_with_progress(download_url, target_zip)

    if not success:
        print("\n[THẤT BẠI] Không thể tải mô hình Vosk. Vui lòng kiểm tra lại kết nối mạng!")
        return 1

    # 4. Giải nén vào thư mục gốc dự án
    extract_success = extract_zip(target_zip, ROOT_DIR)

    # 5. Xóa file zip tạm sau khi giải nén xong
    if target_zip.exists():
        try:
            target_zip.unlink()
            print(f"[Dọn dẹp] ✓ Đã xóa tệp nén tạm '{target_zip.name}'.")
        except Exception as exc:
            print(f"[Dọn dẹp Cảnh báo] Không thể xóa tệp zip tạm: {exc}")

    if not extract_success:
        print("\n[THẤT BẠI] Giải nén không thành công!")
        return 1

    # 6. Kiểm tra và định vị thư mục mô hình vừa giải nén
    final_model_path = None
    if is_model_valid(TARGET_DIR_022):
        final_model_path = TARGET_DIR_022
    elif is_model_valid(TARGET_DIR_04):
        final_model_path = TARGET_DIR_04
        # Tạo alias thư mục vosk-model-vn-0.22 nếu cần
        try:
            if not TARGET_DIR_022.exists():
                print(f"[Cấu hình] Tạo liên kết đồng bộ sang: '{TARGET_DIR_022.name}'...")
                shutil.copytree(str(TARGET_DIR_04), str(TARGET_DIR_022))
                if is_model_valid(TARGET_DIR_022):
                    final_model_path = TARGET_DIR_022
        except Exception as copy_err:
            print(f"[Cấu hình Cảnh báo] Không thể nhân bản thư mục alias: {copy_err}")

    if final_model_path and is_model_valid(final_model_path):
        print("\n" + "=" * 70)
        print("🎉 HOÀN TẤT CÀI ĐẶT MÔ HÌNH VOSK MEDIUM TIẾNG VIỆT!")
        print(f"📁 Đường dẫn mô hình: {final_model_path.resolve()}")
        print("✓ Đã xác thực đầy đủ các thành phần: am, conf, graph, ivector.")
        print("🛡️ Hệ thống Tèo Assistant đã sẵn sàng hoạt động với mô hình mới chất lượng cao!")
        print("=" * 70)
        return 0
    else:
        print("\n[LỖI] Thư mục mô hình sau khi giải nén bị thiếu các thành phần: am, conf, graph, ivector.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
