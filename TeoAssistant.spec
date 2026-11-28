# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec file for Tèo Assistant
Build mode: ONEDIR
Entry point: app_tray.py
Target name: TeoAssistant
Console: False (GUI background mode with System Tray and Glow Orb Overlay)
Wake Word: Vosk Offline tiếng Việt ("Tèo")
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- 1. THU THẬP TÀI NGUYÊN (DATAS) ---
datas = []

# Thư viện âm thanh sounddevice (PortAudio DLLs)
datas += collect_data_files('_sounddevice_data')

# Vosk offline recognition data / DLLs
datas += collect_data_files('vosk')

# Resemblyzer voice authentication model (pretrained.pt)
datas += collect_data_files('resemblyzer')

# Silero VAD assets (backend PyTorch onnx=False)
datas += collect_data_files('silero_vad')

# Gom các file dữ liệu cục bộ vào thư mục phân phối
# - da.wav: Âm thanh phản hồi Wake Word ("Dạ")
# - memory.json: Bộ nhớ đệm Fast Path
# - learned_phonetics.json: Từ lóng / phiên âm
# - offline_learning.json: Từ điển tự học Vosk
# - model: Thư mục mô hình Vosk Offline tiếng Việt
# (Không đóng gói config.json & my_voice.pkl để bản .exe trên máy khác không bị dính micro/giọng chủ cũ)
local_assets = [
    ('da.wav', '.'),
    ('memory.json', '.'),
    ('learned_phonetics.json', '.'),
    ('offline_learning.json', '.'),
    ('vosk-model-vn-0.22', 'vosk-model-vn-0.22'),
    ('model', 'model'),
]

for src, dst in local_assets:
    if os.path.exists(src):
        datas.append((src, dst))

# --- 2. THU THẬP THƯ VIỆN ĐỘNG (BINARIES) ---
binaries = []

# --- 3. KHAI BÁO HIDDEN IMPORTS ĐẦY ĐỦ ---
hiddenimports = [
    # Giao diện đồ họa PyQt6
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtMultimedia',
    'PyQt6.sip',

    # Thu âm và nhận diện giọng nói
    'sounddevice',
    '_sounddevice',
    'soundfile',
    'speech_recognition',
    'vosk',
    'webrtcvad',
    '_webrtcvad',

    # Học máy, xử lý tín hiệu và tính toán
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.special',
    'scipy.ndimage',
    'sklearn',
    'sklearn.utils',
    'rapidfuzz',

    # Tiện ích hệ thống, mạng và điều khiển
    'winreg',
    'logging',
    'logging.handlers',
    'keyboard',
    'requests',
    'urllib3',
    'winsound',
    'comtypes',
    'pycaw',
    'screen_brightness_control',
    'wmi',
    'psutil',
    'appopener',
    'difflib',
    'json',
    'shutil',
    'threading',
    'queue',
    're',
    'utils',
    'wake_word_engine',
    'ui_overlay',
    'stt_engine',
    'brain',
    'actions',
    'memory_manager',
    'auto_learn',
    'voice_auth',
    'record_my_voice',
    'silero_vad',
    'noisereduce',
    'resemblyzer',
    'librosa',
    'torch',
]

# Tự động gom thêm submodules của vosk, PyQt6, resemblyzer và silero_vad
hiddenimports += collect_submodules('vosk')
hiddenimports += collect_submodules('PyQt6')
hiddenimports += collect_submodules('resemblyzer')
hiddenimports += collect_submodules('silero_vad')
hiddenimports += collect_submodules('noisereduce')

# Loại bỏ trùng lặp
hiddenimports = list(dict.fromkeys(hiddenimports))

# Các thư viện nặng không sử dụng, loại trừ để build nhanh và nhẹ dung lượng
excludes = [
    'openwakeword',
    'onnxruntime',
    'faster_whisper',
    'ctranslate2',
    'torchaudio',
    'torchinfo',
    'matplotlib',
    'tkinter',
    'openpyxl',
    'fastapi',
    'uvicorn',
    'sqlalchemy',
    'PIL',
]

a = Analysis(
    ['app_tray.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TeoAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Ẩn cửa sổ đen CMD console (chạy ngầm trong System Tray)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TeoAssistant',
)
