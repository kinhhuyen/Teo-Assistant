from PyInstaller.utils.hooks import copy_metadata

try:
    datas = copy_metadata('webrtcvad-wheels')
except Exception:
    try:
        datas = copy_metadata('webrtcvad')
    except Exception:
        datas = []

hiddenimports = ['_webrtcvad']
