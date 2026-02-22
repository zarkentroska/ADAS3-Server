import os
import sys
import tkinter as tk


def set_opencv_window_icon(window_title, base_dir=None):
    """
    Establece adas3.ico como icono de la ventana OpenCV (barra de tareas en Windows).
    Solo tiene efecto en Windows. Debe llamarse después de cv2.imshow().
    """
    if os.name != "nt":
        return False
    for path in _candidate_icon_paths(base_dir=base_dir):
        if not os.path.exists(path):
            continue
        try:
            import ctypes
            VP = ctypes.c_void_p
            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
            user32.FindWindowW.restype = VP
            hwnd = user32.FindWindowW(None, window_title)
            if not hwnd:
                continue
            # LoadImageW: cargar a 256x256 para que la barra de tareas escale correctamente
            # (0,0 hace que Windows use una resolución pequeña y se vea diminuto)
            user32.LoadImageW.argtypes = [VP, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            user32.LoadImageW.restype = VP
            hicon = user32.LoadImageW(None, path, 1, 256, 256, 0x10)
            if not hicon:
                continue
            WM_SETICON = 0x0080
            ICON_BIG = 1
            ICON_SMALL = 0
            user32.SendMessageW.argtypes = [VP, ctypes.c_uint, ctypes.c_void_p, VP]
            user32.SendMessageW.restype = VP
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
            return True
        except Exception:
            continue
    return False


def _candidate_icon_paths(base_dir=None):
    """Rutas candidatas para adas3.ico (ventana principal y barra de tareas)."""
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, "adas3.ico"))
    # PyInstaller: recursos en _MEIPASS
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, "adas3.ico"))
    module_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(module_parent, "adas3.ico"))
    candidates.append(os.path.join(os.getcwd(), "adas3.ico"))
    return candidates


def _candidate_png_fallback_paths(base_dir=None):
    """Rutas candidatas para settings.png (fallback si no hay .ico)."""
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, "assets", "icons", "settings.png"))
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, "assets", "icons", "settings.png"))
    module_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(module_parent, "assets", "icons", "settings.png"))
    candidates.append(os.path.join(os.getcwd(), "assets", "icons", "settings.png"))
    return candidates


def apply_window_icon(window, base_dir=None):
    """
    Aplica settings.png a ventanas de opciones (Tkinter).
    Solo para diálogos: IP, Tailscale, Telegram, idioma, TinySA, YOLO.
    La ventana principal OpenCV usa adas3.ico vía set_opencv_window_icon.
    """
    for path in _candidate_png_fallback_paths(base_dir=base_dir):
        if not os.path.exists(path):
            continue
        try:
            icon_img = tk.PhotoImage(file=path)
            window.iconphoto(True, icon_img)
            window._adas3_icon_ref = icon_img
            return True
        except Exception:
            continue
    return False
