import threading
from tkinter import Tk, messagebox, simpledialog
from modules.ui_window_icon import apply_window_icon


def show_warning_async(translate_fn, title_key, message_key):
    """Muestra un warning en un hilo aparte para no bloquear la UI principal."""

    def _show_warning():
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        apply_window_icon(root)
        messagebox.showwarning(translate_fn(title_key), translate_fn(message_key))
        root.destroy()

    threading.Thread(target=_show_warning, daemon=True).start()


def solicitar_nueva_ip(ip_actual, t_func):
    """Muestra diálogo para cambiar la IP."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    apply_window_icon(root)

    nueva_ip = simpledialog.askstring(
        t_func("change_camera_ip"),
        t_func("enter_new_ip", ip_actual),
        initialvalue=ip_actual,
    )

    root.destroy()

    if nueva_ip and nueva_ip.strip():
        return nueva_ip.strip()
    return None
