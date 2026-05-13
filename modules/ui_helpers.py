from tkinter import Tk, messagebox, simpledialog

from modules.mainthread_dispatch import IS_MACOS, schedule_dialog
from modules.ui_window_icon import apply_window_icon


def show_warning_async(translate_fn, title_key, message_key):
    """Muestra un warning sin bloquear el hilo principal.

    En Windows/Linux se lanza en un hilo aparte. En macOS se encola para
    ejecutarse en el hilo principal (tkinter en macOS no es thread-safe).
    """

    def _show_warning():
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        apply_window_icon(root)
        messagebox.showwarning(translate_fn(title_key), translate_fn(message_key))
        root.destroy()

    schedule_dialog(_show_warning)


def solicitar_nueva_ip(ip_actual, t_func):
    """Muestra diálogo para cambiar la IP. Debe invocarse en el hilo principal en macOS."""
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


# Re-export para módulos que prefieran la constante desde aquí.
__all__ = ["show_warning_async", "solicitar_nueva_ip", "IS_MACOS"]
