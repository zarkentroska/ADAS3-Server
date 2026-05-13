"""Dispatcher de diálogos Tkinter y bootstrap/proxy Tk para macOS.

Problemas que resuelve este módulo (todos específicos de macOS):

1. **SIGTRAP al abrir diálogos desde hilos**. En macOS ``tkinter`` sólo
   puede usarse desde el hilo principal; crear ``Tk()`` en un hilo
   secundario provoca ``SIGTRAP`` ("trace trap"). Se mitiga encolando las
   callables de diálogo con :func:`schedule_dialog` y drenándolas en el
   hilo principal con :func:`pump_main_thread_dialogs`.

2. **``-[NSApplication macOSVersion]: unrecognized selector``** al abrir
   un diálogo después de haber importado OpenCV/PyAudio/matplotlib. Tk
   instala una subclase propia de ``NSApplication`` con métodos extra
   (``macOSVersion``). Si otra librería crea ``NSApplication`` antes, Tk
   se cae al leer los colores del sistema. Se mitiga creando un ``Tk()``
   oculto al importar este módulo, **antes** de cualquier import de
   ``cv2``/``pyaudio``/``matplotlib``.

3. **Cuelgue de la ventana de OpenCV al cerrar un diálogo**. Tk y OpenCV
   comparten el mismo ``NSApplication`` y el mismo runloop Cocoa en el
   hilo principal. Cuando cada diálogo crea su propio intérprete Tcl con
   ``Tk()`` y luego lo destruye, el runloop de Cocoa queda en un estado
   en el que la ventana de OpenCV deja de recibir eventos (aparece
   "congelada"). Se mitiga parcheando ``tkinter.Tk`` para que en macOS
   **reutilice** el intérprete del root de bootstrap y cada diálogo se
   cree como ``Toplevel`` de ese root. ``Toplevel.mainloop`` se traduce a
   ``wait_window(self)``, que bloquea hasta que el usuario cierra la
   ventana sin destruir el intérprete Tcl global. Así OpenCV recupera el
   foco del runloop al terminar el diálogo.

En Windows/Linux nada de esto es necesario: no se crea root de bootstrap,
no se parchea ``tkinter.Tk`` y los diálogos siguen lanzándose en hilos
daemon (UX asíncrona intacta).
"""

from __future__ import annotations

import platform
import queue
import threading
from typing import Callable, Optional


IS_MACOS = platform.system() == "Darwin"

_pending: "queue.Queue[Callable[[], None]]" = queue.Queue()


# ---------------------------------------------------------------------------
# Bootstrap: crear un Tk oculto ANTES de que otras libs inicialicen Cocoa.
# ---------------------------------------------------------------------------

_TK_BOOTSTRAP_ROOT = None  # type: ignore[var-annotated]


def _bootstrap_tk_on_macos() -> None:
    """Crea un root Tk oculto y permanente en el hilo principal (macOS).

    Idempotente. Sin esto, en macOS el primer diálogo Tk tras importar
    OpenCV se cae con ``NSInvalidArgumentException`` por
    ``-[NSApplication macOSVersion]: unrecognized selector``.
    """
    global _TK_BOOTSTRAP_ROOT
    if not IS_MACOS or _TK_BOOTSTRAP_ROOT is not None:
        return
    try:
        import tkinter as _tk

        root = _tk.Tk()
        root.withdraw()
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            pass
        _TK_BOOTSTRAP_ROOT = root
    except Exception as exc:  # noqa: BLE001
        print(f"[TK_BOOTSTRAP] No se pudo inicializar Tk en el arranque: {exc}")


def get_bootstrap_root():
    """Devuelve el root Tk oculto y persistente, o ``None`` si no hay."""
    return _TK_BOOTSTRAP_ROOT


# ---------------------------------------------------------------------------
# Proxy de ``tkinter.Tk`` -> ``Toplevel`` del root bootstrap (sólo macOS).
# ---------------------------------------------------------------------------


_TK_PROXY_INSTALLED = False


def _install_tk_proxy_on_macos() -> None:
    """Sustituye ``tkinter.Tk`` por un ``Toplevel`` del root bootstrap.

    Cada llamada ``Tk()`` del código existente creará en su lugar un
    ``Toplevel`` del único intérprete Tcl activo (el bootstrap). Su
    ``mainloop()`` se traduce a ``wait_window(self)``: bloquea hasta que
    el usuario cierra la ventana sin matar el intérprete, y por tanto sin
    corromper el runloop Cocoa compartido con OpenCV.

    Debe llamarse **después** del bootstrap y **antes** de que los
    módulos importen ``from tkinter import Tk``. Como este archivo se
    importa el primero de todo en :mod:`testcam`, los módulos UI que
    luego hagan ``from tkinter import Tk`` recibirán la versión parcheada.
    """
    global _TK_PROXY_INSTALLED
    if not IS_MACOS or _TK_BOOTSTRAP_ROOT is None or _TK_PROXY_INSTALLED:
        return

    import tkinter as _tk

    bootstrap_root = _TK_BOOTSTRAP_ROOT
    _original_tk = _tk.Tk

    # Kwargs que admite Tk pero no Toplevel; los descartamos silenciosamente
    # para no romper llamadas como ``Tk(className="MiApp")``.
    _TK_ONLY_KWARGS = ("className", "screenName", "baseName", "useTk", "sync", "use")

    class _ToplevelAsTk(_tk.Toplevel):
        """``Toplevel`` con interfaz compatible con ``Tk``.

        Se usa en macOS para evitar crear/destruir intérpretes Tcl, lo
        que rompería la integración Cocoa con OpenCV.
        """

        def __init__(self, *args, **kwargs):
            for key in _TK_ONLY_KWARGS:
                kwargs.pop(key, None)
            super().__init__(bootstrap_root, *args, **kwargs)
            # Por si el usuario cierra con la X sin que el código llame a
            # destroy explícitamente: que wait_window retorne limpiamente.
            try:
                self.protocol("WM_DELETE_WINDOW", self.destroy)
            except _tk.TclError:
                pass

        def mainloop(self, n: int = 0):  # type: ignore[override]
            # Traer al frente y esperar al cierre sin detener el
            # intérprete bootstrap (que debe seguir vivo para siempre).
            try:
                self.lift()
                self.focus_force()
            except _tk.TclError:
                pass
            try:
                self.wait_window(self)
            except _tk.TclError:
                pass

        # Exponemos el mismo alias que Tk por compatibilidad (algunos
        # códigos hacen ``root.quit()`` para salir del mainloop).
        def quit(self):  # type: ignore[override]
            try:
                self.destroy()
            except _tk.TclError:
                pass

    # Guardamos referencia al original por si hiciera falta un escape.
    _ToplevelAsTk._original_tk = _original_tk  # type: ignore[attr-defined]

    _tk.Tk = _ToplevelAsTk  # type: ignore[assignment]
    _TK_PROXY_INSTALLED = True


# Ejecutar bootstrap + instalación del proxy al importar este módulo.
_bootstrap_tk_on_macos()
_install_tk_proxy_on_macos()


# ---------------------------------------------------------------------------
# Cola de diálogos para el hilo principal.
# ---------------------------------------------------------------------------


def schedule_dialog(fn: Callable[[], None]) -> Optional[threading.Thread]:
    """Programa la ejecución de un diálogo Tkinter.

    En macOS la encola para el hilo principal; devuelve ``None``.
    En Windows/Linux lanza un hilo daemon y lo devuelve.
    """
    if IS_MACOS:
        _pending.put(fn)
        return None

    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def pump_main_thread_dialogs() -> None:
    """Ejecuta los diálogos encolados en el hilo principal.

    Debe llamarse desde el hilo principal (normalmente desde el bucle de
    OpenCV, una vez por iteración). Cada callable bloquea hasta que el
    usuario cierra el diálogo. También da un ciclo al root bootstrap
    para procesar eventos Tk pendientes.
    """
    while True:
        try:
            fn = _pending.get_nowait()
        except queue.Empty:
            break
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - queremos no romper el bucle
            print(f"[DIALOG] error ejecutando diálogo en hilo principal: {exc}")

    root = _TK_BOOTSTRAP_ROOT
    if root is not None:
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            globals()["_TK_BOOTSTRAP_ROOT"] = None


def has_pending_dialogs() -> bool:
    """Indica si hay diálogos encolados pendientes (útil en tests)."""
    return not _pending.empty()
