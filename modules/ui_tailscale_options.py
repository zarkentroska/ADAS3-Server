import os
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from modules.ui_window_icon import apply_window_icon


def show_tailscale_config_dialog(
    base_dir,
    t_func,
    tailscale_installed_fn,
    tailscale_installer_win,
    tailscale_installer_linux,
    install_tailscale_fn,
    get_tailscale_username_fn,
    get_tailscale_ip_fn,
    get_tailscale_connected_devices_fn,
):
    """Muestra el diálogo de configuración de Tailscale."""
    root = tk.Tk()
    root.title(t_func("tailscale_config_title"))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    apply_window_icon(root, base_dir=base_dir)

    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    info_text = t_func("tailscale_oauth_info")
    info_label = ttk.Label(
        main_frame,
        text=info_text,
        font=("Arial", 9),
        foreground="gray",
        wraplength=300,
        justify="left",
    )
    info_label.pack(anchor="w", pady=(0, 15))

    if not tailscale_installed_fn():
        is_macos = sys.platform == "darwin"
        if os.name == "nt":
            installer_exists = os.path.exists(tailscale_installer_win)
        elif is_macos:
            # En macOS instalamos descargando el .pkg oficial en tiempo real.
            installer_exists = True
        else:
            installer_exists = os.path.exists(tailscale_installer_linux)

        if installer_exists:
            install_btn_frame = ttk.Frame(main_frame)
            install_btn_frame.pack(fill="x", pady=(0, 15))
            install_button = None

            def on_install():
                progress_win = tk.Toplevel(root)
                progress_win.title(t_func("install_tailscale"))
                progress_win.attributes("-topmost", True)
                progress_win.resizable(False, False)
                progress_win.protocol("WM_DELETE_WINDOW", lambda: None)
                apply_window_icon(progress_win, base_dir=base_dir)

                pframe = ttk.Frame(progress_win, padding=16)
                pframe.pack(fill="both", expand=True)
                ttk.Label(
                    pframe,
                    text=t_func("tailscale_install_wait_message"),
                    wraplength=320,
                    justify="left",
                ).pack(anchor="w", pady=(0, 10))
                progress = ttk.Progressbar(pframe, orient="horizontal", mode="indeterminate", length=300)
                progress.pack(fill="x")
                progress.start(11)

                if install_button is not None:
                    install_button.configure(state="disabled")

                def _finish(ok, err_text):
                    try:
                        progress.stop()
                    except Exception:
                        pass
                    try:
                        progress_win.destroy()
                    except Exception:
                        pass
                    if install_button is not None:
                        install_button.configure(state="normal")

                    if ok:
                        messagebox.showinfo(
                            t_func("tailscale_install_success"),
                            t_func("tailscale_service_installed"),
                        )
                        root.destroy()
                    else:
                        messagebox.showerror(
                            t_func("error"),
                            err_text or t_func("tailscale_install_error"),
                        )

                def _worker():
                    ok = False
                    err_text = ""
                    try:
                        result = install_tailscale_fn()
                        if isinstance(result, tuple):
                            ok = bool(result[0])
                            err_text = str(result[1] or "")
                        else:
                            ok = bool(result)
                    except Exception as e:
                        err_text = str(e)
                    root.after(0, lambda: _finish(ok, err_text))

                threading.Thread(target=_worker, daemon=True).start()

            install_button = ttk.Button(
                install_btn_frame,
                text=t_func("install_tailscale"),
                command=on_install,
                width=25,
            )
            install_button.pack()
        else:
            no_installer_label = ttk.Label(
                main_frame,
                text=t_func("tailscale_not_installed"),
                font=("Arial", 9),
                foreground="orange",
                wraplength=300,
                justify="left",
            )
            no_installer_label.pack(anchor="w", pady=(0, 15))
    else:
        status_text = t_func("tailscale_installed_info")
        status_label = ttk.Label(
            main_frame,
            text=status_text,
            font=("Arial", 9),
            foreground="green",
            wraplength=300,
            justify="left",
        )
        status_label.pack(anchor="w", pady=(0, 15))

        username = get_tailscale_username_fn()
        tailscale_ip = get_tailscale_ip_fn()

        if username or tailscale_ip:
            info_frame = ttk.Frame(main_frame)
            info_frame.pack(anchor="w", pady=(0, 15))

            if username:
                logged_in_label = ttk.Label(
                    info_frame,
                    text=f"{t_func('tailscale_logged_in_as')} {username}",
                    font=("Arial", 9),
                    foreground="gray",
                    wraplength=300,
                    justify="left",
                )
                logged_in_label.pack(anchor="w", pady=(0, 5))

            if tailscale_ip:
                ip_label = ttk.Label(
                    info_frame,
                    text=f"{t_func('tailscale_ip_device')} {tailscale_ip}",
                    font=("Arial", 9),
                    foreground="darkblue",
                    wraplength=300,
                    justify="left",
                )
                ip_label.pack(anchor="w", pady=(0, 5))

            connected_devices = get_tailscale_connected_devices_fn()
            if tailscale_ip and connected_devices:
                connected_devices = [d for d in connected_devices if d.get("ip") != tailscale_ip]

            if connected_devices:
                other_devices_label = ttk.Label(
                    info_frame,
                    text=t_func("tailscale_other_devices"),
                    font=("Arial", 9),
                    foreground="darkblue",
                    wraplength=300,
                    justify="left",
                )
                other_devices_label.pack(anchor="w", pady=(0, 5))

                for device in connected_devices:
                    device_ip = device.get("ip", "?")
                    device_name = device.get("name", "?")
                    device_label = ttk.Label(
                        info_frame,
                        text=f"{device_ip} ({device_name})",
                        font=("Arial", 9),
                        foreground="red",
                        wraplength=300,
                        justify="left",
                    )
                    device_label.pack(anchor="w", pady=(0, 2))

    def on_close():
        root.destroy()

    def on_create_account():
        webbrowser.open("https://login.tailscale.com/start")

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(15, 0))

    ttk.Button(btn_frame, text=t_func("ok"), command=on_close, width=12).pack(side="left", padx=5)
    ttk.Button(
        btn_frame,
        text=t_func("tailscale_create_account"),
        command=on_create_account,
        width=25,
    ).pack(side="left", padx=5)

    root.mainloop()
    return True
