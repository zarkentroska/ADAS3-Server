import os
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from modules.ui_window_icon import apply_window_icon


def show_telegram_options_dialog(
    *,
    base_dir,
    t_func,
    get_telegram_config_fn,
    save_telegram_config_fn,
    detect_telegram_chat_id_fn,
):
    """Muestra el diálogo específico de configuración de Telegram."""
    root = tk.Tk()
    root.title(t_func("telegram_window_title"))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    apply_window_icon(root, base_dir=base_dir)

    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    title_label = ttk.Label(main_frame, text=t_func("telegram_section_title"), font=("Arial", 11, "bold"))
    title_label.pack(anchor="w", pady=(0, 6))
    help_label = ttk.Label(
        main_frame,
        text=t_func("telegram_setup_help"),
        font=("Arial", 8),
        foreground="gray",
        wraplength=440,
        justify="left",
    )
    help_label.pack(anchor="w", pady=(0, 12))

    telegram_config = get_telegram_config_fn()
    enabled_var = tk.BooleanVar(value=bool(telegram_config.get("enabled", False)))
    enabled_check = ttk.Checkbutton(
        main_frame,
        text=t_func("telegram_enable_notifications"),
        variable=enabled_var,
    )
    enabled_check.pack(anchor="w", pady=(0, 8))

    token_row = ttk.Frame(main_frame)
    token_row.pack(fill="x", pady=(0, 6))
    token_label = ttk.Label(token_row, text=t_func("telegram_bot_token"), width=17)
    token_label.pack(side="left")
    token_var = tk.StringVar(value=str(telegram_config.get("token", "")))
    token_entry = ttk.Entry(token_row, textvariable=token_var)
    token_entry.pack(side="left", fill="x", expand=True)

    chat_row = ttk.Frame(main_frame)
    chat_row.pack(fill="x", pady=(0, 8))
    chat_id_label = ttk.Label(chat_row, text=t_func("telegram_chat_id"), width=17)
    chat_id_label.pack(side="left")
    chat_id_var = tk.StringVar(value=str(telegram_config.get("chat_id", "")))
    chat_id_entry = ttk.Entry(chat_row, textvariable=chat_id_var)
    chat_id_entry.pack(side="left", fill="x", expand=True)

    action_row = ttk.Frame(main_frame)
    action_row.pack(fill="x", pady=(0, 10))

    def open_botfather():
        webbrowser.open("https://t.me/BotFather")

    open_botfather_button = ttk.Button(
        action_row,
        text=t_func("telegram_open_botfather"),
        command=open_botfather,
    )
    open_botfather_button.pack(side="left", padx=(0, 8))

    status_var = tk.StringVar(value="")
    status_label = ttk.Label(main_frame, textvariable=status_var, font=("Arial", 8))

    def detect_chat_id():
        token_value = token_var.get().strip()
        if not token_value:
            messagebox.showwarning(t_func("error"), t_func("telegram_chat_id_missing_token"))
            return

        chat_id, error = detect_telegram_chat_id_fn(token_value)
        if error:
            status_var.set(error)
            messagebox.showwarning(t_func("error"), error)
            return

        chat_id_var.set(str(chat_id))
        ok_text = t_func("telegram_chat_id_detected", chat_id)
        status_var.set(ok_text)
        messagebox.showinfo(t_func("ok"), ok_text)

    detect_chat_button = ttk.Button(
        action_row,
        text=t_func("telegram_detect_chat_id"),
        command=detect_chat_id,
    )
    detect_chat_button.pack(side="left")
    status_label.pack(anchor="w", pady=(0, 10))

    media_title = ttk.Label(main_frame, text=t_func("telegram_media_options"), font=("Arial", 9, "bold"))
    media_title.pack(anchor="w", pady=(0, 4))

    send_yolo_photo_var = tk.BooleanVar(value=bool(telegram_config.get("send_yolo_photo", True)))
    send_rf_image_var = tk.BooleanVar(value=bool(telegram_config.get("send_rf_image", True)))
    send_audio_clip_var = tk.BooleanVar(value=bool(telegram_config.get("send_audio_clip", True)))

    send_yolo_photo_check = ttk.Checkbutton(
        main_frame,
        text=t_func("telegram_send_yolo_photo"),
        variable=send_yolo_photo_var,
    )
    send_yolo_photo_check.pack(anchor="w")
    send_rf_image_check = ttk.Checkbutton(
        main_frame,
        text=t_func("telegram_send_rf_image"),
        variable=send_rf_image_var,
    )
    send_rf_image_check.pack(anchor="w")
    send_audio_clip_check = ttk.Checkbutton(
        main_frame,
        text=t_func("telegram_send_audio_clip"),
        variable=send_audio_clip_var,
    )
    send_audio_clip_check.pack(anchor="w", pady=(0, 8))

    cooldown_title = ttk.Label(main_frame, text=t_func("telegram_cooldowns_label"), font=("Arial", 9, "bold"))
    cooldown_title.pack(anchor="w", pady=(0, 4))

    cooldowns = telegram_config.get("cooldowns", {})
    yolo_cooldown_var = tk.StringVar(value=str(int(float(cooldowns.get("yolo", 30)))))
    rf_cooldown_var = tk.StringVar(value=str(int(float(cooldowns.get("rf", 30)))))
    audio_cooldown_var = tk.StringVar(value=str(int(float(cooldowns.get("audio", 30)))))

    grid = ttk.Frame(main_frame)
    grid.pack(anchor="w", pady=(0, 10))
    ttk.Label(grid, text=t_func("telegram_cooldown_yolo")).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 4))
    ttk.Spinbox(grid, from_=0, to=600, width=6, textvariable=yolo_cooldown_var).grid(row=0, column=1, sticky="w", pady=(0, 4))
    ttk.Label(grid, text=t_func("telegram_cooldown_rf")).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 4))
    ttk.Spinbox(grid, from_=0, to=600, width=6, textvariable=rf_cooldown_var).grid(row=1, column=1, sticky="w", pady=(0, 4))
    ttk.Label(grid, text=t_func("telegram_cooldown_audio")).grid(row=2, column=0, sticky="w", padx=(0, 8))
    ttk.Spinbox(grid, from_=0, to=600, width=6, textvariable=audio_cooldown_var).grid(row=2, column=1, sticky="w")

    def on_save():
        try:
            yolo_cd = int(yolo_cooldown_var.get())
            rf_cd = int(rf_cooldown_var.get())
            audio_cd = int(audio_cooldown_var.get())
        except ValueError:
            messagebox.showerror(t_func("error"), t_func("telegram_cooldown_number_error"))
            return

        if yolo_cd < 0 or rf_cd < 0 or audio_cd < 0:
            messagebox.showerror(t_func("error"), t_func("telegram_cooldown_range_error"))
            return

        updated_telegram = {
            "enabled": enabled_var.get(),
            "token": token_var.get().strip(),
            "chat_id": chat_id_var.get().strip(),
            "cooldowns": {
                "yolo": yolo_cd,
                "rf": rf_cd,
                "audio": audio_cd,
            },
            "send_yolo_photo": send_yolo_photo_var.get(),
            "send_rf_image": send_rf_image_var.get(),
            "send_audio_clip": send_audio_clip_var.get(),
        }

        if updated_telegram["enabled"] and (not updated_telegram["token"] or not updated_telegram["chat_id"]):
            messagebox.showerror(t_func("error"), t_func("telegram_required_fields_error"))
            return

        if not save_telegram_config_fn(updated_telegram):
            messagebox.showerror(t_func("error"), t_func("telegram_config_save_error"))
            return

        messagebox.showinfo(t_func("ok"), t_func("telegram_save_success"))
        root.destroy()

    def on_cancel():
        root.destroy()

    buttons = ttk.Frame(main_frame)
    buttons.pack(fill="x", pady=(4, 0))
    ttk.Button(buttons, text=t_func("ok"), command=on_save, width=12).pack(side="left", padx=5)
    ttk.Button(buttons, text=t_func("cancel"), command=on_cancel, width=12).pack(side="left", padx=5)

    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(14, 12))
    footer_frame = ttk.Frame(main_frame)
    footer_frame.pack(fill="x")

    github_logo_path = os.path.join(base_dir, "assets", "icons", "ghlogo.png")
    if not os.path.exists(github_logo_path):
        github_logo_path = os.path.join(base_dir, "ghlogo.png")
    github_logo = None

    if os.path.exists(github_logo_path):
        try:
            from PIL import Image, ImageTk

            img = Image.open(github_logo_path)
            img = img.resize((16, 16), Image.Resampling.LANCZOS)
            github_logo = ImageTk.PhotoImage(img)
            github_button = tk.Button(
                footer_frame,
                image=github_logo,
                command=lambda: webbrowser.open("https://github.com/zarkentroska/ADAS3-Server"),
                cursor="hand2",
                relief="flat",
                borderwidth=0,
            )
            github_button.pack(side="left", padx=(0, 10))
            github_button.image = github_logo
        except ImportError:
            try:
                github_logo = tk.PhotoImage(file=github_logo_path)
                github_button = tk.Button(
                    footer_frame,
                    image=github_logo,
                    command=lambda: webbrowser.open("https://github.com/zarkentroska/ADAS3-Server"),
                    cursor="hand2",
                    relief="flat",
                    borderwidth=0,
                )
                github_button.pack(side="left", padx=(0, 10))
                github_button.image = github_logo
            except Exception as exc:
                print(f"No se pudo cargar el logo de GitHub: {exc}")
        except Exception as exc:
            print(f"Error al cargar el logo de GitHub: {exc}")

    copyright_label = ttk.Label(
        footer_frame,
        text="ADAS3 Server v0.7 |  Copyright (C) 2026 GNU GPL 3.0",
        font=("Arial", 8),
        foreground="gray",
    )
    copyright_label.pack(side="left")

    root.mainloop()
