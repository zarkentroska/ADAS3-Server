import os
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from modules.ui_window_icon import apply_window_icon


def show_language_selection_dialog(
    base_dir,
    t_func,
    translate_for_language_fn,
    get_current_language_fn,
    get_audio_confidence_threshold_fn,
    guardar_idioma_fn,
    guardar_audio_threshold_fn,
):
    """Muestra el diálogo para seleccionar idioma y sensibilidad de audio."""
    root = tk.Tk()
    root.title(t_func("language_selection_title"))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    apply_window_icon(root, base_dir=base_dir)

    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    title_label = ttk.Label(main_frame, text=t_func("select_language"), font=("Arial", 11, "bold"))
    title_label.pack(anchor="w", pady=(0, 15))

    languages = [
        ("es", "Español"),
        ("en", "English"),
        ("fr", "Français"),
        ("it", "Italiano"),
        ("pt", "Português"),
    ]

    selected_lang = tk.StringVar(value=get_current_language_fn())

    nvidia_label = ttk.Label(
        main_frame,
        text=t_func("nvidia_cuda_info"),
        font=("Arial", 9),
        foreground="gray",
        wraplength=400,
        justify="left",
    )

    sensitivity_frame = ttk.Frame(main_frame)
    sensitivity_label = ttk.Label(sensitivity_frame, text=t_func("audio_sensitivity_label"), font=("Arial", 9))

    ok_button = None
    cancel_button = None

    def update_ui_texts():
        temp_lang = selected_lang.get()
        root.title(translate_for_language_fn(temp_lang, "language_selection_title"))
        title_label.config(text=translate_for_language_fn(temp_lang, "select_language"))
        nvidia_label.config(text=translate_for_language_fn(temp_lang, "nvidia_cuda_info"))
        sensitivity_label.config(text=translate_for_language_fn(temp_lang, "audio_sensitivity_label"))
        percent_label.config(text=translate_for_language_fn(temp_lang, "audio_sensitivity_percent"))
        if ok_button is not None:
            ok_button.config(text=translate_for_language_fn(temp_lang, "ok"))
        if cancel_button is not None:
            cancel_button.config(text=translate_for_language_fn(temp_lang, "cancel"))

    for lang_code, lang_name in languages:
        ttk.Radiobutton(
            main_frame,
            text=lang_name,
            variable=selected_lang,
            value=lang_code,
            command=update_ui_texts,
        ).pack(anchor="w", pady=5)

    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(15, 15))

    sensitivity_frame.pack(fill="x", pady=(0, 15))
    sensitivity_label.pack(anchor="w", pady=(0, 5))

    sensitivity_control_frame = ttk.Frame(sensitivity_frame)
    sensitivity_control_frame.pack(fill="x")

    current_threshold_percent = int(get_audio_confidence_threshold_fn() * 100)
    sensitivity_var = tk.StringVar(value=str(current_threshold_percent))

    sensitivity_spinbox = ttk.Spinbox(
        sensitivity_control_frame,
        from_=1,
        to=100,
        textvariable=sensitivity_var,
        width=10,
    )
    sensitivity_spinbox.pack(side="left", padx=(0, 5))
    percent_label = ttk.Label(sensitivity_control_frame, text=t_func("audio_sensitivity_percent"))
    percent_label.pack(side="left")

    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(15, 15))
    nvidia_label.pack(anchor="w", pady=(0, 15))

    result = {"selected": None}

    def on_ok():
        result["selected"] = selected_lang.get()
        if not guardar_idioma_fn(result["selected"]):
            messagebox.showerror(t_func("error"), t_func("could_not_save_language"))
            return

        try:
            threshold_percent = int(sensitivity_var.get())
            if 1 <= threshold_percent <= 100:
                threshold_value = threshold_percent / 100.0
                if not guardar_audio_threshold_fn(threshold_value):
                    messagebox.showerror(t_func("error"), t_func("audio_sensitivity_save_error"))
            else:
                messagebox.showerror(t_func("error"), t_func("audio_sensitivity_range_error"))
                return
        except ValueError:
            messagebox.showerror(t_func("error"), t_func("audio_sensitivity_number_error"))
            return

        root.destroy()

    def on_cancel():
        result["selected"] = None
        root.destroy()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(15, 0))

    ok_button = ttk.Button(btn_frame, text=t_func("ok"), command=on_ok, width=12)
    ok_button.pack(side="left", padx=5)
    cancel_button = ttk.Button(btn_frame, text=t_func("cancel"), command=on_cancel, width=12)
    cancel_button.pack(side="left", padx=5)

    # Reaplicar textos desde el idioma seleccionado para mantener todo sincronizado.
    update_ui_texts()

    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(15, 15))

    footer_frame = ttk.Frame(main_frame)
    footer_frame.pack(fill="x", pady=(0, 0))

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
            except Exception as e:
                print(f"No se pudo cargar el logo de GitHub: {e}")
        except Exception as e:
            print(f"Error al cargar el logo de GitHub: {e}")

    copyright_label = ttk.Label(
        footer_frame,
        text="ADAS3 Server v0.7 |  Copyright (C) 2026 GNU GPL 3.0",
        font=("Arial", 8),
        foreground="gray",
    )
    copyright_label.pack(side="left")

    root.mainloop()
    return result["selected"]
