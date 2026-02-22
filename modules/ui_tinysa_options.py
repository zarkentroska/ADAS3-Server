import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from modules.ui_window_icon import apply_window_icon


def show_advanced_interval_dialog(
    parent,
    t_func,
    default_sweeps,
    advanced_intervals_file,
    prefills,
):
    """Diálogo para configurar hasta 5 intervalos personalizados."""
    dialog = tk.Toplevel(parent)
    dialog.title(t_func("advanced_interval_title"))
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    apply_window_icon(dialog)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.lift()
    dialog.focus_force()

    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text=t_func("up_to_5_intervals")).grid(row=0, column=0, columnspan=5, pady=(0, 10))

    entries = []
    for i in range(5):
        start_var = tk.StringVar()
        stop_var = tk.StringVar()
        sweeps_var = tk.StringVar(value=str(default_sweeps))
        ttk.Label(frame, text=t_func("interval", i + 1)).grid(row=i + 1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(frame, text=t_func("start")).grid(row=i + 1, column=1, sticky="e")
        ttk.Entry(frame, textvariable=start_var, width=10).grid(row=i + 1, column=2, padx=5, pady=2)
        ttk.Label(frame, text=t_func("stop")).grid(row=i + 1, column=3, sticky="e")
        ttk.Entry(frame, textvariable=stop_var, width=10).grid(row=i + 1, column=4, padx=5, pady=2)
        ttk.Label(frame, text=t_func("sweeps")).grid(row=i + 1, column=5, sticky="e")
        ttk.Entry(frame, textvariable=sweeps_var, width=6).grid(row=i + 1, column=6, padx=5, pady=2)
        entries.append((start_var, stop_var, sweeps_var))

    for (start_var, stop_var, sweeps_var), saved in zip(entries, prefills):
        start_var.set(str(saved.get("start_mhz", "")))
        stop_var.set(str(saved.get("stop_mhz", "")))
        sweeps_var.set(str(saved.get("sweeps", default_sweeps)))

    result = {"ranges": None}

    def on_ok():
        ranges = []
        for idx, (start_var, stop_var, sweeps_var) in enumerate(entries, start=1):
            start_text = start_var.get().strip()
            stop_text = stop_var.get().strip()
            if not start_text and not stop_text:
                continue
            if not start_text or not stop_text:
                messagebox.showerror(t_func("error"), t_func("complete_start_end", idx))
                return
            try:
                start_val = float(start_text)
                stop_val = float(stop_text)
                sweeps_val = int(sweeps_var.get().strip())
            except ValueError:
                messagebox.showerror(t_func("error"), t_func("invalid_values", idx))
                return
            if stop_val <= start_val:
                messagebox.showerror(t_func("error"), t_func("end_must_be_greater", idx))
                return
            if sweeps_val <= 0:
                messagebox.showerror(t_func("error"), t_func("sweeps_must_be_positive", idx))
                return
            ranges.append((start_val, stop_val, sweeps_val))

        if not ranges:
            messagebox.showerror(t_func("error"), t_func("enter_valid_interval"))
            return

        result["ranges"] = ranges
        try:
            with open(advanced_intervals_file, "w", encoding="utf-8") as f:
                json.dump(
                    [{"start_mhz": r[0], "stop_mhz": r[1], "sweeps": r[2]} for r in ranges],
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[TinySA] No se pudo guardar configuración avanzada: {e}")
        dialog.destroy()

    def on_cancel():
        result["ranges"] = None
        dialog.destroy()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=7, column=0, columnspan=5, pady=(15, 0))
    ttk.Button(btn_frame, text=t_func("ok"), command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t_func("cancel"), command=on_cancel, width=12).pack(side="left", padx=5)

    dialog.wait_window()
    return result["ranges"]


def show_tinysa_menu(
    t_func,
    advanced_intervals_file,
    default_sweeps,
    last_advanced_intervals,
):
    """Muestra el selector gráfico para TinySA."""
    loaded_intervals = list(last_advanced_intervals)
    if os.path.exists(advanced_intervals_file):
        try:
            with open(advanced_intervals_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    loaded_intervals = data
        except Exception as e:
            print(f"[TinySA] No se pudo leer configuración avanzada previa: {e}")

    root = tk.Tk()
    root.title(t_func("tinysa_mode_selection"))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    apply_window_icon(root)

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    selection_var = tk.StringVar(value="preset1")
    custom_start = tk.StringVar()
    custom_stop = tk.StringVar()

    ttk.Label(main_frame, text=t_func("select_mode"), font=("Arial", 11, "bold")).pack(anchor="w")
    options_frame = ttk.Frame(main_frame)
    options_frame.pack(fill="x", pady=10)

    ttk.Radiobutton(options_frame, text=t_func("fpv_normal"), variable=selection_var, value="preset1").pack(
        anchor="w", pady=2
    )
    ttk.Radiobutton(options_frame, text=t_func("fpv_5g_detection_mode"), variable=selection_var, value="preset5gdet").pack(
        anchor="w", pady=2
    )
    ttk.Radiobutton(options_frame, text=t_func("custom_range"), variable=selection_var, value="custom").pack(
        anchor="w", pady=2
    )

    custom_frame = ttk.Frame(options_frame)
    custom_frame.pack(anchor="w", padx=20, pady=(0, 5))
    ttk.Label(custom_frame, text=t_func("start_mhz")).grid(row=0, column=0, sticky="w")
    custom_start_entry = ttk.Entry(custom_frame, textvariable=custom_start, width=10, state="disabled")
    custom_start_entry.grid(row=0, column=1, padx=5)
    ttk.Label(custom_frame, text=t_func("stop_mhz")).grid(row=0, column=2, sticky="w")
    custom_stop_entry = ttk.Entry(custom_frame, textvariable=custom_stop, width=10, state="disabled")
    custom_stop_entry.grid(row=0, column=3, padx=5)

    ttk.Radiobutton(options_frame, text=t_func("advanced_range"), variable=selection_var, value="advanced").pack(
        anchor="w", pady=2
    )

    result = {"selection": None, "custom": None, "advanced": None}

    def update_custom_state(*_):
        state = "normal" if selection_var.get() == "custom" else "disabled"
        custom_start_entry.configure(state=state)
        custom_stop_entry.configure(state=state)

    selection_var.trace_add("write", update_custom_state)

    def finish_and_close():
        root.quit()

    def on_ok():
        sel = selection_var.get()
        if sel == "custom":
            try:
                start_val = float(custom_start.get())
                stop_val = float(custom_stop.get())
            except ValueError:
                messagebox.showerror(t_func("error"), t_func("enter_numeric_values"))
                return
            if stop_val <= start_val:
                messagebox.showerror(t_func("error"), t_func("end_greater_than_start"))
                return
            result["selection"] = sel
            result["custom"] = (start_val, stop_val)
            finish_and_close()
        elif sel == "advanced":
            try:
                root.attributes("-disabled", True)
            except Exception:
                pass
            ranges = show_advanced_interval_dialog(
                root,
                t_func=t_func,
                default_sweeps=default_sweeps,
                advanced_intervals_file=advanced_intervals_file,
                prefills=loaded_intervals,
            )
            try:
                root.attributes("-disabled", False)
            except Exception:
                pass
            if ranges is None:
                root.focus_set()
                return
            result["selection"] = sel
            result["advanced"] = ranges
            finish_and_close()
        else:
            result["selection"] = sel
            finish_and_close()

    def on_cancel():
        result["selection"] = None
        finish_and_close()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(10, 0))
    ttk.Button(btn_frame, text=t_func("ok"), command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t_func("cancel"), command=on_cancel, width=12).pack(side="left", padx=5)

    def on_close():
        result["selection"] = None
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    if root.winfo_exists():
        root.destroy()

    return result, loaded_intervals
