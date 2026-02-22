import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from modules.ui_window_icon import apply_window_icon


def show_yolo_options_window(
    yolo_model_slots,
    yolo_default_slot,
    yolo_default_model_path,
    translate_fn,
    normalize_model_path_fn,
    apply_yolo_model_fn,
):
    """Ventana para gestionar modelos YOLO."""
    root = tk.Tk()
    root.title(translate_fn("yolo_options_title"))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    apply_window_icon(root)

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    ttk.Label(main_frame, text=translate_fn("available_models"), font=("Arial", 11, "bold")).pack(anchor="w")

    slots_frame = ttk.Frame(main_frame)
    slots_frame.pack(fill="both", expand=True, pady=(10, 15))

    total_slots = len(yolo_model_slots)
    slots_per_page = 5
    total_pages = max(1, (total_slots + slots_per_page - 1) // slots_per_page)
    current_page = tk.IntVar(value=0)

    path_vars = [tk.StringVar(value=slot.get("path", "")) for slot in yolo_model_slots]
    desc_vars = [tk.StringVar(value=slot.get("description", "")) for slot in yolo_model_slots]
    selected_var = tk.IntVar(value=yolo_default_slot)

    def browse_file(idx):
        filepath = filedialog.askopenfilename(
            title=translate_fn("select_yolo_model"),
            filetypes=[(translate_fn("yolo_models"), "*.pt"), (translate_fn("all_files"), "*.*")],
            parent=root,
        )
        if filepath:
            path_vars[idx].set(filepath)

    rows_container = ttk.Frame(slots_frame)
    rows_container.pack(fill="both", expand=True)

    def build_page(page_idx):
        for child in rows_container.winfo_children():
            child.destroy()

        start_idx = page_idx * slots_per_page
        end_idx = min(start_idx + slots_per_page, total_slots)

        for idx in range(start_idx, end_idx):
            frame_slot = ttk.Frame(rows_container, padding=5)
            frame_slot.pack(fill="x", pady=3)

            ttk.Radiobutton(frame_slot, variable=selected_var, value=idx).grid(row=0, column=0, rowspan=2, padx=(0, 8))
            ttk.Label(frame_slot, text=translate_fn("model", idx + 1)).grid(row=0, column=1, sticky="w")
            entry_path = ttk.Entry(frame_slot, textvariable=path_vars[idx], width=45)
            entry_path.grid(row=0, column=2, padx=5, sticky="we")
            ttk.Button(frame_slot, text=translate_fn("browse"), command=lambda i=idx: browse_file(i)).grid(
                row=0, column=3, padx=5
            )
            ttk.Label(frame_slot, text=translate_fn("description")).grid(row=1, column=1, sticky="e", pady=2)
            ttk.Entry(frame_slot, textvariable=desc_vars[idx], width=45).grid(row=1, column=2, padx=5, sticky="we")
            frame_slot.columnconfigure(2, weight=1)

    nav_frame = ttk.Frame(main_frame)
    nav_frame.pack(fill="x", pady=(5, 5))

    page_label_var = tk.StringVar()

    def update_page_label():
        page_label_var.set(translate_fn("page", current_page.get() + 1, total_pages))

    def go_prev():
        if current_page.get() > 0:
            current_page.set(current_page.get() - 1)
            build_page(current_page.get())
            update_page_label()

    def go_next():
        if current_page.get() < total_pages - 1:
            current_page.set(current_page.get() + 1)
            build_page(current_page.get())
            update_page_label()

    ttk.Button(nav_frame, text="◀", width=3, command=go_prev).pack(side="left")
    ttk.Label(nav_frame, textvariable=page_label_var).pack(side="left", padx=10)
    ttk.Button(nav_frame, text="▶", width=3, command=go_next).pack(side="left")

    build_page(0)
    update_page_label()

    status_var = tk.StringVar(value="")

    def sync_slots():
        for idx in range(len(yolo_model_slots)):
            yolo_model_slots[idx]["path"] = path_vars[idx].get().strip()
            yolo_model_slots[idx]["description"] = desc_vars[idx].get().strip()

    def apply_action(save_default=False, reset_default=False):
        sync_slots()
        slot_idx = selected_var.get()

        if reset_default:
            yolo_model_slots[0]["path"] = yolo_default_model_path
            yolo_model_slots[0]["description"] = "Modelo por defecto"
            path_vars[0].set(yolo_default_model_path)
            desc_vars[0].set("Modelo por defecto")
            slot_idx = 0
            save_default = True

        path = yolo_model_slots[slot_idx]["path"]
        if not path:
            messagebox.showerror(translate_fn("error"), translate_fn("model_empty", slot_idx + 1))
            return

        normalized_path = normalize_model_path_fn(path)
        if not normalized_path or not os.path.exists(normalized_path):
            messagebox.showerror(translate_fn("error"), translate_fn("file_not_found", path))
            return

        if apply_yolo_model_fn(
            normalized_path,
            save_default=save_default,
            selected_slot=slot_idx if save_default else None,
        ):
            status_var.set(translate_fn("model_updated"))
            root.destroy()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(5, 10))

    ttk.Button(btn_frame, text=translate_fn("load_model"), command=lambda: apply_action(False)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=translate_fn("load_and_save_default"), command=lambda: apply_action(True)).pack(
        side="left", padx=5
    )
    ttk.Button(btn_frame, text=translate_fn("load_default_config"), command=lambda: apply_action(True, True)).pack(
        side="left", padx=5
    )
    ttk.Button(btn_frame, text=translate_fn("cancel"), command=root.destroy).pack(side="right", padx=5)

    ttk.Label(main_frame, textvariable=status_var, foreground="#0077cc").pack(anchor="w")

    root.mainloop()
