"""
MRI + Report Analysis System - Tkinter version (no Gradio needed)
Required packages: torch transformers accelerate qwen-vl-utils pillow
"""
import os
import sqlite3
import sys
import threading
import uuid
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from highlight import GROUND_PROMPT, draw_highlight, parse_boxes
from pdf_report import make_report_pdf

print("Python:", sys.executable, flush=True)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hastalar.db")
IMG_DIR = os.path.join(os.path.dirname(DB_PATH), "mr_goruntuleri")
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------- DATABASE
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            age INTEGER, created TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS records(
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER,
            image_path TEXT, report_text TEXT, analysis TEXT, created TEXT)""")


def ensure_highlight_column():
    """Older databases do not have the highlight_path column yet; add it."""
    with db() as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(records)")]
        if "highlight_path" not in cols:
            c.execute("ALTER TABLE records ADD COLUMN highlight_path TEXT")


def now():
    return datetime.now().isoformat(timespec="seconds")


def patient_list():
    with db() as c:
        rows = c.execute("SELECT id, name, age FROM patients ORDER BY id DESC").fetchall()
    return [f"{r[0]} - {r[1]} ({r[2]})" for r in rows]


# --------------------------------------------------------------------- MODEL
PROMPT = """You are a clinical decision-support assistant helping a doctor.
You will receive an MRI image and the patient's medical report. Answer in English and use these headings:

1. IMAGE FINDINGS: Describe what you see in the MRI (region, possible herniation/pathology). If possible,
   give an approximate percentage/grade (e.g. "about 30% canal narrowing") and state that it is only a rough estimate.
2. REPORT SUMMARY: Extract the key information from the report (diseases, current medications, allergies, etc.).
3. RECOMMENDATIONS (for the doctor): Possible treatment/medication/physical therapy suggestions based on the
   findings and the report, drug interactions and points to watch out for.
4. CONCLUSION: State that this output is not a definitive diagnosis and the final decision belongs to the doctor.

PATIENT REPORT:
{report}
"""

PROMPT_TEXT = """You are a clinical decision-support assistant helping a doctor.
There is NO MRI image for this patient; only the report below is available. Answer in English and use these headings:

1. TENTATIVE ASSESSMENT: Possible causes based on the complaints and examination findings (e.g. likelihood of
   lumbar disc herniation, muscle spasm, etc.). State that without an MRI this is only an estimate.
2. REPORT SUMMARY: Extract the key information (chronic diseases, current medications, allergies).
3. RECOMMENDATIONS (for the doctor): Possible treatment/medication/physical therapy suggestions, drug interactions,
   precautions due to allergies and chronic diseases, and which additional tests (e.g. MRI) should be requested.
4. CONCLUSION: State that this output is not a definitive diagnosis and the final decision belongs to the doctor.

PATIENT REPORT:
{report}
"""

model = None
processor = None


def load_model():
    global model, processor
    print("Loading model...", flush=True)
    print("GPU available:", torch.cuda.is_available(), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    # Resolution limit so very large images do not fill the memory
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
    )
    print("Model ready.", flush=True)


def locate_problem(image_path):
    """Ask the model where the abnormal area is; returns boxes in ORIGINAL image pixel coordinates."""
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": GROUND_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)
    # Qwen2.5-VL returns coordinates in the RESIZED image space (patches of 14 px)
    grid = inputs["image_grid_thw"][0]
    resized_h, resized_w = int(grid[1]) * 14, int(grid[2]) * 14
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)
    out = out[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    print("Grounding answer:", raw, flush=True)
    sx, sy = img.width / resized_w, img.height / resized_h
    return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy, label)
            for x1, y1, x2, y2, label in parse_boxes(raw)]


def make_highlight(image_path):
    """Returns the path of a copy of the MRI with the suspected area painted red (or None)."""
    boxes = locate_problem(image_path)
    if not boxes:
        return None
    out = os.path.join(IMG_DIR, f"hl_{uuid.uuid4().hex}.png")
    return draw_highlight(image_path, boxes, out)


def analyze(image_path, report):
    if image_path:
        content = [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": PROMPT.format(report=report or "No report provided.")},
        ]
    else:  # No MRI: estimate based on the report only
        content = [{"type": "text", "text": PROMPT_TEXT.format(report=report or "No report provided.")}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


# ------------------------------------------------------------------- UI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MRI Analysis Assistant (homework prototype)")
        self.geometry("1100x750")
        self.image_path = None

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Patient:").pack(side="left")
        self.combo = ttk.Combobox(top, values=patient_list(), width=35, state="readonly")
        self.combo.pack(side="left", padx=4)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.fill_report())
        ttk.Label(top, text="New name:").pack(side="left", padx=(12, 0))
        self.name = ttk.Entry(top, width=18)
        self.name.pack(side="left", padx=4)
        ttk.Label(top, text="Age:").pack(side="left")
        self.age = ttk.Entry(top, width=5)
        self.age.pack(side="left", padx=4)
        ttk.Button(top, text="Add patient", command=self.add_patient).pack(side="left", padx=4)
        ttk.Button(top, text="History", command=self.show_history).pack(side="left", padx=4)

        top2 = ttk.Frame(self)
        top2.pack(fill="x", padx=8)
        ttk.Button(top2, text="Save PDF report", command=self.save_pdf).pack(side="left", padx=4)
        ttk.Button(top2, text="Delete record...", command=self.delete_record).pack(side="left", padx=4)
        ttk.Button(top2, text="Delete patient", command=self.delete_patient).pack(side="left", padx=4)

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=8)
        left = ttk.Frame(mid)
        left.pack(side="left", fill="y")
        ttk.Button(left, text="Select MRI (PNG)", command=self.pick_image).pack(pady=4)
        imgs = ttk.Frame(left)
        imgs.pack()
        f1 = ttk.Frame(imgs)
        f1.pack(side="left", padx=4)
        ttk.Label(f1, text="MRI").pack()
        self.img_label = ttk.Label(f1, text="No image")
        self.img_label.pack()
        f2 = ttk.Frame(imgs)
        f2.pack(side="left", padx=4)
        ttk.Label(f2, text="Suspected area (AI estimate)").pack()
        self.hl_label = ttk.Label(f2, text="-")
        self.hl_label.pack()

        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=10)
        ttk.Label(right, text="Patient report:").pack(anchor="w")
        self.report = tk.Text(right, height=10)
        self.report.pack(fill="x")

        self.btn = ttk.Button(self, text="Analyze", command=self.run_analysis)
        self.btn.pack(pady=6)
        self.status = ttk.Label(self, text="Loading model, please wait...")
        self.status.pack()
        self.output = tk.Text(self, height=18, wrap="word")
        self.output.pack(fill="both", expand=True, padx=8, pady=6)

        self.btn.config(state="disabled")
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        load_model()
        self.after(0, lambda: (self.btn.config(state="normal"),
                               self.status.config(text="Model ready.")))

    def _show(self, label, img, attr):
        if img is None:
            label.config(image="", text="-")
            setattr(self, attr, None)
            return
        img = img.copy()
        img.thumbnail((260, 260))
        photo = ImageTk.PhotoImage(img)
        setattr(self, attr, photo)  # keep a reference so Tk does not discard it
        label.config(image=photo, text="")

    def fill_report(self):
        pid = int(self.combo.get().split(" - ")[0])
        with db() as c:
            row = c.execute(
                "SELECT report_text FROM records WHERE patient_id=? ORDER BY id DESC LIMIT 1",
                (pid,),
            ).fetchone()
        self.report.delete("1.0", "end")
        self.report.insert("end", row[0] if row and row[0] else "")
        with db() as c:
            ana = c.execute(
                "SELECT created, analysis, image_path, highlight_path FROM records "
                "WHERE patient_id=? AND analysis<>'' ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
        self.output.delete("1.0", "end")
        mri = hl = None
        self.image_path = None
        if ana:
            self.output.insert("end", f"[Saved analysis - {ana[0]}]\n\n{ana[1]}")
            if ana[2] and os.path.exists(ana[2]):
                self.image_path = ana[2]
                mri = Image.open(ana[2]).convert("RGB")
            if ana[3] and os.path.exists(ana[3]):
                hl = Image.open(ana[3]).convert("RGB")
        self._show(self.img_label, mri, "tk_img")
        self._show(self.hl_label, hl, "tk_hl")

    def _selected_pid(self):
        if not self.combo.get():
            messagebox.showwarning("Warning", "Select a patient first.")
            return None
        return int(self.combo.get().split(" - ")[0])

    def save_pdf(self):
        pid = self._selected_pid()
        if pid is None:
            return
        with db() as c:
            p = c.execute("SELECT name, age FROM patients WHERE id=?", (pid,)).fetchone()
            r = c.execute(
                "SELECT created, image_path, report_text, analysis, highlight_path FROM records "
                "WHERE patient_id=? AND analysis<>'' ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
        if not p or not r:
            return messagebox.showwarning("Warning", "No analysis found for this patient. Run an analysis first.")
        safe = "".join(ch if ch.isalnum() else "_" for ch in p[0])
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", initialfile=f"report_{safe}.pdf", filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        make_report_pdf(path, p[0], p[1], r[0], r[2], r[3], r[1], r[4])
        messagebox.showinfo("Done", f"PDF saved:\n{path}")
        try:
            os.startfile(path)  # opens the PDF on Windows
        except Exception:
            pass

    def delete_patient(self):
        pid = self._selected_pid()
        if pid is None:
            return
        if not messagebox.askyesno(
                "Confirm", f"Delete '{self.combo.get()}' and ALL their records?\nThis cannot be undone."):
            return
        with db() as c:
            c.execute("DELETE FROM records WHERE patient_id=?", (pid,))
            c.execute("DELETE FROM patients WHERE id=?", (pid,))
        self.combo.config(values=patient_list())
        self.combo.set("")
        self.report.delete("1.0", "end")
        self.output.delete("1.0", "end")
        self.image_path = None
        self._show(self.img_label, None, "tk_img")
        self._show(self.hl_label, None, "tk_hl")

    def delete_record(self):
        pid = self._selected_pid()
        if pid is None:
            return
        with db() as c:
            rows = c.execute(
                "SELECT id, created, image_path FROM records WHERE patient_id=? ORDER BY id DESC",
                (pid,)).fetchall()
        if not rows:
            return messagebox.showinfo("Info", "No records for this patient.")
        win = tk.Toplevel(self)
        win.title("Delete record")
        win.geometry("420x300")
        ttk.Label(win, text="Select a record to delete:").pack(pady=6)
        lb = tk.Listbox(win, height=10)
        lb.pack(fill="both", expand=True, padx=8)
        for r in rows:
            lb.insert("end", f"{r[0]} - {r[1]} ({'MRI' if r[2] else 'no MRI'})")

        def do_delete():
            sel = lb.curselection()
            if not sel:
                return
            if not messagebox.askyesno("Confirm", "Delete this record permanently?", parent=win):
                return
            with db() as c:
                c.execute("DELETE FROM records WHERE id=?", (rows[sel[0]][0],))
            win.destroy()
            self.fill_report()

        ttk.Button(win, text="Delete selected", command=do_delete).pack(pady=8)

    def add_patient(self):
        name = self.name.get().strip()
        if not name:
            return messagebox.showwarning("Warning", "Name is required.")
        age = int(self.age.get()) if self.age.get().isdigit() else 0
        with db() as c:
            c.execute("INSERT INTO patients(name, age, created) VALUES(?,?,?)",
                      (name, age, now()))
        vals = patient_list()
        self.combo.config(values=vals)
        self.combo.set(vals[0])

    def pick_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.image_path = path
        self._show(self.img_label, Image.open(path).convert("RGB"), "tk_img")
        self._show(self.hl_label, None, "tk_hl")

    def run_analysis(self):
        if not self.combo.get():
            return messagebox.showwarning("Warning", "Select or add a patient first.")
        self.btn.config(state="disabled")
        self.status.config(text="Analyzing...")
        report = self.report.get("1.0", "end").strip()
        pid = int(self.combo.get().split(" - ")[0])
        threading.Thread(target=self._analyze, args=(pid, report), daemon=True).start()

    def _analyze(self, pid, report):
        hl = None
        try:
            result = analyze(self.image_path, report)
            if self.image_path:  # MRI selected: also mark the suspected area
                try:
                    hl = make_highlight(self.image_path)
                except Exception as e:
                    print("Highlight failed:", e, flush=True)
            with db() as c:
                c.execute("""INSERT INTO records(patient_id, image_path, report_text,
                             analysis, created, highlight_path) VALUES(?,?,?,?,?,?)""",
                          (pid, self.image_path, report, result, now(), hl))
        except Exception as e:
            result = f"Error: {e}"
        self.after(0, lambda: self._done(result, hl))

    def _done(self, result, hl=None):
        self._show(self.hl_label, Image.open(hl).convert("RGB") if hl else None, "tk_hl")
        self.output.delete("1.0", "end")
        self.output.insert("end", result)
        self.btn.config(state="normal")
        self.status.config(text="Done.")

    def show_history(self):
        if not self.combo.get():
            return
        pid = int(self.combo.get().split(" - ")[0])
        with db() as c:
            rows = c.execute("""SELECT created, report_text, analysis FROM records
                                WHERE patient_id=? ORDER BY id DESC""", (pid,)).fetchall()
        text = "\n\n".join(f"=== {r[0]} ===\nREPORT: {r[1]}\n\nANALYSIS:\n{r[2]}" for r in rows) \
            or "No records."
        self.output.delete("1.0", "end")
        self.output.insert("end", text)


if __name__ == "__main__":
    init_db()
    ensure_highlight_column()
    App().mainloop()
