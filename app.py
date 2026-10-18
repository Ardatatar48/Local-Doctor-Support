"""
MRI + Report Analysis System (HOMEWORK PROTOTYPE)
Qwen2.5-VL examines the image and the report text and gives decision-support suggestions to the doctor.
Database: SQLite (patients + records tables)

Installation:
    pip install torch transformers accelerate qwen-vl-utils gradio pillow
Run:
    python app.py
"""
import os
import sqlite3
import uuid
from datetime import datetime

import gradio as gr
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from highlight import GROUND_PROMPT, draw_highlight, parse_boxes
from pdf_report import make_report_pdf

# Use 3B on a weaker GPU, 7B on a stronger one
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hastalar.db")
IMG_DIR = os.path.join(BASE_DIR, "mr_goruntuleri")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------- DATABASE
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS patients(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                created TEXT)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS records(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                image_path TEXT,
                report_text TEXT,
                analysis TEXT,
                created TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id))"""
        )


def add_patient(name, age):
    with db() as c:
        cur = c.execute(
            "INSERT INTO patients(name, age, created) VALUES(?,?,?)",
            (name, int(age or 0), datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def patient_choices():
    with db() as c:
        rows = c.execute("SELECT id, name, age FROM patients ORDER BY id DESC").fetchall()
    return [f"{r[0]} - {r[1]} ({r[2]})" for r in rows]


def ensure_highlight_column():
    """Older databases do not have the highlight_path column yet; add it."""
    with db() as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(records)")]
        if "highlight_path" not in cols:
            c.execute("ALTER TABLE records ADD COLUMN highlight_path TEXT")


def save_record(patient_id, image_path, report, analysis, highlight_path=None):
    with db() as c:
        c.execute(
            "INSERT INTO records(patient_id, image_path, report_text, analysis, created, highlight_path) "
            "VALUES(?,?,?,?,?,?)",
            (patient_id, image_path, report, analysis,
             datetime.now().isoformat(timespec="seconds"), highlight_path),
        )


def _open_if_exists(path):
    if path and os.path.exists(path):
        with Image.open(path) as im:
            return im.convert("RGB")
    return None


def load_patient(choice):
    """When a patient is selected: load the latest report and (if any) the latest saved analysis."""
    if not choice:
        return "", "", None
    pid = int(choice.split(" - ")[0])
    with db() as c:
        rep = c.execute(
            "SELECT report_text FROM records WHERE patient_id=? ORDER BY id DESC LIMIT 1",
            (pid,),
        ).fetchone()
        ana = c.execute(
            "SELECT created, analysis, highlight_path FROM records WHERE patient_id=? AND analysis<>'' "
            "ORDER BY id DESC LIMIT 1",
            (pid,),
        ).fetchone()
    return (
        rep[0] if rep and rep[0] else "",
        f"[Saved analysis - {ana[0]}]\n\n{ana[1]}" if ana else "",
        _open_if_exists(ana[2]) if ana else None,
    )


def history(choice):
    if not choice:
        return "No patient selected."
    pid = int(choice.split(" - ")[0])
    with db() as c:
        rows = c.execute(
            "SELECT created, report_text, analysis, image_path FROM records WHERE patient_id=? ORDER BY id DESC",
            (pid,),
        ).fetchall()
    if not rows:
        return "No records for this patient."
    return "\n\n".join(
        f"=== {r[0]} ===\nMRI: {r[3] or 'none'}\nREPORT: {r[1]}\n\nANALYSIS:\n{r[2] or '(not analyzed yet)'}" for r in rows
    )


def record_choices(choice):
    if not choice:
        return []
    pid = int(choice.split(" - ")[0])
    with db() as c:
        rows = c.execute(
            "SELECT id, created, image_path FROM records WHERE patient_id=? ORDER BY id DESC",
            (pid,),
        ).fetchall()
    return [f"{r[0]} - {r[1]} ({'MRI' if r[2] else 'no MRI'})" for r in rows]


def _remove_own_image(path):
    """Only delete MRI copies that this app saved itself (inside IMG_DIR)."""
    try:
        if path and os.path.dirname(os.path.abspath(path)) == os.path.abspath(IMG_DIR):
            if os.path.exists(path):
                os.remove(path)
    except OSError:
        pass


def delete_record_db(rec_id):
    with db() as c:
        row = c.execute("SELECT image_path, highlight_path FROM records WHERE id=?", (rec_id,)).fetchone()
        c.execute("DELETE FROM records WHERE id=?", (rec_id,))
    if row:
        _remove_own_image(row[0])
        _remove_own_image(row[1])


def delete_patient_db(pid):
    with db() as c:
        paths = []
        for r in c.execute("SELECT image_path, highlight_path FROM records WHERE patient_id=?", (pid,)):
            paths += [r[0], r[1]]
        c.execute("DELETE FROM records WHERE patient_id=?", (pid,))
        c.execute("DELETE FROM patients WHERE id=?", (pid,))
    for p in paths:
        _remove_own_image(p)


def export_pdf(choice):
    """Create a PDF from the patient's latest saved analysis. Returns (path, message)."""
    if not choice:
        return None, "Select a patient first."
    pid = int(choice.split(" - ")[0])
    with db() as c:
        p = c.execute("SELECT name, age FROM patients WHERE id=?", (pid,)).fetchone()
        r = c.execute(
            "SELECT created, image_path, report_text, analysis, highlight_path FROM records "
            "WHERE patient_id=? AND analysis<>'' ORDER BY id DESC LIMIT 1",
            (pid,),
        ).fetchone()
    if not p or not r:
        return None, "No analysis found for this patient. Run an analysis first."
    safe = "".join(ch if ch.isalnum() else "_" for ch in p[0])
    path = os.path.join(REPORT_DIR, f"report_{safe}_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    make_report_pdf(path, p[0], p[1], r[0], r[2], r[3], r[1], r[4])
    return path, f"PDF created: {path}"


# --------------------------------------------------------------------- MODEL
print("Loading model (the first download may take a few minutes)...", flush=True)
print("GPU available:", torch.cuda.is_available(), flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
# Resolution limit so very large images do not fill the memory
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
)
print("Model ready, starting the interface...", flush=True)

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


def analyze_image(image_path, report):
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
    inputs = processor(
        text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def locate_problem(image_path):
    """Ask the model where the abnormal area is; returns boxes in ORIGINAL image pixel coordinates."""
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": GROUND_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
    ).to(model.device)
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


# ------------------------------------------------------------------- UI
def ui_new_patient(name, age):
    if not name:
        return gr.update(), "Name is required."
    pid = add_patient(name, age)
    choices = patient_choices()
    return gr.update(choices=choices, value=choices[0]), f"Patient added (id={pid})."


def ui_analyze(choice, image, report):
    if not choice:
        return "Select or add a patient first.", None
    pid = int(choice.split(" - ")[0])
    path = None
    if image is not None:  # Save the MRI if uploaded; otherwise only the report is analyzed
        path = os.path.join(IMG_DIR, f"{uuid.uuid4().hex}.png")
        Image.fromarray(image).convert("RGB").save(path)
    result = analyze_image(path, report)
    hl_path = None
    if path:  # MRI uploaded: also mark the suspected area
        try:
            hl_path = make_highlight(path)
        except Exception as e:
            print("Highlight failed:", e, flush=True)
    save_record(pid, path, report, result, hl_path)
    return result, _open_if_exists(hl_path)


def ui_export_pdf(choice):
    return export_pdf(choice)


def ui_delete_record(p_choice, rec_choice, confirm):
    nochange = (gr.update(), gr.update(), gr.update(), gr.update())
    if not p_choice or not rec_choice:
        return (*nochange, "Select a patient and a record first.", confirm)
    if not confirm:
        return (*nochange, "Tick the confirmation box first.", confirm)
    delete_record_db(int(rec_choice.split(" - ")[0]))
    return (
        gr.update(choices=patient_choices(), value=p_choice),
        gr.update(choices=record_choices(p_choice), value=None),
        history(p_choice),
        gr.update(choices=patient_choices()),
        "Record deleted.",
        False,
    )


def ui_delete_patient(p_choice, confirm):
    nochange = (gr.update(), gr.update(), gr.update(), gr.update())
    if not p_choice:
        return (*nochange, "Select a patient first.", confirm)
    if not confirm:
        return (*nochange, "Tick the confirmation box first.", confirm)
    delete_patient_db(int(p_choice.split(" - ")[0]))
    choices = patient_choices()
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=[], value=None),
        "",
        gr.update(choices=choices, value=None),
        "Patient and all their records were deleted.",
        False,
    )


init_db()
ensure_highlight_column()
with gr.Blocks(title="MRI Analysis Assistant") as demo:
    gr.Markdown(
        "# MRI + Report Analysis Assistant\n"
        "⚠️ Homework prototype. Outputs are not a diagnosis and must not be used on real patients."
    )
    with gr.Tab("Analysis"):
        with gr.Row():
            patient = gr.Dropdown(patient_choices(), label="Patient")
            name = gr.Textbox(label="New patient name")
            age = gr.Number(label="Age", precision=0)
            add_btn = gr.Button("Add patient")
        status = gr.Markdown()
        with gr.Row():
            img = gr.Image(type="numpy", label="MRI (PNG)")
            report = gr.Textbox(lines=10, label="Patient report (text)")
        run = gr.Button("Analyze", variant="primary")
        output = gr.Textbox(lines=18, label="Model output")
        hl_img = gr.Image(label="Suspected area highlighted in red (AI estimate, not a measurement)",
                          interactive=False)
        pdf_btn = gr.Button("Download PDF report")
        pdf_file = gr.File(label="PDF report (click to download)")

        add_btn.click(ui_new_patient, [name, age], [patient, status])
        run.click(ui_analyze, [patient, img, report], [output, hl_img])
        patient.change(load_patient, patient, [report, output, hl_img])
        pdf_btn.click(ui_export_pdf, patient, [pdf_file, status])

    with gr.Tab("History") as tab2:
        p2 = gr.Dropdown(patient_choices(), label="Patient")
        show = gr.Button("Show records")
        hist = gr.Textbox(lines=20, label="Records")
        gr.Markdown("### Delete data")
        rec = gr.Dropdown([], label="Record (only needed to delete a single record)")
        confirm = gr.Checkbox(label="I confirm that I want to delete this data permanently")
        with gr.Row():
            del_rec = gr.Button("Delete selected record")
            del_pat = gr.Button("Delete patient and ALL records", variant="stop")
        msg2 = gr.Markdown()

        def on_patient2(choice):
            return history(choice), gr.update(choices=record_choices(choice), value=None)

        tab2.select(lambda: gr.update(choices=patient_choices()), None, p2)
        show.click(on_patient2, p2, [hist, rec])
        p2.change(on_patient2, p2, [hist, rec])
        outs = [p2, rec, hist, patient, msg2, confirm]
        del_rec.click(ui_delete_record, [p2, rec, confirm], outs)
        del_pat.click(ui_delete_patient, [p2, confirm], outs)

if __name__ == "__main__":
    demo.launch(inbrowser=True, allowed_paths=[REPORT_DIR])
