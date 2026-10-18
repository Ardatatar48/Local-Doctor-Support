"""
Minimal PDF report writer. No extra installs needed (only Pillow, which you already have).
Used by app.py and app_tk.py:  make_report_pdf(...)
"""
import io
import os
import re
import textwrap

from PIL import Image

PAGE_W, PAGE_H, MARGIN = 595, 842, 50  # A4 in points
WRAP = 88  # max characters per line (Helvetica 10 pt)

# Helvetica only supports Latin-1/cp1252, so map other characters to simple ones
_TRANS = str.maketrans({
    "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I",
    "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
    "•": "-", "…": "...", "→": "->", "≥": ">=", "≤": "<=",
})


def _enc(text):
    text = text.translate(_TRANS).replace("\t", "    ")
    text = re.sub(r"[\x00-\x1f]", " ", text)
    return text.encode("cp1252", "replace")


def _esc(b):
    return b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


class _Layout:
    def __init__(self):
        self.pages = [[]]
        self.y = PAGE_H - MARGIN

    def _need(self, h):
        if self.y - h < MARGIN:
            self.pages.append([])
            self.y = PAGE_H - MARGIN

    def text(self, s, font="F1", size=10, leading=14):
        self._need(leading)
        self.y -= leading
        self.pages[-1].append(
            b"BT /%s %d Tf %d %d Td (" % (font.encode(), size, MARGIN, self.y)
            + _esc(_enc(s)) + b") Tj ET"
        )

    def para(self, s, **kw):
        for raw in (s.splitlines() or [""]):
            for ln in (textwrap.wrap(raw, WRAP) or [""]):
                self.text(ln, **kw)

    def images(self, items):
        """items = [(dw, dh), ...]  placed side by side as /Im1, /Im2 ..."""
        row_h = max(h for _, h in items)
        self._need(row_h + 10)
        self.y -= row_h
        x = MARGIN
        for i, (w, h) in enumerate(items, 1):
            self.pages[-1].append(b"q %d 0 0 %d %d %d cm /Im%d Do Q" % (w, h, x, self.y, i))
            x += w + 15
        self.y -= 10


def _jpeg(path, max_w=260, max_h=240):
    im = Image.open(path).convert("RGB")
    im.thumbnail((1200, 1200))
    pw, ph = im.size
    scale = min(max_w / pw, max_h / ph)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue(), pw, ph, max(1, int(pw * scale)), max(1, int(ph * scale))


def make_report_pdf(path, patient_name, age, created, report, analysis, image_path=None,
                    highlight_path=None):
    analysis = re.sub(r"\*\*", "", analysis or "")
    analysis = re.sub(r"^#+\s*", "", analysis, flags=re.M)

    L = _Layout()
    L.text("MRI + Report Analysis", "F2", 16, 22)
    L.text(f"Patient: {patient_name}    Age: {age}")
    L.text(f"Date: {created}")
    L.text("MRI file: " + (os.path.basename(image_path) if image_path else "none (report-only analysis)"))
    L.y -= 8

    jpegs = []  # [(bytes, pixel_w, pixel_h, draw_w, draw_h), ...]
    for p, w in ((image_path, 240), (highlight_path, 240)):
        if p and os.path.exists(p):
            try:
                jpegs.append(_jpeg(p, max_w=w, max_h=240))
            except Exception:
                pass
    if jpegs:
        L.images([(j[3], j[4]) for j in jpegs])
        if len(jpegs) == 2:
            L.text("Left: original MRI.  Right: suspected area highlighted in red (AI estimate, not a measurement).",
                   "F3", 8, 11)
            L.y -= 6

    L.text("Patient report", "F2", 12, 20)
    L.para(report or "-")
    L.y -= 8
    L.text("Model analysis", "F2", 12, 20)
    L.para(analysis or "-")
    L.y -= 12
    L.para("This output is AI-generated decision support, not a diagnosis. "
           "The final decision belongs to the doctor.", font="F3", size=8, leading=11)

    # ---- assemble PDF objects
    # obj 1 = catalog, obj 2 = pages (filled in at the end), obj 3-5 = fonts
    objs = [
        None,
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>",
    ]
    xobj = []
    for i, (data, pw, ph, _, _) in enumerate(jpegs, 1):
        objs.append(
            b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
            b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (pw, ph, len(data))
            + data + b"\nendstream"
        )
        xobj.append(b"/Im%d %d 0 R" % (i, len(objs)))
    img_ref = (b" /XObject << " + b" ".join(xobj) + b" >>") if xobj else b""

    kids = []
    for ops in L.pages:
        data = b"\n".join(ops)
        objs.append(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
        c_num = len(objs)
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >>%s >> /Contents %d 0 R >>"
            % (PAGE_W, PAGE_H, img_ref, c_num)
        )
        kids.append(b"%d 0 R" % len(objs))

    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = b"<< /Type /Pages /Kids [" + b" ".join(kids) + b"] /Count %d >>" % len(kids)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)
    return path
