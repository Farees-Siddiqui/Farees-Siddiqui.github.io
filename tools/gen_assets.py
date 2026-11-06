#!/usr/bin/env python3
"""Generate every image on the Strata demo site with the real distortion pipeline.

    python3 tools/gen_assets.py          (from website/)

Writes assets/img/... and assets/data.json. Every image is a real Strata output:
  - slider: the Commerce invoice at difficulty 0.0 ... 1.0 (seed 7, realistic pool),
    six answer-key boxes carried through each warp by the pipeline's point map;
  - gallery: the medical lab report under each effect at full severity, the lab_005
    row box carried through the warp;
  - pipeline / stack images copied from the poster's generated assets.
"""
import json, shutil, sys
from pathlib import Path
import fitz, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/farees/Masters/Structured_Data/CASCON/strata_v2/src")
from strata import distort

SITE = Path(__file__).resolve().parent.parent
IMG = SITE / "assets" / "img"
for d in ("slider", "gallery", "pipeline"): (IMG / d).mkdir(parents=True, exist_ok=True)
POSTER = SITE.parent / "Team_Draft" / "draft_D" / "assets"
CASE = Path("/home/farees/Masters/Structured_Data/CASCON/CaseStudy/pdf/doc-med-1-lab-report.pdf")
INVOICE = SITE.parent / "Team_Draft/case_studies/commerce/pdf/doc-com-1-invoice.pdf"
DPI, SEED = 300, 7
COLS = [(37, 84, 214), (214, 72, 37), (22, 140, 92), (156, 54, 181), (200, 130, 0), (0, 140, 170)]
KEYS = [("INV-2026-49354", "invoice_004.invoice_no"), ("2026-07-15", "invoice_004.issue_date"),
        ("1Z7222695482", "shipment_004.tracking_no"), ("128.97", "line_005.line_total"),
        ("330.65", "line_006.line_total"), ("519.37", "invoice_004.total")]


def save_jpg(arr_or_im, path, width):
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(arr_or_im)
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(path, quality=84, optimize=True, progressive=True)


def corners(b):
    return np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], float)


def draw(arr, polys, cols, lw):
    im = Image.fromarray(arr); d = ImageDraw.Draw(im)
    for p, c in zip(polys, cols):
        if p is not None: d.polygon([tuple(q) for q in p], outline=c, width=lw)
    return im


def render(pdf, clip=None):
    page = fitz.open(pdf)[0]
    pix = page.get_pixmap(dpi=DPI, clip=fitz.Rect(*clip) if clip else None)
    return page, np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()


def mapped(r, pts, n):
    mp, ok = r.map_points_checked(pts)
    return [mp[i*4:(i+1)*4] if ok[i*4:(i+1)*4].all() else None for i in range(n)]


data = {"slider": [], "gallery": [], "keys": [{"value": v, "field": f, "colour": "#%02x%02x%02x" % c} for (v, f), c in zip(KEYS, COLS)]}

# ---------- slider: invoice, difficulty 0..1 ----------
page, inv = render(INVOICE)
s = DPI / 72
words = page.get_text("words")
inv = inv[: int(max(w[3] for w in words) * s + 0.5 * DPI)].copy()
boxes = []
for val, _ in KEYS:
    w = next(w for w in words if w[4] == val)
    boxes.append([w[0]*s-8, w[1]*s-6, w[2]*s+8, w[3]*s+6])
pts = np.concatenate([corners(b) for b in boxes])
lw = max(5, inv.shape[1] // 350)
for k in range(11):
    diff = round(k / 10, 1)
    if diff == 0:
        out, polys, ops = inv, [corners(b) for b in boxes], []
    else:
        r = distort.apply_recipe(inv, distort.sample_recipe(difficulty=diff, seed=SEED, pool="realistic"))
        out, polys, ops = r.image, mapped(r, pts, len(boxes)), [st["op"] for st in r.recipe["steps"]]
    save_jpg(draw(out, polys, COLS, lw), IMG / "slider" / f"d{k:02d}.jpg", 1800)
    data["slider"].append({"difficulty": diff, "src": f"assets/img/slider/d{k:02d}.jpg", "ops": ops,
                           "boxes_in_frame": sum(p is not None for p in polys)})
    print("slider", diff, ops)

# ---------- gallery: lab report, each effect at full severity ----------
CROP = (28, 40, 584, 310)
lpage, lab = render(CASE, CROP)
h = next(w for w in lpage.get_text("words") if w[4] == "HbA1c")
row = [w for w in lpage.get_text("words") if abs(w[1] - h[1]) < 2]
rb = [(min(w[0] for w in row) - CROP[0]) * s - 18, (min(w[1] for w in row) - CROP[1]) * s - 12,
      (max(w[2] for w in row) - CROP[0]) * s + 18, (max(w[3] for w in row) - CROP[1]) * s + 12]
save_jpg(draw(lab, [corners(rb)], [(47, 69, 200)], 7), IMG / "gallery" / "clean.jpg", 900)
for op in distort.list_ops():
    name = op["name"]
    try:
        r = distort.apply_recipe(lab, {"seed": SEED, "steps": [{"op": name, "severity": 1.0}]})
    except Exception as e:                      # report, never hide
        print("FAILED", name, e); continue
    poly = mapped(r, corners(rb), 1)
    save_jpg(draw(r.image, poly, [(47, 69, 200)], 7), IMG / "gallery" / f"{name}.jpg", 900)
    data["gallery"].append({"name": name, "label": {"jpeg": "JPEG"}.get(name, name.replace("_", " ")), "category": op["category"],
                            "summary": op["summary"].split(": ", 1)[-1], "moves_points": op["moves_points"],
                            "src": f"assets/img/gallery/{name}.jpg", "recipe": r.recipe["steps"][0]})
    print("gallery", name)

# ---------- pipeline + stack images from the poster ----------
for src, dst in [(POSTER / "zoom/doc_clean.png", "doc_clean.jpg"), (POSTER / "zoom/doc_distorted.png", "doc_distorted.jpg"),
                 (POSTER / "distort/pair_clean.png", "pair_clean.jpg"), (POSTER / "distort/pair_distorted.png", "pair_distorted.jpg"),
                 (POSTER / "distort/combo_a_boxes_crop.png", "stack.jpg")]:
    save_jpg(Image.open(src), IMG / "pipeline" / dst, 1800)
shutil.copy(POSTER / "logo/ontariotech_crop.png", IMG / "ontariotech.png")
data["stack_recipe"] = json.loads((POSTER / "distort/combo_a.recipe.json").read_text())["recipe"]   # recipe only; no local paths

# ---------- answer key for the stacked bill of lading (from the logistics case study) ----------
LOG = SITE.parent / "Team_Draft/case_studies/logistics/json"
def stack_answer_key():
    docs = json.loads((LOG / "documents.json").read_text()); docs = docs["documents"] if isinstance(docs, dict) else docs
    m = next(d for d in docs if d["document_id"] == "doc-log-1-bill-of-lading")
    objs = json.loads((LOG / "objects.json").read_text()); objs = objs["objects"] if isinstance(objs, dict) else objs
    if isinstance(objs, dict): objs = [x for v in objs.values() for x in v]
    rec = {o["object_id"]: o for o in objs if o["object_id"] in m["scope"]}
    # the boxed tokens on the image (same rule as draft_D/tools/gen_combo.py) and the record field each comes from
    import re
    words = [w[4] for w in fitz.open(SITE.parent / "Team_Draft/case_studies/logistics/pdf/doc-log-1-bill-of-lading.pdf")[0].get_text("words")]
    boxed = [t for t in words if re.fullmatch(r"[\d.,\-/]{4,}", t)]
    def field_of(t):
        for oid, o in rec.items():
            for k, v in o["attributes"].items():
                sv = str(v)
                if sv == t or sv.split()[0] == t: return f"{oid}.{k}", v
                try:
                    if float(t.replace(",", "")) == float(v): return f"{oid}.{k}", v
                except (TypeError, ValueError): pass
        return None, None
    vals = []
    for t in boxed:
        f, v = field_of(t)
        assert f, f"boxed token {t} not found in the records"
        vals.append({"printed": t, "field": f, "value": v})
    return {"document_id": m["document_id"], "root_object": m["root_object"], "scope": m["scope"],
            "joins_on_page": m["joins_on_page"], "boxed": vals}
data["stack_key"] = stack_answer_key()
(SITE / "assets" / "data.json").write_text(json.dumps(data, indent=1))
# data.js so the page also works when opened from disk (fetch is blocked on file://)
(SITE / "assets" / "data.js").write_text("window.STRATA = " + json.dumps(data) + ";\n")
print("done:", len(data["slider"]), "slider frames,", len(data["gallery"]), "effects")
