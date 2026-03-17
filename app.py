import streamlit as st
import openai
import base64
from PIL import Image, ImageDraw, ImageFont
import os
import json
import io
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd


from google import genai
from google.genai import types

# ------------------ CONFIG ------------------
st.set_page_config(page_title="PPE Compliance Analysis", layout="wide", page_icon="🦺")

openai.api_key = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)

# ------------------ STYLES ------------------
st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:800; color:var(--primary-color); margin:0; }
    .main-subtitle { font-size:0.85rem; color:var(--text-color); opacity:0.6; margin-top:4px; }
    .analysis-complete {
        background-color:var(--secondary-background-color);
        border:1px solid var(--primary-color); border-radius:10px;
        padding:14px 20px; margin:16px 0;
        display:flex; align-items:center; justify-content:space-between;
    }
    .analysis-complete-title { color:var(--primary-color); font-size:1.1rem; font-weight:700; }
    .report-generated { color:#3fb950; font-size:0.82rem; font-weight:600; }
    .critical-alert {
        background-color:var(--secondary-background-color);
        border:1px solid #f85149; border-left:4px solid #f85149;
        border-radius:8px; padding:12px 16px; margin:10px 0 18px 0;
    }
    .critical-alert-title  { color:#f85149; font-weight:700; font-size:0.95rem; }
    .critical-alert-sub    { color:#f85149; opacity:0.75; font-size:0.82rem; margin-top:2px; }
    .metrics-row { display:flex; gap:12px; margin:16px 0; }
    .metric-card {
        flex:1; background-color:var(--secondary-background-color);
        border:1px solid rgba(128,128,128,0.2); border-radius:10px;
        padding:18px 14px 14px 14px; text-align:center;
    }
    .metric-number-blue   { font-size:2.2rem; font-weight:800; color:var(--primary-color); }
    .metric-number-red    { font-size:2.2rem; font-weight:800; color:#f85149; }
    .metric-number-yellow { font-size:2.2rem; font-weight:800; color:#d29922; }
    .metric-number-green  { font-size:2.2rem; font-weight:800; color:#3fb950; }
    .metric-label { font-size:0.78rem; color:var(--text-color); opacity:0.6; margin-top:4px; }
    .overview-header { color:var(--text-color); font-weight:700; font-size:1rem; margin:20px 0 12px 0; }
    .img-card-meta { background-color:var(--secondary-background-color); border-radius:0 0 8px 8px; padding:10px 12px; margin-top:-4px; }
    .img-card-verdict { font-weight:700; font-size:0.85rem; }
    .img-card-fname { font-size:0.75rem; opacity:0.55; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .img-card-notes { font-size:0.8rem; opacity:0.75; margin-top:6px; line-height:1.4; }
    .upload-section {
        background-color:var(--secondary-background-color);
        border:1px solid rgba(128,128,128,0.2); border-radius:10px;
        padding:20px; margin-bottom:16px;
    }
    .report-section-title {
        color:var(--primary-color); font-size:1rem; font-weight:700;
        margin:16px 0 8px 0;
        border-bottom:1px solid rgba(128,128,128,0.2); padding-bottom:6px;
    }
    .violation-card-high   { background-color:var(--secondary-background-color); border:1px solid #f85149; border-left:4px solid #f85149; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
    .violation-card-medium { background-color:var(--secondary-background-color); border:1px solid #d29922; border-left:4px solid #d29922; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
    .violation-card-low    { background-color:var(--secondary-background-color); border:1px solid #3fb950; border-left:4px solid #3fb950; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
    .compliant-item {
        background-color:var(--secondary-background-color);
        border:1px solid #3fb950; border-left:4px solid #3fb950;
        border-radius:6px; padding:8px 12px; margin-bottom:6px;
        color:#3fb950; font-size:0.85rem;
    }
    .score-card {
        background-color:var(--secondary-background-color);
        border:1px solid rgba(128,128,128,0.2); border-radius:10px;
        padding:16px; margin-bottom:14px; display:flex; align-items:center; gap:24px;
    }
    .score-label   { font-size:0.75rem; color:var(--text-color); opacity:0.6; }
    .score-summary { font-size:0.85rem; color:var(--text-color); opacity:0.7; margin-top:4px; }
    .worker-card {
        background-color:var(--secondary-background-color);
        border:1px solid rgba(128,128,128,0.2); border-radius:8px;
        padding:12px 14px; margin-bottom:8px;
    }
    .timestamp-badge {
        background-color:rgba(128,128,128,0.15); border-radius:4px;
        padding:2px 7px; font-size:0.75rem; font-weight:600; font-family:monospace;
    }
    .violation-badge {
        display:inline-block; background:#d29922; color:#000;
        font-size:0.72rem; font-weight:700; border-radius:4px;
        padding:2px 7px; margin-bottom:6px;
    }
    .hse-tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:700; margin-right:4px; margin-bottom:4px; }
    .hse-tag-a { background:rgba(248,81,73,0.15);  color:#f85149; border:1px solid #f85149; }
    .hse-tag-b { background:rgba(210,153,34,0.15); color:#d29922; border:1px solid #d29922; }
    .hse-tag-c { background:rgba(63,185,80,0.15);  color:#3fb950; border:1px solid #3fb950; }
    .bocw-ref  { font-size:0.73rem; opacity:0.55; margin-bottom:4px; font-style:italic; }
</style>
""", unsafe_allow_html=True)

#Image  Analysis Prompt
SAFETY_SYSTEM_PROMPT = """
You are an expert construction site safety inspector with deep knowledge of OSHA,
ISO 45001, and BOCW Act 1998.

Use these EXACT HSE Risk Category rules when classifying violations:

CATEGORY A (Fatal / Amputation Risk) — assign when:
- Worker at height without safety belt / full-body harness
- Unsafe or damaged scaffolding in use
- Missing edge protection / guardrails on elevated surfaces
- Workers performing height/hot/lifting work without work permit
- Crane/hoist/hydra operating without a certified signalman/banksman
- Damaged wire rope or lifting slings/gear in use
- Hot work (welding/grinding/gas cutting) without fire extinguisher within 3 metres
- Workers riding unsafely on material shifting vehicles
- Gas cylinders unsecured or stored horizontally
- Operating heavy equipment without valid license
- Grinding/cutting machine used without safety guard
- Damaged or missing barricading around active hazard zones

CATEGORY B (Injury Risk) — assign when:
- Workers without PPE: helmet, safety vest, gloves, goggles
- Power cables without proper male-female plugs
- Damaged electrical cables in use
- Materials dumped blocking access paths or walkways
- Missing wheel stopper on vehicles/compressors
- Damaged safety belt in use (single hook where double hook required)
- DG with open doors, missing panel covers
- Unsafe jack supports under heavy loads

CATEGORY C (Health / Environmental) — assign when:
- Oil leakage from DG or machinery causing soil contamination
- Concrete debris or construction waste dumped openly on site
- Waste materials in open area / poor housekeeping
- Disorganized store / poor 5S implementation
- Missing drinking water signage or cleanliness issues
- Workers exposed to dust/fumes without respiratory protection

BOUNDING BOX INSTRUCTIONS:
For every violation, use this STRICT method:
1. Mentally divide the image into a 10x10 grid (0-9 left→right, 0-9 top→bottom)
2. Find which grid cell the violation CENTER falls in
3. Estimate how many cells wide/tall the object spans
4. Convert: grid_col/10 = x_center, grid_row/10 = y_center

Rules:
- Bbox must tightly enclose ONLY the specific object/person causing the violation
- Single standing worker: width 0.08-0.15, height 0.20-0.35
- Small tool/object: width 0.10-0.20, height 0.10-0.20
- NEVER use 0.5,0.5 unless the object is truly in the center
- NEVER make width/height larger than 0.5 unless violation spans the whole image

Return ONLY valid JSON:
{
  "overall_safety_score": <0-100>,
  "overall_verdict": "<PASS | FAIL | NEEDS_REVIEW>",
  "items_analyzed": ["list of what you see"],
  "compliant_items": ["what is correctly followed"],
  "violations": [
    {
      "issue": "<description>",
      "severity": "<HIGH | MEDIUM | LOW>",
      "hse_category": "<A | B | C>",
      "bocw_reference": "<BOCW Act 1998 rule/clause>",
      "recommendation": "<corrective action>",
      "bbox": {
        "image_index": <1-based int>,
        "x_center": <0.0-1.0>,
        "y_center": <0.0-1.0>,
        "width":    <0.0-1.0>,
        "height":   <0.0-1.0>,
        "location_hint": "<top-left | top-center | top-right | center-left | center | center-right | bottom-left | bottom-center | bottom-right>"
      }
    }
  ],
  "per_image_analysis": [
    {
      "image_index": <1-based int>,
      "items_found": ["what was seen"],
      "verdict": "<SAFE | WARNING | CRITICAL>",
      "notes": "<brief observation>"
    }
  ],
  "hse_category_breakdown": {
    "A_fatal_count": <int>,
    "B_injury_count": <int>,
    "C_environmental_count": <int>
  },
  "summary": "<2-3 line overall summary>"
}
"""

#Gemini Video Prompt

VIDEO_SAFETY_PROMPT = """
You are an expert construction site safety inspector with deep knowledge of OSHA,
ISO 45001, BOCW Act 1998, and HSE standards.

Analyze this construction site VIDEO comprehensively and note key safety moments.

CATEGORY A (Fatal / Amputation Risk):
- Worker at height without safety belt/harness, unsafe scaffolding,
  missing guardrails, no work permit for height/hot/lifting work,
  crane/hoist without signalman, damaged wire rope,
  hot work without fire extinguisher,
  workers riding unsafely on vehicles, unsecured gas cylinders,
  heavy equipment without license, grinding without guard,
  damaged/missing barricading

CATEGORY B (Injury Risk):
- Workers without PPE (helmet/vest/gloves/goggles),
  damaged electrical cables, blocked walkways,
  missing wheel stopper, damaged safety belt,
  DG with open doors, unsafe jack supports

CATEGORY C (Health / Environmental):
- Oil leakage, construction debris, poor housekeeping,
  poor 5S, missing drinking water signage,
  dust/fumes without respiratory protection

Also check: PPE compliance, fall protection, hot work fire safety,
lifting operations, housekeeping, electrical hazards, supervision.

Return ONLY valid JSON:
{
  "overall_safety_score": <0-100>,
  "overall_verdict": "<PASS | FAIL | NEEDS_REVIEW>",
  "total_workers_detected": <int>,
  "compliant_items": ["what is correctly followed"],
  "violations": [
    {
      "issue": "<description>",
      "severity": "<HIGH | MEDIUM | LOW>",
      "hse_category": "<A | B | C>",
      "bocw_reference": "<BOCW Act 1998 rule/clause>",
      "recommendation": "<corrective action>"
    }
  ],
  "per_image_analysis": [
    {
      "image_index": <int>,
      "timestamp": "<MM:SS>",
      "items_found": ["what was seen"],
      "verdict": "<SAFE | WARNING | CRITICAL>",
      "active_checks": {
        "ppe_compliant": <true|false>,
        "fall_protection_ok": <true|false>,
        "hot_work_fire_safety_ok": <true|false>,
        "lifting_safety_ok": <true|false>,
        "housekeeping_ok": <true|false>
      },
      "notes": "<brief observation>"
    }
  ],
  "hse_category_breakdown": {
    "A_fatal_count": <int>,
    "B_injury_count": <int>,
    "C_environmental_count": <int>
  },
  "summary": "<2-3 line overall summary>"
}
"""

# Video Critical Frames Annotation using Gemini
GEMINI_FRAME_BBOX_PROMPT = """
You are an expert construction site safety inspector.

Analyze this construction site image for ALL PPE and safety violations.
For every violation, return a precise bounding box using box_2d in [y_min, x_min, y_max, x_max] format
where all values are between 0 and 1000 (normalized to image dimensions).

Use these HSE categories:
- Category A: Fatal/Amputation risk (height work, scaffolding, crane, hot work, gas cylinders)
- Category B: Injury risk (missing PPE: helmet/vest/gloves/goggles, electrical, blocked walkways)
- Category C: Health/Environmental (oil leakage, debris, poor housekeeping, dust/fumes)

Return ONLY valid JSON:
{
  "violations": [
    {
      "issue": "<specific description>",
      "severity": "<HIGH | MEDIUM | LOW>",
      "hse_category": "<A | B | C>",
      "bocw_reference": "<BOCW Act 1998 rule/clause>",
      "recommendation": "<corrective action>",
      "box_2d": [y_min, x_min, y_max, x_max]
    }
  ], 
  "verdict": "<SAFE | WARNING | CRITICAL>",
  "items_found": ["list of what you see"],
  "notes": "<brief observation>"
}
"""

# ======================================================
# HELPERS — GENERAL
# ======================================================
def verdict_icon(v):
    return {"SAFE":"✅","WARNING":"⚠️","CRITICAL":"🔴"}.get(v.upper(), "❓")

def verdict_color(v):
    return {"SAFE":"#3fb950","WARNING":"#d29922","CRITICAL":"#f85149"}.get(v.upper(), "#8b949e")

def hse_badge(cat):
    cat   = str(cat).upper()
    label = {"A":"A – Fatal","B":"B – Injury","C":"C – Env"}.get(cat, cat)
    css   = {"A":"hse-tag-a","B":"hse-tag-b","C":"hse-tag-c"}.get(cat,"hse-tag-b")
    return f'<span class="hse-tag {css}">{label}</span>'

# ======================================================
# HELPERS — IMAGE ENCODING
# ======================================================
def get_image_mime(image_file) -> str:
    image_file.seek(0)
    try:
        img = Image.open(image_file)
        fmt = img.format.lower() if img.format else "jpeg"
        return {"jpeg":"image/jpeg","jpg":"image/jpeg",
                "png":"image/png","gif":"image/gif","webp":"image/webp"}.get(fmt,"image/jpeg")
    except Exception:
        return "image/jpeg"

def encode_image(image_file) -> tuple:
    mime = get_image_mime(image_file)
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode("utf-8"), mime

# Gpt 4.1 Image Analysis
def analyze_images(images_data: list, context: str = "") -> dict:
    content = [{
        "type": "text",
        "text": (f"Context: {context}\n\nAnalyze all images for PPE / construction safety compliance. "
                 "For each image, provide per-image breakdown AND bbox for every violation found."
                 if context else
                 "Analyze all these images for PPE / construction safety compliance. "
                 "For each image, provide per-image breakdown AND bbox for every violation found.")
    }]
    for b64, mime in images_data:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
        })
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
            {"role": "user",   "content": content}
        ],
        max_tokens=3000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# GPT 4.1 boxes annotation
HINT_MAP = {
    "top-left":     (0.20, 0.20), "top-center":    (0.50, 0.18),
    "top-right":    (0.80, 0.20), "center-left":   (0.18, 0.50),
    "center":       (0.50, 0.50), "center-right":  (0.82, 0.50),
    "bottom-left":  (0.20, 0.80), "bottom-center": (0.50, 0.82),
    "bottom-right": (0.80, 0.80),
}


def draw_violation_ellipses(image_bytes: bytes, violations: list, image_index: int) -> bytes:
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    image_violations = [
        v for v in violations
        if isinstance(v.get("bbox"), dict)
        and v["bbox"].get("image_index") == image_index
    ]
    CAT_COLORS = {"A": (255,50,50), "B": (255,210,0), "C": (63,200,80)}

    for v in image_violations:
        bbox = v["bbox"]
        xc   = float(bbox.get("x_center", 0.5))
        yc   = float(bbox.get("y_center", 0.5))
        bw   = float(bbox.get("width",    0.25))
        bh   = float(bbox.get("height",   0.25))
        hint = bbox.get("location_hint", "").lower().strip()
        if (abs(xc - 0.5) < 0.03 and abs(yc - 0.5) < 0.03) and hint in HINT_MAP:
            xc, yc = HINT_MAP[hint]
        bw = max(0.10, min(bw, 0.55))
        bh = max(0.10, min(bh, 0.55))
        cx = int(xc * w); cy = int(yc * h)
        ew = int(bw * w); eh = int(bh * h)
        x0 = max(4, cx - ew//2); y0 = max(4, cy - eh//2)
        x1 = min(w-4, cx + ew//2); y1 = min(h-4, cy + eh//2)
        cat   = v.get("hse_category","B").upper()
        color = CAT_COLORS.get(cat,(255,210,0))
        line_w = max(3, min(w,h)//90)
        for offset in range(line_w):
            draw.ellipse([x0-offset, y0-offset, x1+offset, y1+offset], outline=color, width=1)
        sev   = v.get("severity","")
        label = f"Cat {cat} | {sev}"
        try:
            tb = draw.textbbox((0,0), label, font=font)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
        except Exception:
            tw, th = len(label)*7, 14
        lx = max(0,x0); ly = max(0,y0-th-10)
        draw.rectangle([lx-2,ly-2,lx+tw+8,ly+th+4], fill=color)
        draw.text((lx+3,ly+1), label, fill=(0,0,0), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def add_violation_overlay(image_bytes: bytes, verdict: str) -> bytes:
    verdict = verdict.upper()
    if verdict == "SAFE":
        return image_bytes
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w,h), (0,0,0,0))
    draw    = ImageDraw.Draw(overlay)
    border  = max(5, min(w,h)//60)
    draw.rectangle([0,0,w-1,h-1], outline=(255,200,0,210), width=border)
    strip_h = max(26, h//16)
    draw.rectangle([0,0,w,strip_h], fill=(255,200,0,170))
    label = "! WARNING" if verdict == "WARNING" else "!! CRITICAL VIOLATION"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((0,0), label, font=font)
    tx = (w-(bbox[2]-bbox[0]))//2; ty = (strip_h-(bbox[3]-bbox[1]))//2
    draw.text((tx,ty), label, fill=(60,30,0,255), font=font)
    combined = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO(); combined.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# Extacting frames from videos
def extract_frames_at_fps(video_path: str, sample_fps: float = 1.0) -> tuple:
    import cv2
    cap       = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval  = max(1, int(video_fps / sample_fps))
    frames, frame_idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % interval == 0:
            _, buf = cv2.imencode(".jpg", frame)
            frames.append((frame_idx, buf.tobytes()))
        frame_idx += 1
    cap.release()
    return frames, video_fps, frame_idx

def frame_index_to_timestamp(frame_idx: int, video_fps: float) -> str:
    s = int(frame_idx / video_fps)
    return f"{s//60:02d}:{s%60:02d}"

def extract_frames_at_timestamps(video_path: str, timestamps: list) -> list:
    """Extract frames at specific MM:SS timestamps from video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    for ts in timestamps:
        try:
            parts    = ts.split(":")
            seconds  = int(parts[0]) * 60 + int(parts[1])
            frame_no = int(seconds * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame)
                frames.append({"timestamp": ts, "bytes": buf.tobytes()})
        except Exception:
            continue
    cap.release()
    return frames

def analyze_video_with_gemini(video_bytes: bytes, context: str = "",
                               detail_level: str = "Standard") -> dict:
    import time
    depth = {
        "Quick (key moments only)": "Analyze only the 3–5 most critical safety moments.",
        "Standard":                  "Analyze roughly 1 key moment per 10 seconds of video.",
        "Deep (every scene change)": "Analyze every distinct scene or camera angle change."
    }
    prompt = VIDEO_SAFETY_PROMPT
    if context:
        prompt = f"Context: {context}\n\n" + prompt
    prompt += f"\n\nAnalysis depth: {depth.get(detail_level,'')}"

    max_retries = 5
    base_delay  = 10  # seconds

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                    prompt,
                ],
                config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            err_str = str(e)
            is_retryable = any(c in err_str for c in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500"])
            if is_retryable and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)  # 10s → 20s → 40s → 80s
                # st.warning(f"⏳ Gemini busy (attempt {attempt+1}/{max_retries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Gemini video analysis failed after maximum retries.")


def analyze_frame_with_gemini_bbox(frame_bytes: bytes, timestamp: str) -> dict:
    """Send a single frame to Gemini 2.5 Pro for precise bbox violation detection."""
    import time
    prompt = GEMINI_FRAME_BBOX_PROMPT + f"\n\nThis frame is from timestamp: {timestamp}"

    max_retries = 4
    base_delay  = 8  # seconds

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            err_str = str(e)
            is_retryable = any(c in err_str for c in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500"])
            if is_retryable and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)  # 8s → 16s → 32s
                time.sleep(wait)
            else:
                return {"violations": [], "verdict": "SAFE", "notes": f"Gemini error: {str(e)}"}
    return {"violations": [], "verdict": "SAFE", "notes": "Gemini bbox failed after retries."}


def draw_gemini_bboxes(image_bytes: bytes, violations: list) -> bytes:
    """Draw bounding boxes from Gemini box_2d [y1,x1,y2,x2] on 0-1000 scale."""
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    CAT_COLORS = {"A": (255,50,50), "B": (255,210,0), "C": (63,200,80)}
    for v in violations:
        box = v.get("box_2d")
        if not box or len(box) != 4:
            continue
        y1 = int(box[0]/1000*h); x1 = int(box[1]/1000*w)
        y2 = int(box[2]/1000*h); x2 = int(box[3]/1000*w)
        cat   = v.get("hse_category","B").upper()
        color = CAT_COLORS.get(cat,(255,210,0))
        sev   = v.get("severity","")
        label = f"Cat {cat} | {sev}"
        line_w = max(2, min(w,h)//200)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=line_w)
        try:
            tb = draw.textbbox((0,0), label, font=font)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
        except Exception:
            tw, th = len(label)*7, 14
        lx = max(0,x1); ly = max(0,y1-th-6)
        draw.rectangle([lx,ly,lx+tw+8,ly+th+6], fill=color)
        draw.text((lx+3,ly+2), label, fill=(0,0,0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

# Plotly Visualization
def render_video_visualizations(result: dict):
    score     = result.get("overall_safety_score", 0)
    per_image = result.get("per_image_analysis", [])
    hse_bd    = result.get("hse_category_breakdown", {})

    st.markdown('<div class="report-section-title">📊 Analysis Visualizations</div>', unsafe_allow_html=True)
    col_gauge, col_donut = st.columns(2)

    with col_gauge:
        gc = "#3fb950" if score >= 75 else "#d29922" if score >= 50 else "#f85149"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={"text":"Overall Safety Score","font":{"size":14}},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":gc},
                "steps":[
                    {"range":[0,50],   "color":"rgba(248,81,73,0.15)"},
                    {"range":[50,75],  "color":"rgba(210,153,34,0.15)"},
                    {"range":[75,100], "color":"rgba(63,185,80,0.15)"},
                ],
                "threshold":{"line":{"color":gc,"width":4},"thickness":0.75,"value":score},
            },
            number={"suffix":"/100","font":{"size":28,"color":gc}},
        ))
        fig.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

    with col_donut:
        a = hse_bd.get("A_fatal_count",0)
        b = hse_bd.get("B_injury_count",0)
        c = hse_bd.get("C_environmental_count",0)
        if a+b+c > 0:
            labels=["Cat A – Fatal","Cat B – Injury","Cat C – Env"]
            values=[a,b,c]; colors=["#f85149","#d29922","#3fb950"]
        else:
            labels,values,colors=["No Violations"],[1],["#3fb950"]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55,
            marker_colors=colors, textinfo="label+value",
            hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        fig2.update_layout(
            title={"text":"Violations by HSE Category","font":{"size":14}},
            height=260, margin=dict(l=10,r=10,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
            legend=dict(orientation="h",y=-0.15), showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    if per_image:
        vmap   = {"SAFE":100,"WARNING":50,"CRITICAL":0}
        tslabs = [item.get("timestamp",f"{i+1}") for i,item in enumerate(per_image)]
        vscrs  = [vmap.get(item.get("verdict","WARNING").upper(),50) for item in per_image]
        vvds   = [item.get("verdict","WARNING").upper() for item in per_image]
        fig4   = go.Figure()
        fig4.add_trace(go.Scatter(
            x=tslabs, y=vscrs, mode="lines+markers",
            line=dict(color="#58a6ff",width=2),
            marker=dict(color=[verdict_color(v) for v in vvds],size=10,
                        line=dict(width=1,color="#fff")),
            text=vvds,
            hovertemplate="Time: %{x}<br>Verdict: %{text}<extra></extra>",
        ))
        fig4.add_hrect(y0=75,y1=110,fillcolor="rgba(63,185,80,0.07)",line_width=0,
                       annotation_text="Safe Zone",annotation_font_size=10)
        fig4.add_hrect(y0=25,y1=75, fillcolor="rgba(210,153,34,0.07)",line_width=0)
        fig4.add_hrect(y0=0, y1=25, fillcolor="rgba(248,81,73,0.07)", line_width=0)
        fig4.update_layout(
            title={"text":"Frame-by-Frame Safety Timeline","font":{"size":14}},
            xaxis={"title":"Timestamp","tickangle":-30},
            yaxis={"title":"Safety Level","range":[-5,115],
                   "tickvals":[0,50,100],"ticktext":["CRITICAL","WARNING","SAFE"]},
            height=280, margin=dict(l=20,r=20,t=40,b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
        )
        st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div style="border-bottom:1px solid rgba(128,128,128,0.2);padding-bottom:14px;margin-bottom:20px;">
    <div class="main-title">🦺 PPE Compliance Analysis</div>
    <div class="main-subtitle">AI-powered construction site safety monitoring</div>
</div>
""", unsafe_allow_html=True)

tab_images, tab_video = st.tabs(["📷 Image Analysis", "🎬 Video Analysis"])

# ======================================================
# TAB 1 — IMAGE ANALYSIS
# ======================================================
with tab_images:
    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**📁 Upload Images**")
        uploaded_files = st.file_uploader(
            "Upload construction site / PPE photos",
            type=["jpg","jpeg","png","webp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        context = st.text_area("Optional context", height=80,
                                placeholder="e.g. 'Check harnesses against EN 361 standard'...")
        analyze_btn = st.button("🔍 Analyze PPE Compliance", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} image(s) uploaded:**")
            cols = st.columns(min(len(uploaded_files), 3))
            for i, f in enumerate(uploaded_files):
                with cols[i % 3]:
                    f.seek(0); st.image(f.read(), caption=f.name, use_container_width=True); f.seek(0)

    with right_col:
        if analyze_btn and uploaded_files:
            with st.spinner("🔄 GPT-4o — Detecting violations + locations..."):
                images_data = []
                for f in uploaded_files:
                    f.seek(0); b64, mime = encode_image(f); images_data.append((b64, mime))
                result = analyze_images(images_data, context)

            violations    = result.get("violations", [])
            per_image_raw = result.get("per_image_analysis", [])
            final_bytes_list = []

            with st.spinner("🎨 Drawing violation ellipses on annotated images..."):
                for idx, f in enumerate(uploaded_files):
                    f.seek(0)
                    raw_bytes   = f.read()
                    image_index = idx + 1
                    annotated   = draw_violation_ellipses(raw_bytes, violations, image_index)
                    verdict = "SAFE"
                    if idx < len(per_image_raw):
                        verdict = per_image_raw[idx].get("verdict","SAFE")
                    annotated = add_violation_overlay(annotated, verdict)
                    final_bytes_list.append(annotated)

            st.session_state["result"]              = result
            st.session_state["uploaded_files_data"] = [
                {"name": f.name, "bytes": final_bytes_list[i]}
                for i, f in enumerate(uploaded_files)
            ]
            st.session_state["show_full_report"] = False

        if "result" in st.session_state:
            result     = st.session_state["result"]
            per_image  = result.get("per_image_analysis", [])
            violations = result.get("violations", [])
            files_data = st.session_state.get("uploaded_files_data", [])
            hse_bd     = result.get("hse_category_breakdown", {})

            critical_count = sum(1 for x in per_image if x.get("verdict","").upper() == "CRITICAL")
            warning_count  = sum(1 for x in per_image if x.get("verdict","").upper() == "WARNING")
            safe_count     = sum(1 for x in per_image if x.get("verdict","").upper() == "SAFE")

            st.markdown("""
            <div class="analysis-complete">
                <div class="analysis-complete-title">✔ Analysis Complete</div>
                <div class="report-generated">● GPT-4o · Violations Marked</div>
            </div>""", unsafe_allow_html=True)

            if critical_count > 0:
                st.markdown(f"""
                <div class="critical-alert">
                    <div class="critical-alert-title">🔴 Critical Safety Violations Detected</div>
                    <div class="critical-alert-sub">{critical_count} critical alert(s) require immediate attention</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card"><div class="metric-number-blue">{len(per_image)}</div><div class="metric-label">Images Analyzed</div></div>
                <div class="metric-card"><div class="metric-number-red">{critical_count}</div><div class="metric-label">Critical</div></div>
                <div class="metric-card"><div class="metric-number-yellow">{warning_count}</div><div class="metric-label">Warnings</div></div>
                <div class="metric-card"><div class="metric-number-green">{safe_count}</div><div class="metric-label">Safe</div></div>
            </div>""", unsafe_allow_html=True)

            # if hse_bd:
                # a = hse_bd.get("A_fatal_count",0)
                # b = hse_bd.get("B_injury_count",0)
                # c = hse_bd.get("C_environmental_count",0)
                # st.markdown(f"""
                # <div class="metrics-row">
                #     <div class="metric-card" style="border-color:#f85149;"><div class="metric-number-red">{a}</div><div class="metric-label">Cat A – Fatal</div></div>
                #     <div class="metric-card" style="border-color:#d29922;"><div class="metric-number-yellow">{b}</div><div class="metric-label">Cat B – Injury</div></div>
                #     <div class="metric-card" style="border-color:#3fb950;"><div class="metric-number-green">{c}</div><div class="metric-label">Cat C – Environmental</div></div>
                # </div>""", unsafe_allow_html=True)

            st.markdown('<div class="overview-header">🖼 Annotated Images Overview</div>', unsafe_allow_html=True)
            grid_cols = st.columns(min(len(files_data), 3))
            for idx, fd in enumerate(files_data):
                with grid_cols[idx % 3]:
                    st.image(fd["bytes"], use_container_width=True)
                    pi = per_image[idx] if idx < len(per_image) else {}
                    v  = pi.get("verdict","SAFE").upper()
                    vc = verdict_color(v)
                    st.markdown(f"""
                    <div class="img-card-meta">
                        <div class="img-card-verdict" style="color:{vc};">{verdict_icon(v)} {v}</div>
                        <div class="img-card-fname">{fd['name']}</div>
                        <div class="img-card-notes">{pi.get('notes','')}</div>
                    </div>""", unsafe_allow_html=True)

            if st.button("📋 View / Hide Full Compliance Report", use_container_width=True):
                st.session_state["show_full_report"] = not st.session_state.get("show_full_report", False)

            if st.session_state.get("show_full_report", False):
                score   = result.get("overall_safety_score", 0)
                verdict = result.get("overall_verdict", "NEEDS_REVIEW")
                gc      = "#3fb950" if score >= 75 else "#d29922" if score >= 50 else "#f85149"
                st.markdown('<div class="report-section-title">🔍 Overall Safety Score</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="score-card">
                    <div>
                        <div class="score-label">Overall Score</div>
                        <div style="font-size:2.8rem;font-weight:800;color:{gc};line-height:1;">{score}<span style="font-size:1.2rem;opacity:0.6;">/100</span></div>
                    </div>
                    <div>
                        <div class="score-label">Verdict</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{gc};margin-top:4px;">{verdict}</div>
                        <div class="score-summary">{result.get('summary','')}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                compliant = result.get("compliant_items", [])
                if compliant:
                    st.markdown('<div class="report-section-title">✅ Compliant Items</div>', unsafe_allow_html=True)
                    for item in compliant:
                        st.markdown(f'<div class="compliant-item">✅ {item}</div>', unsafe_allow_html=True)

                if violations:
                    st.markdown('<div class="report-section-title">⚠️ Violations Detected</div>', unsafe_allow_html=True)
                    for i, v in enumerate(violations):
                        sev     = v.get("severity","MEDIUM").upper()
                        css_cls = {"HIGH":"violation-card-high","MEDIUM":"violation-card-medium","LOW":"violation-card-low"}.get(sev,"violation-card-medium")
                        cat     = v.get("hse_category","B")
                        st.markdown(f"""
                        <div class="{css_cls}">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                                {hse_badge(cat)}
                                <span class="violation-badge">{sev}</span>
                            </div>
                            <div class="bocw-ref">{v.get('bocw_reference','')}</div>
                            <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px;">{v.get('issue','')}</div>
                            <div style="font-size:0.82rem;opacity:0.75;">💡 {v.get('recommendation','')}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown('<div class="report-section-title">🖼 Per-Image Breakdown</div>', unsafe_allow_html=True)
                for idx, pi in enumerate(per_image):
                    fd    = files_data[idx] if idx < len(files_data) else {}
                    fname = fd.get("name", f"Image {idx+1}")
                    v     = pi.get("verdict","SAFE").upper()
                    vc    = verdict_color(v)
                    with st.expander(f"{verdict_icon(v)} Image {idx+1}: {fname} — {v}", expanded=(v=="CRITICAL")):
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if fd.get("bytes"):
                                st.image(fd["bytes"], use_container_width=True)
                        with c2:
                            st.markdown(f"**Verdict:** <span style='color:{vc};font-weight:700;'>{v}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Notes:** {pi.get('notes','')}")
                            items_found = pi.get("items_found", [])
                            if items_found:
                                st.markdown("**Items Found:**")
                                for item in items_found:
                                    st.markdown(f"- {item}")
                            img_violations = [
                                viol for viol in violations
                                if isinstance(viol.get("bbox"), dict)
                                and viol["bbox"].get("image_index") == idx+1
                            ]
                            if img_violations:
                                st.markdown(f"**Violations in this image ({len(img_violations)}):**")
                                for viol in img_violations:
                                    cat = viol.get("hse_category","B")
                                    st.markdown(f"""
                                    <div style="background:rgba(248,81,73,0.08);border-left:3px solid #f85149;padding:6px 10px;margin:4px 0;border-radius:4px;font-size:0.82rem;">
                                        {hse_badge(cat)} <strong>{viol.get('issue','')}</strong><br>
                                        <span style="opacity:0.7;">💡 {viol.get('recommendation','')}</span>
                                    </div>""", unsafe_allow_html=True)

# ======================================================
# TAB 2 — VIDEO ANALYSIS
# ======================================================
with tab_video:
    vid_left, vid_right = st.columns([1, 1.6], gap="large")

    with vid_left:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**🎬 Upload Video**")
        uploaded_video = st.file_uploader(
            "Upload construction site video",
            type=["mp4","mov","avi","mkv"],
            label_visibility="collapsed"
        )
        vid_context = st.text_area("Optional context", height=70,
                                    placeholder="e.g. 'Focus on fall protection and harness usage'...")
        detail_level = st.selectbox(
            "Analysis Depth",
            ["Quick (key moments only)", "Standard", "Deep (every scene change)"],
            index=1
        )
        analyze_video_btn = st.button("🔍 Analyze Video Safety", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_video:
            st.video(uploaded_video)

    with vid_right:
        if analyze_video_btn and uploaded_video:
            uploaded_video.seek(0)
            video_bytes = uploaded_video.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                tmp_video_path = tmp.name

            # ── Pass 1: Gemini full video analysis ──
            with st.spinner("🤖 Analysing video..."):
                try:
                    result_video = analyze_video_with_gemini(video_bytes, vid_context, detail_level)
                except Exception as e:
                    st.error(f"❌ Gemini video analysis failed after all retries: {e}")
                    st.info("💡 Gemini 2.5 Pro is under high demand. Please wait 1–2 minutes and try again.")
                    st.stop()

            # ── Pass 2: Extract flagged frames & get precise bboxes ──
            risky_frames_info = [
                f for f in result_video.get("per_image_analysis", [])
                if f.get("verdict", "").upper() in ["WARNING", "CRITICAL"]
            ]
            risky_timestamps = [f["timestamp"] for f in risky_frames_info if f.get("timestamp")]

            annotated_frames = []
            if risky_timestamps:
                with st.spinner(f"🎯 bbox detection on {len(risky_timestamps)} flagged frame(s)..."):
                    extracted = extract_frames_at_timestamps(tmp_video_path, risky_timestamps)
                    for frame_data in extracted:
                        frame_result    = analyze_frame_with_gemini_bbox(frame_data["bytes"], frame_data["timestamp"])
                        annotated_bytes = draw_gemini_bboxes(frame_data["bytes"], frame_result.get("violations", []))
                        annotated_bytes = add_violation_overlay(annotated_bytes, frame_result.get("verdict", "SAFE"))
                        annotated_frames.append({
                            "timestamp":  frame_data["timestamp"],
                            "bytes":      annotated_bytes,
                            "violations": frame_result.get("violations", []),
                            "verdict":    frame_result.get("verdict", "SAFE"),
                            "notes":      frame_result.get("notes", "")
                        })

            try:
                os.unlink(tmp_video_path)
            except Exception:
                pass

            st.session_state["video_result"]      = result_video
            st.session_state["annotated_frames"]  = annotated_frames
            st.session_state["show_video_report"] = False

        if "video_result" in st.session_state:
            result_video = st.session_state["video_result"]
            violations_v = result_video.get("violations", [])
            per_image_v  = result_video.get("per_image_analysis", [])
            hse_bd_v     = result_video.get("hse_category_breakdown", {})

            critical_v = sum(1 for x in violations_v if x.get("severity","").upper() == "HIGH")
            medium_v   = sum(1 for x in violations_v if x.get("severity","").upper() == "MEDIUM")
            low_v      = sum(1 for x in violations_v if x.get("severity","").upper() == "LOW")

            st.markdown("""
            <div class="analysis-complete">
                <div class="analysis-complete-title">✔ Video Analysis Complete</div>
                <div class="report-generated">● Gemini 2.5 Pro</div>
            </div>""", unsafe_allow_html=True)

            if critical_v > 0:
                st.markdown(f"""
                <div class="critical-alert">
                    <div class="critical-alert-title">🔴 High-Severity Violations Found</div>
                    <div class="critical-alert-sub">{critical_v} HIGH severity issue(s) require immediate action</div>
                </div>""", unsafe_allow_html=True)

            workers_detected = result_video.get("total_workers_detected", 0)
            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card"><div class="metric-number-blue">{workers_detected}</div><div class="metric-label">Workers Detected</div></div>
                <div class="metric-card"><div class="metric-number-red">{critical_v}</div><div class="metric-label">High Severity</div></div>
                <div class="metric-card"><div class="metric-number-yellow">{medium_v}</div><div class="metric-label">Medium</div></div>
                <div class="metric-card"><div class="metric-number-green">{low_v}</div><div class="metric-label">Low</div></div>
            </div>""", unsafe_allow_html=True)

            # if hse_bd_v:
            #     a = hse_bd_v.get("A_fatal_count",0)
            #     b = hse_bd_v.get("B_injury_count",0)
            #     c = hse_bd_v.get("C_environmental_count",0)
            #     st.markdown(f"""
            #     <div class="metrics-row">
            #         <div class="metric-card" style="border-color:#f85149;"><div class="metric-number-red">{a}</div><div class="metric-label">Cat A – Fatal</div></div>
            #         <div class="metric-card" style="border-color:#d29922;"><div class="metric-number-yellow">{b}</div><div class="metric-label">Cat B – Injury</div></div>
            #         <div class="metric-card" style="border-color:#3fb950;"><div class="metric-number-green">{c}</div><div class="metric-label">Cat C – Environmental</div></div>
            #     </div>""", unsafe_allow_html=True)

            

            # ── Annotated Flagged Frames ──
            annotated_frames = st.session_state.get("annotated_frames", [])
            if annotated_frames:
                st.markdown('<div class="report-section-title">🎯 Flagged Frames — Gemini Bbox Annotations</div>', unsafe_allow_html=True)
                frame_cols = st.columns(min(len(annotated_frames), 3))
                for fi, fd in enumerate(annotated_frames):
                    with frame_cols[fi % 3]:
                        st.image(fd["bytes"], use_container_width=True)
                        fv  = fd.get("verdict","SAFE").upper()
                        fvc = verdict_color(fv)
                        st.markdown(f"""
                        <div class="img-card-meta">
                            <div class="img-card-verdict" style="color:{fvc};">{verdict_icon(fv)} {fv}</div>
                            <div class="img-card-fname">⏱ {fd['timestamp']}</div>
                            <div class="img-card-notes">{fd.get('notes','')}</div>
                        </div>""", unsafe_allow_html=True)
                        if fd.get("violations"):
                            with st.expander(f"Violations @ {fd['timestamp']} ({len(fd['violations'])})"):
                                for v in fd["violations"]:
                                    cat = v.get("hse_category","B")
                                    st.markdown(f"""
                                    <div style="background:rgba(248,81,73,0.08);border-left:3px solid #f85149;padding:6px 10px;margin:4px 0;border-radius:4px;font-size:0.82rem;">
                                        {hse_badge(cat)} <strong>{v.get('issue','')}</strong><br>
                                        <span style="opacity:0.7;">💡 {v.get('recommendation','')}</span>
                                    </div>""", unsafe_allow_html=True)

            if st.button("📋 View / Hide Full Video Report", use_container_width=True):
                st.session_state["show_video_report"] = not st.session_state.get("show_video_report", False)

            if st.session_state.get("show_video_report", False):
                score_v   = result_video.get("overall_safety_score", 0)
                verdict_v = result_video.get("overall_verdict", "NEEDS_REVIEW")
                gc_v      = "#3fb950" if score_v >= 75 else "#d29922" if score_v >= 50 else "#f85149"
                st.markdown(f"""
                <div class="score-card">
                    <div>
                        <div class="score-label">Overall Score</div>
                        <div style="font-size:2.8rem;font-weight:800;color:{gc_v};line-height:1;">{score_v}<span style="font-size:1.2rem;opacity:0.6;">/100</span></div>
                    </div>
                    <div>
                        <div class="score-label">Verdict</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{gc_v};margin-top:4px;">{verdict_v}</div>
                        <div class="score-summary">{result_video.get('summary','')}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                compliant_v = result_video.get("compliant_items", [])
                if compliant_v:
                    st.markdown('<div class="report-section-title">✅ Compliant Items</div>', unsafe_allow_html=True)
                    for item in compliant_v:
                        st.markdown(f'<div class="compliant-item">✅ {item}</div>', unsafe_allow_html=True)

                if violations_v:
                    st.markdown('<div class="report-section-title">⚠️ Violations Detected</div>', unsafe_allow_html=True)
                    for v in violations_v:
                        sev     = v.get("severity","MEDIUM").upper()
                        css_cls = {"HIGH":"violation-card-high","MEDIUM":"violation-card-medium","LOW":"violation-card-low"}.get(sev,"violation-card-medium")
                        cat     = v.get("hse_category","B")
                        st.markdown(f"""
                        <div class="{css_cls}">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                                {hse_badge(cat)}
                                <span class="violation-badge">{sev}</span>
                            </div>
                            <div class="bocw-ref">{v.get('bocw_reference','')}</div>
                            <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px;">{v.get('issue','')}</div>
                            <div style="font-size:0.82rem;opacity:0.75;">💡 {v.get('recommendation','')}</div>
                        </div>""", unsafe_allow_html=True)

                if per_image_v:
                    st.markdown('<div class="report-section-title">🎞 Frame-by-Frame Analysis</div>', unsafe_allow_html=True)
                    for fi in per_image_v:
                        fv  = fi.get("verdict","WARNING").upper()
                        fvc = verdict_color(fv)
                        ts  = fi.get("timestamp","")
                        checks = fi.get("active_checks", {})
                        with st.expander(f"{verdict_icon(fv)} Frame {fi.get('image_index','')} {f'@ {ts}' if ts else ''} — {fv}"):
                            st.markdown(f"**Verdict:** <span style='color:{fvc};font-weight:700;'>{fv}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Notes:** {fi.get('notes','')}")
                            items_f = fi.get("items_found", [])
                            if items_f:
                                st.markdown("**Items Found:**")
                                for item in items_f:
                                    st.markdown(f"- {item}")
                render_video_visualizations(result_video)
                            # if checks:
                            #     check_labels = {
                            #         "ppe_compliant":          "PPE Compliant",
                            #         "fall_protection_ok":     "Fall Protection",
                            #         "hot_work_fire_safety_ok":"Hot Work Fire Safety",
                            #         "lifting_safety_ok":      "Lifting Safety",
                            #         "housekeeping_ok":        "Housekeeping",
                            #     }
                            #     cols_chk = st.columns(len(checks))
                            #     for ci, (key, label) in enumerate(check_labels.items()):
                            #         if key in checks:
                            #             ok  = checks[key]
                            #             ico = "✅" if ok else "❌"
                            #             clr = "#3fb950" if ok else "#f85149"
                            #             with cols_chk[ci]:
                            #                 st.markdown(f"<div style='text-align:center;font-size:1.2rem;'>{ico}</div><div style='text-align:center;font-size:0.7rem;color:{clr};'>{label}</div>", unsafe_allow_html=True)
