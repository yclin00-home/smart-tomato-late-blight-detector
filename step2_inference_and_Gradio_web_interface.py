import os, json
import torch, torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import gradio as gr

# ==========================================
# Parameters and model settings
# ==========================================
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
MAX_SIDE = 1024         # Max size for internal tile processing
MAX_INPUT_SIDE = 800    # Resize uploaded images right away to avoid out-of-memory (OOM) errors
MIN_CONF = 0.60         # Minimum "Tomato" confidence to treat a region as a tomato leaf
MAX_LEAVES = 10         # Max number of leaf regions to check
MIN_LEAF_FRAC = 0.05    # Min region area (fraction of the image) to count as a leaf

# Render usually runs on CPU; switch to GPU automatically if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Inference device: {device}")


def load_model(weights, classes_json):
    if not os.path.exists(weights) or not os.path.exists(classes_json):
        raise FileNotFoundError(
            f"Cannot find model file {weights} or class file {classes_json}. Please make sure they are uploaded!"
        )
    with open(classes_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(names))
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m, names


# Load the two trained models and their class labels
model1, names1 = load_model("m1_tomato.pth", "m1_classes.json")
model2, names2 = load_model("m2_late_blight.pth", "m2_classes.json")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, names, x):
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=0)
    return {n: float(p[i]) for i, n in enumerate(names)}


# ==========================================
# Computer vision and leaf segmentation
# ==========================================
# Images are RGB arrays, so pure red is (255, 0, 0)
RED = (255, 0, 0)


def find_leaf_regions(img_rgb, max_leaves=MAX_LEAVES, min_area_frac=MIN_LEAF_FRAC):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 15, 15]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(green, brown)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_area_frac * h * w:
            regions.append((int(area), int(x), int(y), int(bw), int(bh)))

    regions.sort(reverse=True)
    out = []
    for area, x, y, bw, bh in regions[:max_leaves]:
        out.append((x, y, x + bw, y + bh))
    return out


def draw_boxes(img_rgb, regions):
    vis = img_rgb.copy()
    t = max(2, img_rgb.shape[1] // 300)
    for k, (x0, y0, x1, y1) in enumerate(regions, 1):
        cv2.rectangle(vis, (x0, y0), (x1, y1), RED, t)
        cv2.putText(vis, str(k), (x0 + 8, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, RED, 3, cv2.LINE_AA)
    return vis


def calc_infection_ratio(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(img_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_diseased = cv2.inRange(img_hsv, np.array([10, 30, 30]), np.array([35, 255, 255]))
    leaf_mask = cv2.bitwise_or(mask_green, mask_diseased)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    mask_dark = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    lesion = cv2.bitwise_or(mask_diseased, mask_dark)
    lesion = cv2.bitwise_and(lesion, lesion, mask=leaf_mask)

    leaf_px = cv2.countNonZero(leaf_mask)
    ratio = 0.0 if leaf_px == 0 else cv2.countNonZero(lesion) / leaf_px * 100.0

    h, w = img_rgb.shape[:2]

    # Leaf outline (red)
    leaf_vis = img_rgb.copy()
    contours = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(leaf_vis, contours, -1, RED, max(2, w // 300))

    # Lesion overlay (red tint)
    lesion_vis = img_rgb.copy()
    m = lesion > 0
    lesion_vis[m] = (0.4 * lesion_vis[m] + 0.6 * np.array(RED)).astype(np.uint8)

    return ratio, leaf_vis, lesion_vis


# ==========================================
# Report layout (HTML/CSS)
# ==========================================
# Status colors: only late blight is red; everything else is neutral
STATUS_COLORS = {
    "non-tomato": "#555555",
    "tomato-without-late-blight": "#1e8449",
    "tomato-with-late-blight": "#a93226",
}


def render_custom_style(diagnosis_status, ratio=None, msg=""):
    r_val = max(0.0, min(100.0, ratio)) if ratio is not None else 0.0
    status_color = STATUS_COLORS.get(diagnosis_status, "#2c2c2c")

    slider_html = f"""
        <div class="score-header" style="margin-top: 15px; font-size: 18px;">
            <span class="score-title">SEVERITY SCORE:</span>
            <span class="score-bracket">[</span><span class="score-value">{r_val:.1f}%</span><span class="score-bracket">]</span>
        </div>
        <div class="slider-container">
            <div class="slider-labels-grid">
                <span class="label-low">Low (0-33%)</span>
                <span class="label-med">Medium (33-66%)</span>
                <span class="label-high">High (66-100%)</span>
            </div>
            <div class="pointer-up" style="left: {r_val}%;">▲</div>
            <div class="color-bar">
                <div class="bar-low"></div>
                <div class="bar-med"></div>
                <div class="bar-high"></div>
            </div>
        </div>
    """ if diagnosis_status == "tomato-with-late-blight" else ""

    html_content = f"""
    <div class="custom-report-card">
        <div class="score-header">
            <span class="score-title">DIAGNOSIS STATUS:</span>
            <span class="score-bracket" style="color: {status_color} !important;">[</span><span class="score-value" style="font-size: 20px; color: {status_color} !important;">{diagnosis_status}</span><span class="score-bracket" style="color: {status_color} !important;">]</span>
        </div>
        {slider_html}
        <div class="treatment-section">
            <div class="treatment-title" style="font-weight:bold; margin-bottom:5px;">RECOMMENDED ACTION:</div>
            <div class="treatment-text">{msg}</div>
        </div>
    </div>
    """
    return html_content


# ==========================================
# Core diagnosis function
# ==========================================
def conf_line(label_html, conf):
    return f"{label_html} (Confidence {conf*100:.1f}%)"


def diagnose(image_path):
    if image_path is None:
        return None, "Please upload or take a photo first.", None, None

    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        # OOM protection: shrink the image to at most MAX_INPUT_SIDE right away
        img.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))

        arr = np.array(img.convert("RGB"))
    except Exception:
        return None, "❌ Unable to read the image. Please try another one.", None, None

    regions = find_leaf_regions(arr)
    boxes_vis = draw_boxes(arr, regions) if regions else arr

    # 1) No leaf region found -> not a tomato (models were not run, so no confidence)
    if not regions:
        status = "non-tomato"
        msg = "🚫 <b>Not a tomato leaf</b> — no leaf region was detected in the image."
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, None

    tomato_found = False
    results = []

    for r in regions:
        x0, y0, x1, y1 = r
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        x = tf(Image.fromarray(crop)).unsqueeze(0).to(device)
        p1 = predict(model1, names1, x)
        tomato_conf = p1["Tomato"]

        if tomato_conf >= MIN_CONF:
            tomato_found = True
            p2 = predict(model2, names2, x)
            lb_conf = p2["Late_blight"]
            nlb_conf = p2["Not_late_blight"]
            if lb_conf >= nlb_conf:
                top, stage2_conf = "Late_blight", lb_conf
            else:
                top, stage2_conf = "Not_late_blight", nlb_conf
            results.append({
                "top": top,
                "tomato_conf": tomato_conf,      # Stage 1: confidence it is a tomato leaf
                "stage2_conf": stage2_conf,      # Stage 2: confidence of late blight / no late blight
                "score": tomato_conf * stage2_conf,
                "crop": crop,
            })
        else:
            results.append({
                "top": "Non_tomato",
                "tomato_conf": tomato_conf,
                "stage2_conf": None,
                "non_tomato_conf": p1["Non_tomato"],  # Stage 1: confidence it is NOT a tomato leaf
                "score": p1["Non_tomato"],
                "crop": crop,
            })

    note = f" (Checked {len(regions)} leaf region(s))"

    # 2) No region is a tomato leaf -> not a tomato, with confidence
    if not tomato_found:
        chosen = max(results, key=lambda res: res["score"])
        status = "non-tomato"
        msg = f"🚫 {conf_line('<b>Not a tomato leaf</b>', chosen['non_tomato_conf'])}.{note}"
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, None

    tomato_results = [res for res in results if res["top"] in ("Late_blight", "Not_late_blight")]
    blighted = [res for res in tomato_results if res["top"] == "Late_blight"]

    if blighted:
        chosen = max(blighted, key=lambda res: res["score"])
        label = "Late_blight"
    else:
        chosen = max(tomato_results, key=lambda res: res["score"])
        label = "Not_late_blight"

    tomato_line = conf_line("🍅 <b>Tomato leaf</b>", chosen["tomato_conf"])

    # Leaf outline + lesion overlay for the chosen leaf (used for both tomato outcomes)
    ratio, leaf_vis, lesion_vis = calc_infection_ratio(chosen["crop"])

    # 3) Tomato, but no late blight: tomato confidence + "no late blight" confidence
    if label == "Not_late_blight":
        status = "tomato-without-late-blight"
        lb_line = conf_line("<b>No late blight detected</b>", chosen["stage2_conf"])
        msg = f"{tomato_line}<br>{lb_line}.{note}"
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, leaf_vis

    # 4) Tomato with late blight: both confidences + severity and advice
    status = "tomato-with-late-blight"
    lb_line = conf_line("⚠️ <b>Late blight detected</b>", chosen["stage2_conf"])

    if ratio < 33.3:
        advice = "Mild late blight. Remove infected leaves promptly and apply an organic copper fungicide."
    elif ratio < 66.6:
        advice = "Moderate late blight. Isolate the plant, remove infected foliage, and use a systemic fungicide."
    else:
        advice = "🚨 Severe late blight! Remove and destroy the plant immediately to prevent spore transmission."

    msg = f"{tomato_line}<br>{lb_line}<br>{advice}{note}"
    html_out = render_custom_style(status, ratio=ratio, msg=msg)
    return lesion_vis, html_out, boxes_vis, leaf_vis


# ==========================================
# Gradio UI
# ==========================================
with gr.Blocks(title="Tomato Disease Report") as demo:
    gr.Markdown("# 🍅 Tomato Leaf Disease Diagnosis")
    gr.Markdown("Designed by Bo Ruei Lin")
    gr.Markdown("Take or upload a photo of a tomato leaf, then click **Diagnose** to view the professional report.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Take or Upload Photo", sources=["upload", "webcam"])
            btn = gr.Button("🔍 Diagnose", variant="primary", elem_id="diag_btn")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Analyzed Image", interactive=False)

    report_output = gr.HTML(label="Diagnostic Report")

    with gr.Accordion("🔬 Expert View: Detailed Analysis & Masks", open=False):
        gr.Markdown("Computer vision segmentation details for experts or advanced users:")
        with gr.Row():
            boxes_output = gr.Image(label="Detected Leaf Regions (Red Boxes)", interactive=False)
            leaf_output = gr.Image(label="Leaf Mask (Red Outline)", interactive=False)

    btn.click(diagnose, inputs=[image_input],
              outputs=[image_output, report_output, boxes_output, leaf_output])

    gr.HTML("""
    <style>
        .custom-report-card, .custom-report-card * {
            color: #2c2c2c !important;
        }
        .custom-report-card {
            background-color: #f4f1ea !important;
            border: 2px solid #dcd6cd !important;
            border-radius: 12px !important;
            padding: 20px !important;
            font-family: Arial, sans-serif !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
            margin-top: 15px !important;
        }
        .score-header {
            font-size: 18px !important;
            font-weight: bold !important;
            margin-bottom: 8px !important;
        }
        .score-bracket {
            color: #a93226 !important;
            font-weight: bold !important;
        }
        .score-value {
            color: #a93226 !important;
            font-weight: bold !important;
            font-size: 22px !important;
        }
        .slider-container {
            position: relative !important;
            margin: 25px 0 15px 0 !important;
        }
        .slider-labels-grid {
            display: grid !important;
            grid-template-columns: 1fr 1fr 1fr !important;
            margin-bottom: 5px !important;
            font-size: 14px !important;
            font-weight: bold !important;
        }
        .label-low { text-align: left; }
        .label-med { text-align: center; }
        .label-high { text-align: right; }
        .pointer-up {
            position: absolute !important;
            top: 18px !important;
            font-size: 16px !important;
            color: #000000 !important;
            transform: translateX(-50%);
        }
        .color-bar {
            display: flex !important;
            height: 24px !important;
            border-radius: 6px !important;
            overflow: hidden !important;
            border: 1.5px solid #b5b0a5 !important;
            margin-top: 5px !important;
        }
        .bar-low { background-color: #27ae60 !important; flex: 1; }
        .bar-med { background-color: #f39c12 !important; flex: 1; }
        .bar-high { background-color: #c0392b !important; flex: 1; }
    </style>
    """)

if __name__ == "__main__":
    # Bind to 0.0.0.0 and port for Render cloud deployment support
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
