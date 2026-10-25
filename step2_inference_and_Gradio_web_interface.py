import os, json
import torch, torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import gradio as gr

# ==========================================
# Parameters & Model Inference Configuration
# ==========================================
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
MAX_SIDE = 1024         # Internal processing max size constraint
MAX_INPUT_SIDE = 800    # 🚨 [OOM Protection] Initial max side limit for uploaded images to prevent out-of-memory crashes
MIN_CONF = 0.60         # Confidence threshold for tomato classification stage 1
MAX_LEAVES = 10         # Maximum number of leaf regions to inspect per image
MIN_LEAF_FRAC = 0.015   # Minimum area ratio for detected leaves

# Inference device setup (automatically uses GPU if available, falls back to CPU on Render)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Inference Device: {device}")

def load_model(weights, classes_json):
    """
    Loads model weights and class labels, initializes the architecture, and sets it to evaluation mode.
    """
    if not os.path.exists(weights) or not os.path.exists(classes_json):
        raise FileNotFoundError(f"Model weights {weights} or classes file {classes_json} not found! Please ensure they are uploaded.")
    with open(classes_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(names))
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m, names

# Load trained dual models and class labels
model1, names1 = load_model("m1_tomato.pth", "m1_classes.json")
model2, names2 = load_model("m2_late_blight.pth", "m2_classes.json")

# Image preprocessing pipeline for inference
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(model, names, x):
    """
    Performs forward pass without gradient tracking and returns class probability dictionary.
    """
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=0)
    return {n: float(p[i]) for i, n in enumerate(names)}


# ==========================================
# Computer Vision & Leaf Segmentation Processing
# ==========================================
def find_leaf_regions(img_rgb, max_leaves=MAX_LEAVES, min_area_frac=MIN_LEAF_FRAC):
    """
    Detects and extracts bounding box regions of individual leaves using color thresholding and contour analysis.
    """
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
    """
    Draws numbered bounding boxes around detected leaf regions for expert view visualization.
    """
    vis = img_rgb.copy()
    t = max(2, img_rgb.shape[1] // 300)
    for k, (x0, y0, x1, y1) in enumerate(regions, 1):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), t)
        cv2.putText(vis, str(k), (x0 + 8, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return vis

def calc_infection_ratio(img_rgb):
    """
    Calculates disease severity percentage (infection ratio) by comparing lesion pixels against total leaf pixels.
    """
    img_hsv = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
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
    leaf_vis = img_rgb.copy()
    contours = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(leaf_vis, contours, -1, (0, 0, 255), max(2, w // 300))

    lesion_vis = img_rgb.copy()
    m = lesion > 0
    lesion_vis[m] = (0.4 * lesion_vis[m] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)

    return ratio, leaf_vis, lesion_vis


# ==========================================
# Web Report Styling (HTML/CSS with Upward Pointer ▲)
# ==========================================
def render_custom_style(diagnosis_status, ratio=None, msg=""):
    """
    Generates customized HTML diagnostic cards containing severity score bars and upward pointing indicators (▲).
    """
    r_val = max(0.0, min(100.0, ratio)) if ratio is not None else 0.0
    
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
            <span class="score-bracket">[</span><span class="score-value" style="font-size: 20px;">{diagnosis_status}</span><span class="score-bracket">]</span>
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
# Core Diagnostic Inference Workflow (with OOM Protection)
# ==========================================
def diagnose(image_path):
    """
    Executes end-to-end inference: loads, resizes for memory safety, detects leaf regions,
    passes crops through hierarchical models (M1 & M2), computes infection ratio, and returns visual results.
    """
    if image_path is None:
        return None, "Please upload or take a photo first.", None, None

    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        # 🚨 [OOM Protection] Downscale input image immediately to MAX_INPUT_SIDE to prevent memory crashes
        img.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))
        
        arr = np.array(img.convert("RGB"))
    except Exception:
        return None, "❌ Unable to read the image. Please try another one.", None, None

    regions = find_leaf_regions(arr)
    boxes_vis = draw_boxes(arr, regions) if regions else arr

    if not regions:
        status = "non-tomato"
        msg = "🚫 No leaf regions detected. Please upload a clear photo of a leaf."
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, None

    tomato_found = False
    results = []

    for r in regions:
        x0, y0, x1, y1 = r
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0: continue
        
        x = tf(Image.fromarray(crop)).unsqueeze(0).to(device)
        p1 = predict(model1, names1, x)
        is_tomato = p1["Tomato"] >= MIN_CONF

        if is_tomato:
            tomato_found = True
            p2 = predict(model2, names2, x)
            conf = {
                "Non_tomato": 0.0,
                "Late_blight": p1["Tomato"] * p2["Late_blight"],
                "Not_late_blight": p1["Tomato"] * p2["Not_late_blight"],
            }
            top = max(["Late_blight", "Not_late_blight"], key=lambda k: conf[k])
            results.append({"conf": conf, "top": top, "top_conf": conf[top], "crop": crop})
        else:
            results.append({
                "conf": {"Non_tomato": p1["Non_tomato"]},
                "top": "Non_tomato",
                "top_conf": p1["Non_tomato"],
                "crop": crop
            })

    if not tomato_found:
        status = "non-tomato"
        msg = "🚫 Leaves detected, but none of them are recognized as tomato leaves."
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, None

    tomato_results = [res for res in results if res["top"] in ("Late_blight", "Not_late_blight")]
    blighted = [res for res in tomato_results if res["top"] == "Late_blight"]

    if blighted:
        chosen = max(blighted, key=lambda res: res["top_conf"])
        label = "Late_blight"
    elif tomato_results:
        chosen = max(tomato_results, key=lambda res: res["top_conf"])
        label = "Not_late_blight"
    else:
        chosen = max(results, key=lambda res: res["top_conf"])
        label = "Non_tomato"

    note = f" (Checked {len(regions)} leaf region(s))"

    if label == "Non_tomato":
        status = "non-tomato"
        msg = f"🚫 Leaves detected, but classified as Non_tomato.{note}"
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, None

    ratio, _, lesion_vis = calc_infection_ratio(chosen["crop"])
    leaf_vis = calc_infection_ratio(chosen["crop"])[1]

    if label == "Not_late_blight":
        status = "tomato-without-late-blight"
        top_conf = chosen["top_conf"]
        msg = f"✅ Confirmed tomato leaf, **no late blight detected** (Confidence {top_conf*100:.1f}%). The leaf is healthy.{note}"
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, leaf_vis

    status = "tomato-with-late-blight"
    if ratio < 33.3:
        msg = f"Mild late blight. Remove infected leaves promptly and apply an organic copper fungicide.{note}"
    elif ratio < 66.6:
        msg = f"Moderate late blight. Isolate the plant, remove infected foliage, and use a systemic fungicide.{note}"
    else:
        msg = f"🚨 Severe late blight! Remove and destroy the plant immediately to prevent spore transmission.{note}"
    
    html_out = render_custom_style(status, ratio=ratio, msg=msg)
    return lesion_vis, html_out, boxes_vis, leaf_vis


# ==========================================
# Gradio Web UI Layout & CSS Configuration
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
