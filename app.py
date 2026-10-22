import json
import os

import cv2
import gradio as gr
import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
MAX_SIDE = 1024         
MIN_CONF = 0.8         
MAX_LEAVES = 10         
MIN_LEAF_FRAC = 0.02    

device = torch.device("cpu")


def load_model(weights, classes_json):
    with open(classes_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(names))
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m, names


# 載入剛剛訓練好的新模型權重與類別設定
model1, names1 = load_model("m1_tomato.pth", "m1_classes.json")      # Non_tomato / Tomato
model2, names2 = load_model("m2_late_blight.pth", "m2_classes.json")      # Late_blight / Not_late_blight

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, names, x):
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=0)
    return {n: float(p[i]) for i, n in enumerate(names)}


# ---------- Leaf Region Extraction ----------
def find_leaf_regions(img_rgb, max_leaves=MAX_LEAVES, min_area_frac=MIN_LEAF_FRAC):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([35, 25, 25]), np.array([85, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 20, 20]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(green, brown)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_area_frac * h * w:
            regions.append((int(area), int(x), int(y), int(bw), int(bh), i))

    regions.sort(reverse=True)
    out = []
    for area, x, y, bw, bh, i in regions[:max_leaves]:
        out.append((x, y, x + bw, y + bh, labels[y:y + bh, x:x + bh] > 0))
        
    return out


def make_crop(img_rgb, region):
    x0, y0, x1, y1, _ = region
    return img_rgb[y0:y1, x0:x1].copy()


def draw_boxes(img_rgb, regions):
    vis = img_rgb.copy()
    t = max(2, img_rgb.shape[1] // 300)
    for k, (x0, y0, x1, y1, _) in enumerate(regions, 1):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), t)
        cv2.putText(vis, str(k), (x0 + 8, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return vis


# ---------- Infection Ratio & Mask Generation (OpenCV) ----------
def calc_infection_ratio(img_rgb):
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
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)

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


# ---------- Core Strict Multi-Stage Analysis Pipeline ----------
def analyze(img_pil, use_crop=True):
    arr = np.array(img_pil)
    regions = find_leaf_regions(arr) if use_crop else []

    if not regions:
        return {"label": "Non_tomato", "chosen": None, "n_crops": 0, "regions": [], "arr": arr}

    results = []
    tomato_found = False

    for r in regions:
        c = make_crop(arr, r)
        x = tf(Image.fromarray(c)).unsqueeze(0)
        
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
            results.append({"conf": conf, "top": top, "top_conf": conf[top], "crop": c, "region": r})
        else:
            results.append({
                "conf": {"Non_tomato": p1["Non_tomato"]},
                "top": "Non_tomato",
                "top_conf": p1["Non_tomato"],
                "crop": c,
                "region": r
            })

    if not tomato_found:
        chosen = results[0] if results else None
        return {"label": "Non_tomato", "chosen": chosen, "n_crops": len(regions), "regions": regions, "arr": arr}

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

    return {"label": label, "chosen": chosen, "n_crops": len(regions), "regions": regions, "arr": arr}


# ---------- Render Custom Style Report Card (HTML) ----------
def render_custom_style(diagnosis_status, ratio=None, msg=""):
    r_val = max(0.0, min(100.0, ratio)) if ratio is not None else 0.0
    
    slider_html = f"""
        <div class="score-header" style="margin-top: 15px; font-size: 18px;">
            <span class="score-title">SEVERITY SCORE:</span> 
            <span class="score-bracket">[</span><span class="score-value">{r_val:.1f}%</span><span class="score-bracket">]</span>
        </div>
        <div class="slider-container">
            <div class="color-bar">
                <div class="bar-low"></div>
                <div class="bar-med"></div>
                <div class="bar-high"></div>
            </div>
            <div class="pointer" style="left: {r_val}%;">▼</div>
            <div class="slider-labels-grid">
                <span class="label-low">Low (0-33%)</span>
                <span class="label-med">Medium (33-66%)</span>
                <span class="label-high">High (66-100%)</span>
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
            <div class="treatment-title">RECOMMENDED ACTION:</div>
            <div class="treatment-text">{msg}</div>
        </div>
    </div>
    """
    return html_content


# ---------- Diagnosis Function ----------
def diagnose(image_path, use_crop):
    if image_path is None:
        return None, "Please upload or take a photo first.", None, None, None

    try:
        img = Image.open(image_path)
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img = img.convert("RGB")
    except Exception:
        return None, "❌ Unable to read the image. Please try another one.", None, None, None

    arr = np.array(img)
    r = analyze(img, use_crop)
    regions = r["regions"]
    boxes_vis = draw_boxes(arr, regions) if regions else arr

    if not regions:
        status = "non-tomato"
        msg = "🚫 No leaf regions detected. Please upload a clear photo of a leaf."
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, None, {}

    label = r["label"]
    chosen = r["chosen"]

    # 即使是 Non_tomato，只要有偵測到葉片，也正常產生 Leaf Mask
    leaf_vis = calc_infection_ratio(chosen["crop"])[1] if chosen is not None else None

    if label == "Non_tomato":
        status = "non-tomato"
        msg = "🚫 Leaves detected, but none of them are recognized as tomato leaves."
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, leaf_vis, {}

    conf = chosen["conf"]
    top_conf = chosen["top_conf"]
    note = f" (Checked {r['n_crops']} leaf region(s))"

    if label == "Not_late_blight":
        status = "tomato-without-late-blight"
        msg = f"✅ Confirmed tomato leaf, **no late blight detected** (Confidence {top_conf*100:.1f}%). The leaf is healthy. Maintain good ventilation and monitor regularly.{note}"
        html_out = render_custom_style(status, msg=msg)
        return arr, html_out, boxes_vis, leaf_vis, conf

    status = "tomato-with-late-blight"
    ratio, _, lesion_vis = calc_infection_ratio(chosen["crop"])
    if ratio < 33.3:
        msg = f"Mild late blight. Remove infected leaves promptly and apply an organic copper fungicide.{note}"
    elif ratio < 66.6:
        msg = f"Moderate late blight. Isolate the plant, remove infected foliage, and use a systemic fungicide to limit spread.{note}"
    else:
        msg = f"🚨 Severe late blight! Remove and destroy the plant immediately to prevent spore transmission.{note}"
    
    html_out = render_custom_style(status, ratio=ratio, msg=msg)
    return lesion_vis, html_out, boxes_vis, leaf_vis, conf


# ---------- Gradio Blocks UI ----------
with gr.Blocks(title="Tomato Disease Report") as demo:
    gr.Markdown("# 🍅 Tomato Leaf Disease Diagnosis")
    gr.Markdown("Take or upload a photo of a tomato leaf, then click **Diagnose** to view the professional report.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Take or Upload Photo", sources=["upload", "webcam"])
            use_crop_hidden = gr.Checkbox(value=True, visible=False)
            btn = gr.Button("🔍 Diagnose", variant="primary", elem_id="diag_btn")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Analyzed Image", interactive=False)

    report_output = gr.HTML(label="Diagnostic Report")

    with gr.Accordion("🔬 Expert View: Detailed Analysis & Masks", open=False):
        gr.Markdown("Computer vision segmentation details for experts or advanced users:")
        with gr.Row():
            boxes_output = gr.Image(label="Detected Leaf Regions (Red Boxes)", interactive=False)
            leaf_output = gr.Image(label="Leaf Mask (Red Outline)", interactive=False)

    conf_hidden = gr.Label(visible=False)

    btn.click(diagnose, inputs=[image_input, use_crop_hidden],
              outputs=[image_output, report_output, boxes_output, leaf_output, conf_hidden])

    gr.HTML("""
    <style>
        /* 強制修正 Gradio 深色模式對自訂卡片文字顏色的影響 */
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
            margin: 15px 0 45px 0 !important;
        }
        .color-bar {
            display: flex !important;
            height: 24px !important;
            border-radius: 6px !important;
            overflow: hidden !important;
            border: 1.5px solid #b5b0a5 !important;
        }
        .bar-low { background-color: #27ae60 !important; flex: 1; }
        .bar-med { background-color: #f39c12 !important; flex: 1; }
        .bar-high { background-color: #c0392b !important; flex: 1; }
        .pointer {
            position: absolute !important;
            top: 24px !important;
            font-size: 16px !important;
            color: #000000 !important;
            transform: translateX(-50%);
        }
        .slider-labels-grid {
            display: grid !important;
            grid-template-columns: 1fr 1fr 1fr !important;
            margin-top: 8px !important;
            font-size: 14px !important;
            font-weight: bold !important;
        }
        .label-low { text-align: left; }
        .label-med { text-align: center; }
        .label-high { text-align: right; }
    </style>
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
