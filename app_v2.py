import os, json
import torch
import timm
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import gradio as gr

# ==========================================
# 0. Inference Parameter Configuration
# ==========================================
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
MAX_INPUT_SIDE = 800
MIN_CONF_INFERENCE = 0.75  # Stage-1 strict defense threshold
RED = (255, 0, 0)
GREEN = (0, 255, 0)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Inference Device: {device}")


# ==========================================
# 1. Load Specific Best Models (Optimized for Render RAM)
# ==========================================
def load_specific_model(prefix, specific_filename, classes_json):
    """
    載入指定的單一最佳模型權重與類別標籤，
    避免在 Render 免費方案中因同時載入多個模型而導致記憶體爆炸 (OOM)。
    """
    if not os.path.exists(classes_json):
        raise FileNotFoundError(f"Cannot find class file: {classes_json}")
    
    with open(classes_json, "r", encoding="utf-8") as f:
        names = json.load(f)
        
    if not os.path.exists(specific_filename):
        raise FileNotFoundError(f"Cannot find model weights file: {specific_filename}")
        
    print(f"📥 Loading selected model: {specific_filename}")
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(names))
    m.load_state_dict(torch.load(specific_filename, map_location=device))
    m.eval()
    
    # 包成串列格式以相容後續推論邏輯
    return [m], names

# 載入您指定的 v2 雙階段最佳模型
models1, names1 = load_specific_model("m1_tomato_v2", "m1_tomato_v2_top1_ep25.pth", "m1_classes_v2.json")
models2, names2 = load_specific_model("m2_late_blight_v2", "m2_late_blight_v2_top1_ep22.pth", "m2_classes_v2.json")

inf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_single(models, names, x):
    """使用單一模型進行前向傳播預測"""
    with torch.no_grad():
        p = torch.softmax(models[0](x)[0], dim=0)
    return {n: float(p[i]) for i, n in enumerate(names)}


# ==========================================
# 2. Computer Vision & Precise Contour Extraction
# ==========================================
def find_leaf_contours_and_crops(img_rgb, max_leaves=10, min_area_frac=0.05):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 15, 15]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(green, brown)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    leaf_items = []
    
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_area_frac * h * w:
            single_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            leaf_items.append({
                "area": int(area),
                "box": (int(x), int(y), int(bw), int(bh)),
                "contours": contours
            })
            
    leaf_items.sort(key=lambda item: item["area"], reverse=True)
    return leaf_items[:max_leaves]


def draw_pure_leaf_outlines(img_rgb, analyzed_results):
    vis = img_rgb.copy()
    thickness = max(2, img_rgb.shape[1] // 300)
    
    for idx, res in enumerate(analyzed_results, 1):
        contours = res["contours"]
        x, y, bw, bh = res["box"]
        status = res["status"]
        
        if status == "tomato-with-late-blight":
            color = RED
        elif status == "tomato-without-late-blight":
            color = GREEN
        else:
            color = (128, 128, 128)
            
        cv2.drawContours(vis, contours, -1, color, thickness)
        cv2.putText(vis, f"#{idx}", (x + 8, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        
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

    lesion_vis = img_rgb.copy()
    m = lesion > 0
    lesion_vis[m] = (0.4 * lesion_vis[m] + 0.6 * np.array(RED)).astype(np.uint8)

    return ratio, lesion_vis


# ==========================================
# 3. Comprehensive Report Rendering (Vertical Line Indicator)
# ==========================================
def render_multi_report(analyzed_results):
    html_cards = ""
    for idx, res in enumerate(analyzed_results, 1):
        status = res["status"]
        ratio = res.get("ratio", 0.0)
        msg = res["msg"]
        
        if status == "tomato-with-late-blight":
            status_color = "#a93226"
            r_val = max(0.0, min(100.0, ratio))
            slider_html = f"""
                <div style="margin-top: 10px; font-size: 16px; font-weight: bold; color: #a93226;">
                    📉 Severity Score: [{r_val:.1f}%]
                </div>
                <div style="position: relative; margin-top: 10px; margin-bottom: 20px;">
                    <div style="display: flex; height: 16px; border-radius: 4px; overflow: hidden; border: 1px solid #b5b0a5;">
                        <div style="background-color: #27ae60; flex: 1;" title="Mild (<33.3%)"></div>
                        <div style="background-color: #f39c12; flex: 1;" title="Moderate (33.3% - 66.6%)"></div>
                        <div style="background-color: #c0392b; flex: 1;" title="Severe (>66.6%)"></div>
                    </div>
                    <div style="position: absolute; top: -3px; bottom: -3px; left: {r_val}%; width: 3px; background-color: #111111; transform: translateX(-50%); z-index: 5; box-shadow: 0 0 3px rgba(255,255,255,0.8);"></div>
                    <div style="position: absolute; top: 18px; left: {r_val}%; transform: translateX(-50%); font-size: 12px; font-weight: bold; color: #111111; background: rgba(255,255,255,0.9); padding: 0 4px; border: 1px solid #b5b0a5; border-radius: 3px; white-space: nowrap;">
                        {r_val:.1f}%
                    </div>
                </div>
            """
        else:
            status_color = "#1e8449" if status == "tomato-without-late-blight" else "#555555"
            slider_html = ""

        html_cards += f"""
        <div style="background-color: #f4f1ea; border: 2px solid #dcd6cd; border-radius: 10px; padding: 15px; margin-bottom: 12px; font-family: Arial;">
            <div style="font-size: 16px; font-weight: bold; color: {status_color};">
                🌿 Leaf #{idx} — DIAGNOSIS STATUS: [{status.upper()}]
            </div>
            {slider_html}
            <div style="margin-top: 8px; font-size: 15px; line-height: 1.5;">{msg}</div>
        </div>
        """
    return html_cards


# ==========================================
# 4. Core Diagnosis Main Logic
# ==========================================
def diagnose(image_path):
    if image_path is None: 
        return None, None, "Please upload an image first."
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))
        arr = np.array(img.convert("RGB"))
    except Exception:
        return None, None, "❌ Failed to read image."

    leaf_items = find_leaf_contours_and_crops(arr)
    if not leaf_items:
        empty_report = render_multi_report([{"status": "non-tomato", "msg": "🚫 <b>Non-tomato leaf</b> — No valid leaf regions detected."}])
        return arr, arr, empty_report

    analyzed_results = []
    first_blight_lesion_vis = arr  
    has_blight_found = False

    for item in leaf_items:
        x, y, bw, bh = item["box"]
        crop = arr[y:y+bh, x:x+bw]
        if crop.size == 0: continue
        
        x_tensor = inf_tf(Image.fromarray(crop)).unsqueeze(0).to(device)
        p1 = predict_single(models1, names1, x_tensor)
        tomato_conf = p1["Tomato"]

        if tomato_conf >= MIN_CONF_INFERENCE:
            p2 = predict_single(models2, names2, x_tensor)
            lb_conf = p2["Late_blight"]
            is_blight = lb_conf >= p2["Not_late_blight"]
            
            if is_blight:
                ratio, lesion_vis = calc_infection_ratio(crop)
                if ratio < 33.3:
                    advice = "Advice: Mild late blight. Recommend removing infected leaves and applying copper fungicide."
                elif ratio < 66.6:
                    advice = "Advice: Moderate late blight. Isolate plant and use systemic fungicide."
                else:
                    advice = "Advice: 🚨 Severe late blight! Strongly recommend immediate removal and destruction of the plant to prevent spread."
                
                msg = f"🍅 <b>Tomato Leaf Confidence</b>: {tomato_conf*100:.1f}%<br>⚠️ <b>Late Blight Confidence</b>: {lb_conf*100:.1f}%<br>{advice}"
                analyzed_results.append({
                    "box": (x, y, bw, bh),
                    "contours": item["contours"],
                    "status": "tomato-with-late-blight",
                    "ratio": ratio,
                    "msg": msg
                })
                
                if not has_blight_found:
                    full_lesion_canvas = arr.copy()
                    full_lesion_canvas[y:y+bh, x:x+bw] = lesion_vis
                    first_blight_lesion_vis = full_lesion_canvas
                    has_blight_found = True
            else:
                msg = f"🍅 <b>Tomato Leaf Confidence</b>: {tomato_conf*100:.1f}%<br>✅ <b>No Late Blight Confidence</b>: {p2['Not_late_blight']*100:.1f}%"
                analyzed_results.append({
                    "box": (x, y, bw, bh),
                    "contours": item["contours"],
                    "status": "tomato-without-late-blight",
                    "msg": msg
                })
        else:
            msg = f"🚫 <b>Non-tomato / Background Noise Confidence</b>: {p1['Non_tomato']*100:.1f}%<br>This region is not identified as a tomato leaf."
            analyzed_results.append({
                "box": (x, y, bw, bh),
                "contours": item["contours"],
                "status": "non-tomato",
                "msg": msg
            })

    pure_outlines_vis = draw_pure_leaf_outlines(arr, analyzed_results)
    lesion_output_vis = first_blight_lesion_vis if has_blight_found else arr
    html_report = render_multi_report(analyzed_results)

    return pure_outlines_vis, lesion_output_vis, html_report


# ==========================================
# 5. Launch Gradio Web Interface (Render Port Support)
# ==========================================
port = int(os.environ.get("PORT", 7860))

with gr.Blocks(title="Tomato Disease Report v2 (English)") as demo:
    gr.Markdown("# 🍅 Tomato Late Blight Intelligent Diagnosis System")
    gr.Markdown("Upload an image to automatically outline true leaf contours, display lesion highlights, and review detailed confidence scores.")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Test Image")
        
    btn = gr.Button("🔍 Start Precision Diagnosis", variant="primary")
    
    with gr.Row():
        outlines_output = gr.Image(label="1. True Leaf Edge Outlines")
        lesion_output = gr.Image(label="2. Lesion Highlight (Late Blight Only)")
        
    report_output = gr.HTML(label="Detailed Diagnostic Report & Confidence Scores")

    btn.click(diagnose, inputs=[image_input],
              outputs=[outlines_output, lesion_output, report_output])

    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
