import os, json
import torch
import timm
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import base64
import uvicorn
import io

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
    return [m], names

models1, names1 = load_specific_model("m1_tomato_v2", "m1_tomato_v2_top1_ep25.pth", "m1_classes_v2.json")
models2, names2 = load_specific_model("m2_late_blight_v2", "m2_late_blight_v2_top1_ep22.pth", "m2_classes_v2.json")

inf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_single(models, names, x):
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
    return ratio


# ==========================================
# 3. Core Processing Logic (API Helper)
# ==========================================
def process_image_array(arr):
    leaf_items = find_leaf_contours_and_crops(arr)
    if not leaf_items:
        return [{"status": "non-tomato", "msg": "No valid leaf regions detected."}]

    analyzed_results = []
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
                ratio = calc_infection_ratio(crop)
                if ratio < 33.3: advice = "Mild late blight."
                elif ratio < 66.6: advice = "Moderate late blight."
                else: advice = "Severe late blight!"
                
                analyzed_results.append({
                    "status": "tomato-with-late-blight",
                    "tomato_confidence": round(tomato_conf * 100, 1),
                    "late_blight_confidence": round(lb_conf * 100, 1),
                    "severity_ratio": round(ratio, 1),
                    "advice": advice
                })
            else:
                analyzed_results.append({
                    "status": "tomato-without-late-blight",
                    "tomato_confidence": round(tomato_conf * 100, 1),
                    "healthy_confidence": round(p2['Not_late_blight'] * 100, 1),
                    "advice": "Healthy tomato leaf."
                })
        else:
            analyzed_results.append({
                "status": "non-tomato",
                "non_tomato_confidence": round(p1['Non_tomato'] * 100, 1),
                "advice": "Not a tomato leaf."
            })
            
    return analyzed_results


# ==========================================
# 4. Gradio UI Setup
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
        empty_report = '<div class="custom-report-card"><b>Non-tomato leaf</b></div>'
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
                img_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                mask_green = cv2.inRange(img_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
                mask_diseased = cv2.inRange(img_hsv, np.array([10, 30, 30]), np.array([35, 255, 255]))
                leaf_mask = cv2.bitwise_or(mask_green, mask_diseased)
                mask_dark = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
                lesion = cv2.bitwise_or(mask_diseased, mask_dark)
                lesion = cv2.bitwise_and(lesion, lesion, mask=leaf_mask)
                ratio = calc_infection_ratio(crop)
                
                lesion_vis = crop.copy()
                m = lesion > 0
                lesion_vis[m] = (0.4 * lesion_vis[m] + 0.6 * np.array(RED)).astype(np.uint8)

                if ratio < 33.3: advice = "Advice: Mild late blight. Recommend removing infected leaves and applying copper fungicide."
                elif ratio < 66.6: advice = "Moderate late blight. Isolate plant and use systemic fungicide."
                else: advice = "Severe late blight! Strongly recommend immediate removal and destruction of the plant."
                
                msg = f"🍅 <b>Tomato Leaf Confidence</b>: {tomato_conf*100:.1f}%<br>⚠️ <b>Late Blight Confidence</b>: {lb_conf*100:.1f}%<br>📉 <b>Severity Score</b>: [{ratio:.1f}%]<br>{advice}"
                analyzed_results.append({"box": (x, y, bw, bh), "contours": item["contours"], "status": "tomato-with-late-blight", "ratio": ratio, "msg": msg})
                
                if not has_blight_found:
                    full_lesion_canvas = arr.copy()
                    full_lesion_canvas[y:y+bh, x:x+bw] = lesion_vis
                    first_blight_lesion_vis = full_lesion_canvas
                    has_blight_found = True
            else:
                msg = f"🍅 <b>Tomato Leaf Confidence</b>: {tomato_conf*100:.1f}%<br>✅ <b>Healthy Confidence</b>: {p2['Not_late_blight']*100:.1f}%"
                analyzed_results.append({"box": (x, y, bw, bh), "contours": item["contours"], "status": "tomato-without-late-blight", "msg": msg})
        else:
            analyzed_results.append({"box": (x, y, bw, bh), "contours": item["contours"], "status": "non-tomato", "msg": "Not identified as a tomato leaf."})

    vis = arr.copy()
    thickness = max(2, arr.shape[1] // 300)
    for idx, res in enumerate(analyzed_results, 1):
        color = RED if res["status"] == "tomato-with-late-blight" else (GREEN if res["status"] == "tomato-without-late-blight" else (128, 128, 128))
        cv2.drawContours(vis, res["contours"], -1, color, thickness)
        cv2.putText(vis, f"#{idx}", (res["box"][0] + 8, res["box"][1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

    lesion_output_vis = first_blight_lesion_vis if has_blight_found else arr
    
    html_cards = ""
    for idx, res in enumerate(analyzed_results, 1):
        status_color = "#a93226" if res["status"] == "tomato-with-late-blight" else "#1e8449"
        html_cards += f"""
        <div class="custom-report-card">
            <div style="font-size: 16px; font-weight: bold; color: {status_color};">
                🌿 Leaf #{idx} — STATUS: [{res["status"].upper()}]
            </div>
            <div style="margin-top: 8px; font-size: 15px; line-height: 1.6;">
                {res["msg"]}
            </div>
        </div>
        """

    return vis, lesion_output_vis, html_cards

with gr.Blocks(title="Tomato Disease Report v2") as demo:
    gr.Markdown("# 🍅 Tomato Late Blight Intelligent Diagnosis System")
    gr.Markdown("Upload an image to automatically outline true leaf contours and review detailed confidence scores.")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Test Image")
        
    btn = gr.Button("🔍 Start Precision Diagnosis", variant="primary")
    
    with gr.Row():
        outlines_output = gr.Image(label="1. True Leaf Edge Outlines")
        lesion_output = gr.Image(label="2. Lesion Highlight")
        
    report_output = gr.HTML(label="Diagnostic Report")

    btn.click(diagnose, inputs=[image_input], outputs=[outlines_output, lesion_output, report_output])

    gr.HTML("""
    <style>
        .custom-report-card {
            background-color: #f4f1ea !important;
            border: 2px solid #dcd6cd !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin-bottom: 12px !important;
            font-family: Arial, sans-serif !important;
        }
        .custom-report-card, .custom-report-card *, .custom-report-card span, .custom-report-card b {
            color: #2c2c2c !important;
        }
    </style>
    """)


# ==========================================
# 5. FastAPI Integration (終極萬用接收端點：同時支援 JSON 與檔案上傳)
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict")
async def api_predict(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        image_bytes = None
        
        # 情況 A：接收 JSON (Base64) 格式
        if "application/json" in content_type:
            body = await request.json()
            image_data = body.get("image") or body.get("file") or body.get("data")
            if image_data:
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                
        # 情況 B：接收 Multipart 檔案上傳格式
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file") or form.get("image")
            if file:
                image_bytes = await file.read()
                
        # 情況 C：保險預備（萬一 headers 沒對到，直接嘗試解析 body）
        if not image_bytes:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    data = json.loads(body_bytes)
                    image_data = data.get("image") or data.get("file")
                    if image_data:
                        if "," in image_data: image_data = image_data.split(",")[1]
                        image_bytes = base64.b64decode(image_data)
                except:
                    image_bytes = body_bytes

        if not image_bytes:
            return {"success": False, "error": "No valid image data found in request."}
        
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))
        arr = np.array(img.convert("RGB"))
        
        results = process_image_array(arr)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 將 Gradio 介面掛載到根目錄
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
