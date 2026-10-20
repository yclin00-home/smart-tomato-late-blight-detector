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
MAX_SIDE = 1024      # 圖片最長邊超過這個值就縮小
MIN_CONF = 0.60      # 信心度低於此值 -> 顯示「不確定」

device = torch.device("cpu")


def load_model(weights, classes_json):
    with open(classes_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(names))
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m, names


model1, names1 = load_model("m1_tomato.pth", "m1_classes.json")          # Non_tomato / Tomato
model2, names2 = load_model("m2_late_blight.pth", "m2_classes.json")     # Late_blight / Not_late_blight

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, names, x):
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=0)
    return {n: float(p[i]) for i, n in enumerate(names)}


# ---------- 病斑面積比例 (OpenCV) ----------
# 回傳: (病斑比例 %, 葉子邊緣圖, 病斑圖)
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

    # 圖 1：葉子邊緣 (綠色線圈出葉子範圍)
    leaf_vis = img_rgb.copy()
    contours = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(leaf_vis, contours, -1, (0, 255, 0), max(2, w // 300))

    # 圖 2：病斑 (紅色) + 比例文字
    lesion_vis = img_rgb.copy()
    m = lesion > 0
    lesion_vis[m] = (0.4 * lesion_vis[m] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    scale = max(0.8, w / 700)
    text = f"Lesion ratio: {ratio:.1f}%"
    cv2.putText(lesion_vis, text, (10, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(lesion_vis, text, (10, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)

    return ratio, leaf_vis, lesion_vis


# ---------- 診斷 ----------
# 回傳 5 個值: 機率, 病斑比例文字, 建議, 葉子邊緣圖, 病斑圖
def diagnose(image_path):
    if image_path is None:
        return None, "N/A", "📷 Please take or upload a photo.", None, None

    try:
        img = Image.open(image_path)
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img = img.convert("RGB")
    except Exception:
        return None, "N/A", "❌ Cannot read this image. Please try another one.", None, None

    x = tf(img).unsqueeze(0)
    p1 = predict(model1, names1, x)     # 是不是番茄
    p2 = predict(model2, names2, x)     # 是不是晚疫病 (只有番茄才有意義)

    conf = {
        "Non_tomato": p1["Non_tomato"],
        "Late_blight": p1["Tomato"] * p2["Late_blight"],
        "Not_late_blight": p1["Tomato"] * p2["Not_late_blight"],
    }
    top = max(conf, key=conf.get)
    top_conf = conf[top]

    if top_conf < MIN_CONF:
        return conf, "N/A", (f"❓ Not sure (best guess: {top}, {top_conf*100:.1f}%). "
                             "Try a clearer, closer photo of a single leaf."), None, None

    if top == "Non_tomato":
        return conf, "N/A", ("🚫 **This does not look like a tomato leaf.** "
                             "Please photograph a tomato leaf."), None, None

    if top == "Not_late_blight":
        return conf, "N/A", (f"✅ Tomato leaf, **no late blight detected** "
                             f"(confidence {top_conf*100:.1f}%)."), None, None

    # Late_blight
    ratio, leaf_vis, lesion_vis = calc_infection_ratio(np.array(img))
    if ratio < 15:
        msg = (f"💡 Mild late blight (lesions cover {ratio:.1f}%). "
               "Remove infected leaves promptly and apply an organic copper fungicide.")
    elif ratio < 45:
        msg = (f"⚠️ Moderate late blight (lesions cover {ratio:.1f}%). "
               "Isolate the plant and use a systemic fungicide to limit spread.")
    else:
        msg = (f"🚨 Severe late blight (lesions cover {ratio:.1f}%)! "
               "Remove and destroy the plant to prevent spore spread.")
    return conf, f"{ratio:.1f}%", msg, leaf_vis, lesion_vis


# ---------- 網頁介面 ----------
with gr.Blocks(title="Tomato Leaf Disease Diagnosis") as demo:
    gr.Markdown("# 🍅 Tomato Leaf Disease Diagnosis")
    gr.Markdown("Take a photo of a tomato leaf, or upload one.")

    image_input = gr.Image(type="filepath", label="Take or upload a photo",
                           sources=["upload", "webcam"])
    btn = gr.Button("🔍 Diagnose", variant="primary")

    label_output = gr.Label(label="Prediction & confidence")
    ratio_output = gr.Textbox(label="Lesion area ratio (late blight only)")
    advice_output = gr.Markdown()

    with gr.Row():
        leaf_output = gr.Image(label="Leaf Mask (green outline = leaf area)", interactive=False)
        lesion_output = gr.Image(label="Lesion Mask (red = lesions)", interactive=False)

    btn.click(diagnose, inputs=[image_input],
              outputs=[label_output, ratio_output, advice_output, leaf_output, lesion_output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting app on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)
