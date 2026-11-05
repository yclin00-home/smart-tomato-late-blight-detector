import os, json, time, random, shutil, sys
import torch, torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 0. Parameter Configuration
# ==========================================
SRC_DIR = "./data_in_1010"
OUT_DIR = "./data_out_v2"
TEST_DIR = "./real_test_auto_v2"
TOMATO_CLASSES = ["Tomato_Late_blight", "Tomato_healthy", "Tomato_others"]
NEG_NAME = "Non_tomato"
NEG_TOTAL = 2000                       
REAL_FOLDER = "real"
REAL_REPEAT = 8                
REAL_NEG_PER_FOLDER = 120              
REAL_NEG_SPECIAL = {"background": 300, "potato": 200}
HOLDOUT = {"default": 10, "Non_tomato": 30}
EXT = (".jpg", ".jpeg", ".png")

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
NUM_EPOCHS = 25                        
BATCH_SIZE = 32
MIN_LEAF_FRAC = 0.05                   

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Training Device: {device}")
sys.stdout.flush()

# ==========================================
# 1. OpenCV Strict Leaf Cropping
# ==========================================
def extract_and_save_leaf_crops(src_img_path, dst_crop_dir, prefix):
    try:
        img_bgr = cv2.imread(src_img_path)
        if img_bgr is None: return 0
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return 0

    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 15, 15]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(green, brown)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    saved_count = 0
    os.makedirs(dst_crop_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(src_img_path))[0]

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= MIN_LEAF_FRAC * h * w:
            crop = img_rgb[y:y+bh, x:x+bw]
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(dst_crop_dir, f"{prefix}_leaf_{i}_{base_name}.jpg")
            cv2.imwrite(save_path, crop_bgr)
            saved_count += 1
    return saved_count

# ==========================================
# 2. Dataset Processing
# ==========================================
def process_dataset():
    print("="*70)
    print("📂 Processing dataset and extracting leaf crops...")
    print("="*70)
    sys.stdout.flush()
    random.seed(42)
    if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)

    for c in TOMATO_CLASSES:
        cdir = os.path.join(SRC_DIR, c)
        if not os.path.isdir(cdir): continue
        pv_imgs = []
        for dp, dn, fs in os.walk(cdir):
            if os.path.basename(dp).lower() == REAL_FOLDER: continue
            for f in fs:
                if f.lower().endswith(EXT): pv_imgs.append(os.path.join(dp, f))
        random.shuffle(pv_imgs)
        cut = int(len(pv_imgs) * 0.8)
        tr_files, va_files = pv_imgs[:cut], pv_imgs[cut:]
        
        for p in tr_files: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{c}", "pv")
        for p in va_files: extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{c}", "pv")

        r_dir = os.path.join(cdir, REAL_FOLDER)
        if os.path.isdir(r_dir):
            real_imgs = [os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.lower().endswith(EXT)]
            random.shuffle(real_imgs)
            n_hold = min(HOLDOUT.get(c, HOLDOUT["default"]), len(real_imgs) // 4)
            hold, rest = real_imgs[:n_hold], real_imgs[n_hold:]
            for p in hold: extract_and_save_leaf_crops(p, f"{TEST_DIR}/{c}", "test")
            cut_r = int(len(rest) * 0.8)
            r_tr, r_va = rest[:cut_r], rest[cut_r:]
            for p in r_tr: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{c}", "real_base")
            for r_idx in range(REAL_REPEAT):
                for p in r_tr: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{c}", f"real{r_idx}")
            for p in r_va: extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{c}", "real")

    neg_dir = os.path.join(SRC_DIR, NEG_NAME)
    if os.path.isdir(neg_dir):
        subs = [d for d in sorted(os.listdir(neg_dir)) if os.path.isdir(os.path.join(neg_dir, d)) and d.lower() != REAL_FOLDER]
        per_source = NEG_TOTAL // max(len(subs), 1)
        for s in subs:
            s_path = os.path.join(neg_dir, s)
            imgs = [os.path.join(s_path, f) for f in os.listdir(s_path) if f.lower().endswith(EXT)]
            random.shuffle(imgs)
            selected = imgs[:per_source]
            cut_s = int(len(selected) * 0.8)
            for p in selected[:cut_s]: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{NEG_NAME}", s)
            for p in selected[cut_s:]: extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{NEG_NAME}", s)

        real_neg_dir = os.path.join(neg_dir, REAL_FOLDER)
        if os.path.isdir(real_neg_dir):
            r_neg_imgs = []
            for d in sorted(os.listdir(real_neg_dir)):
                sub_p = os.path.join(real_neg_dir, d)
                if os.path.isdir(sub_p):
                    sub_imgs = [os.path.join(sub_p, f) for f in os.listdir(sub_p) if f.lower().endswith(EXT)]
                    random.shuffle(sub_imgs)
                    cap = REAL_NEG_SPECIAL.get(d.lower(), REAL_NEG_PER_FOLDER)
                    r_neg_imgs += sub_imgs[:cap]
            random.shuffle(r_neg_imgs)
            n_hold = min(HOLDOUT["Non_tomato"], len(r_neg_imgs) // 4)
            hold, rest = r_neg_imgs[:n_hold], r_neg_imgs[n_hold:]
            for p in hold: extract_and_save_leaf_crops(p, f"{TEST_DIR}/{NEG_NAME}", "test")
            cut_rn = int(len(rest) * 0.8)
            rn_tr, rn_va = rest[:cut_rn], rest[cut_rn:]
            for p in rn_tr: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{NEG_NAME}", "real_base")
            for r_idx in range(REAL_REPEAT):
                for p in rn_tr: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{NEG_NAME}", f"real{r_idx}")
            for p in rn_va: extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{NEG_NAME}", "real")

    print("✅ Dataset mapping completed!\n")
    sys.stdout.flush()

process_dataset()

GROUPS_CONFIG = {
    "./data_m1_v2": {"Tomato": ["Tomato_Late_blight", "Tomato_healthy", "Tomato_others"], "Non_tomato": ["Non_tomato"]},
    "./data_m2_v2": {"Late_blight": ["Tomato_Late_blight"], "Not_late_blight": ["Tomato_healthy", "Tomato_others"]}
}

for out_path, groups in GROUPS_CONFIG.items():
    if os.path.exists(out_path): shutil.rmtree(out_path)
    for split_name in ("train", "val"):
        for new_cls, old_classes in groups.items():
            dst = os.path.join(out_path, split_name, new_cls)
            os.makedirs(dst, exist_ok=True)
            for old in old_classes:
                src_node = os.path.join(OUT_DIR, split_name, old)
                if os.path.exists(src_node):
                    for f in os.listdir(src_node):
                        shutil.copy(os.path.join(src_node, f), os.path.join(dst, f"{old}_{f}"))

# ==========================================
# 3. Model Training Function (Top-3 Ensemble)
# ==========================================
def train_model(data_dir, model_prefix, class_file):
    print(f"\n🏋️ Training model: {model_prefix} (Path: {data_dir})")
    sys.stdout.flush()
    
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), norm,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)
    class_names = train_ds.classes
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(class_names)).float()
    weights = (counts.sum() / (len(class_names) * counts)).to(device)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    top_models = []
    for ep in range(NUM_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        ok = n = 0
        per_ok = [0] * len(class_names)
        per_n = [0] * len(class_names)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                ok += (pred == y).sum().item(); n += y.size(0)
                for t, p in zip(y.tolist(), pred.tolist()):
                    per_n[t] += 1; per_ok[t] += int(t == p)
                    
        score = sum(o / t for o, t in zip(per_ok, per_n)) / len(per_n)
        print(f"🎉 Epoch [{ep+1}/{NUM_EPOCHS}] Val Accuracy: {ok/n*100:.2f}% | Per-class Avg: {score*100:.2f}%")
        sys.stdout.flush()

        import copy
        top_models.append((score, ep + 1, copy.deepcopy(model.state_dict())))
        top_models.sort(key=lambda x: x[0], reverse=True)
        if len(top_models) > 3: top_models.pop()

    json.dump(class_names, open(class_file, "w", encoding="utf-8"), ensure_ascii=False)
    for rank, (sc, ep_num, state_dict) in enumerate(top_models, 1):
        ckpt_path = f"{model_prefix}_top{rank}_ep{ep_num}.pth"
        torch.save(state_dict, ckpt_path)
        print(f"💾 Saved [Top-{rank}] Epoch {ep_num} (Score: {sc*100:.2f}%) -> {ckpt_path}")

    print(f"🏆 [{model_prefix}] Training Completed!\n")
    sys.stdout.flush()

train_model("./data_m1_v2", "m1_tomato_v2", "m1_classes_v2.json")
train_model("./data_m2_v2", "m2_late_blight_v2", "m2_classes_v2.json")
