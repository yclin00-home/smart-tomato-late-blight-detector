import os, json, time, random, shutil, sys
import torch, torch.nn as nn
import timm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
from PIL import Image, ImageOps
import gradio as gr

# ==========================================
# 0. Configuration & Hyperparameters Setup
# ==========================================
SRC_DIR = "./data_in_1010"          # Root directory containing raw input datasets
OUT_DIR = "./data_out_1010"        # Directory for processed and cropped leaf image datasets
TEST_DIR = "./real_test_auto"      # Directory reserved for holdout real test images
TOMATO_CLASSES = ["Tomato_Late_blight", "Tomato_healthy", "Tomato_others"]
NEG_NAME = "Non_tomato"            # Negative background/non-tomato class name
NEG_TOTAL = 1000                   # Target sample count for negative samples
REAL_FOLDER = "real"               # Subfolder name for real-world photos
REAL_REPEAT = 8                    # Data augmentation repetition factor for real images
REAL_MAX_TRAIN = 800
REAL_NEG_PER_FOLDER = 60
REAL_NEG_SPECIAL = {"background": 150, "potato": 100}
HOLDOUT = {"default": 10, "Non_tomato": 30} # Number of test samples held out per class
EXT = (".jpg", ".jpeg", ".png")

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k" # Lightweight backbone architecture
NUM_EPOCHS = 12                    # Total training epochs per model
BATCH_SIZE = 32                    # Batch size for DataLoader
MIN_LEAF_FRAC = 0.015              # Minimum area ratio threshold for leaf contour filtering (1.5% of image area)

# Automatically detect available hardware acceleration device (CUDA -> MPS -> CPU)
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Execution Device: {device}")
sys.stdout.flush()

# ==========================================
# 1. Core Function: OpenCV Strict Leaf Segmentation & Cropping
# ==========================================
def extract_and_save_leaf_crops(src_img_path, dst_crop_dir, prefix):
    """
    Reads an image, converts it to HSV color space, extracts green and brown plant tissue masks,
    applies morphological operations to clean noise, finds connected components (individual leaves),
    and saves valid leaf crops that exceed the minimum area fraction threshold.
    """
    try:
        img_bgr = cv2.imread(src_img_path)
        if img_bgr is None: return 0
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return 0

    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Define HSV color ranges for healthy green foliage and brown/diseased tissue
    green = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 15, 15]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(green, brown)
    
    # Morphological filtering (Close then Open) to remove background noise and fill holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # Connected component analysis to isolate individual leaf blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    saved_count = 0

    os.makedirs(dst_crop_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(src_img_path))[0]

    # Filter and save individual leaf crops based on area threshold
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
# 2. Automated Dataset Cleaning & Split Statistics Reporting
# ==========================================
def process_dataset():
    """
    Scans source directories, processes PlantVillage and real-world photos through leaf cropping,
    splits them into train/validation sets (8:2 ratio), and holds out a separate test set.
    """
    print("="*70)
    print("📂 Starting folder scanning, leaf segmentation, and dataset split statistics...")
    print("="*70)
    sys.stdout.flush()
    
    random.seed(42)
    if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)

    # Process 3 tomato disease classes
    for c in TOMATO_CLASSES:
        cdir = os.path.join(SRC_DIR, c)
        if not os.path.isdir(cdir):
            print(f"⚠️ Directory not found: {cdir}")
            continue
        
        # 1. Process PlantVillage dataset images (excluding real subfolders)
        pv_imgs = []
        for dp, dn, fs in os.walk(cdir):
            if os.path.basename(dp).lower() == REAL_FOLDER: continue
            for f in fs:
                if f.lower().endswith(EXT): pv_imgs.append(os.path.join(dp, f))
        
        random.shuffle(pv_imgs)
        cut = int(len(pv_imgs) * 0.8)
        tr_files, va_files = pv_imgs[:cut], pv_imgs[cut:]
        
        pv_tr_count = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{c}", "pv") for p in tr_files)
        pv_va_count = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{c}", "pv") for p in va_files)
        
        print(f"\n🍅 Class: 【 {c} 】")
        print(f"   🔹 Source 1: PlantVillage Dataset (Path: {cdir})")
        print(f"      - Original images found: {len(pv_imgs)}")
        print(f"      - Split allocation: 8:2 ratio -> Train: {pv_tr_count} crops | Val: {pv_va_count} crops")

        # 2. Process real-world photos
        r_dir = os.path.join(cdir, REAL_FOLDER)
        if os.path.isdir(r_dir):
            real_imgs = [os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.lower().endswith(EXT)]
            random.shuffle(real_imgs)
            n_hold = min(HOLDOUT.get(c, HOLDOUT["default"]), len(real_imgs) // 4)
            hold, rest = real_imgs[:n_hold], real_imgs[n_hold:]
            
            test_count = sum(extract_and_save_leaf_crops(p, f"{TEST_DIR}/{c}", "test") for p in hold)
            cut_r = int(len(rest) * 0.8)
            r_tr, r_va = rest[:cut_r], rest[cut_r:]
            
            r_tr_count_base = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{c}", f"real_base") for p in r_tr)
            for r_idx in range(REAL_REPEAT):
                for p in r_tr: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{c}", f"real{r_idx}")
            r_tr_total = r_tr_count_base * REAL_REPEAT
            r_va_count = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{c}", "real") for p in r_va)
            
            print(f"   🔹 Source 2: Real Photos (Path: {r_dir})")
            print(f"      - Original real photos found: {len(real_imgs)}")
            print(f"      - Split allocation:")
            print(f"        1. Holdout Test Set (excluded from training): Extracted {test_count} leaf crops from {len(hold)} images (Saved in {TEST_DIR}/{c})")
            print(f"        2. Training Set (with x{REAL_REPEAT} augmentation): Base crops {r_tr_count_base} -> Total augmented crops {r_tr_total}")
            print(f"        3. Validation Set: {r_va_count} crops")
        sys.stdout.flush()

    # Process Non_tomato negative class
    neg_dir = os.path.join(SRC_DIR, NEG_NAME)
    if os.path.isdir(neg_dir):
        print(f"\n🌿 Class: 【 {NEG_NAME} (Non-tomato Background) 】")
        subs = [d for d in sorted(os.listdir(neg_dir)) if os.path.isdir(os.path.join(neg_dir, d)) and d.lower() != REAL_FOLDER]
        per_source = NEG_TOTAL // max(len(subs), 1)
        
        for s in subs:
            s_path = os.path.join(neg_dir, s)
            imgs = [os.path.join(s_path, f) for f in os.listdir(s_path) if f.lower().endswith(EXT)]
            random.shuffle(imgs)
            selected = imgs[:per_source]
            cut_s = int(len(selected) * 0.8)
            tr_s = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{NEG_NAME}", s) for p in selected[:cut_s])
            va_s = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{NEG_NAME}", s) for p in selected[cut_s:])
            print(f"   🔹 Negative Subsource [{s}] (Path: {s_path})")
            print(f"      - Sampled {len(selected)} images -> Train: {tr_s} crops | Val: {va_s} crops")

        real_neg_dir = os.path.join(neg_dir, REAL_FOLDER)
        if os.path.isdir(real_neg_dir):
            r_neg_imgs = [os.path.join(real_neg_dir, f) for f in os.listdir(real_neg_dir) if f.lower().endswith(EXT)]
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
            test_neg_count = sum(extract_and_save_leaf_crops(p, f"{TEST_DIR}/{NEG_NAME}", "test") for p in hold)
            cut_rn = int(len(rest) * 0.8)
            rn_tr, rn_va = rest[:cut_rn], rest[cut_rn:]
            
            rn_tr_base = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{NEG_NAME}", "real_base") for p in rn_tr)
            for r_idx in range(REAL_REPEAT):
                for p in rn_tr: extract_and_save_leaf_crops(p, f"{OUT_DIR}/train/{NEG_NAME}", f"real{r_idx}")
            rn_tr_total = rn_tr_base * REAL_REPEAT
            rn_va_count = sum(extract_and_save_leaf_crops(p, f"{OUT_DIR}/val/{NEG_NAME}", "real") for p in rn_va)
            
            print(f"   🔹 Real Negative Photos (Path: {real_neg_dir})")
            print(f"      - Holdout Test Set: {test_neg_count} crops (Saved in {TEST_DIR}/{NEG_NAME})")
            print(f"      - Training Set (with x{REAL_REPEAT} augmentation): Base {rn_tr_base} -> Total {rn_tr_total} crops")
            print(f"      - Validation Set: {rn_va_count} crops")
        sys.stdout.flush()

    print("\n" + "="*70)
    print("📊 Final Output Statistics of Leaf Crop Counts per Class:")
    print("="*70)
    for c in TOMATO_CLASSES + [NEG_NAME]:
        tr_path = f"{OUT_DIR}/train/{c}"
        va_path = f"{OUT_DIR}/val/{c}"
        n_tr = len(os.listdir(tr_path)) if os.path.exists(tr_path) else 0
        n_va = len(os.listdir(va_path)) if os.path.exists(va_path) else 0
        print(f"   - {c:22s} | Train Crops: {n_tr:5d} | Val Crops: {n_va:4d}")
    print("="*70 + "\n")
    sys.stdout.flush()

process_dataset()

# Setup two-stage hierarchical model mapping directories (M1 and M2)
GROUPS_CONFIG = {
    "./data_m1": {"Tomato": ["Tomato_Late_blight", "Tomato_healthy", "Tomato_others"], "Non_tomato": ["Non_tomato"]},
    "./data_m2": {"Late_blight": ["Tomato_Late_blight"], "Not_late_blight": ["Tomato_healthy", "Tomato_others"]}
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
print("✅ Two-stage model mapping directories successfully created!\n")
sys.stdout.flush()


# ==========================================
# 3. Model Training Function (with real-time progress, Cosine Annealing, and Curve Plotting)
# ==========================================
def train_model(data_dir, model_file, class_file):
    """
    Builds data loaders with data augmentation, computes class weights to handle imbalance,
    initializes the timm model, optimizes with AdamW and CosineAnnealingLR, records history,
    and saves the best-performing checkpoint and convergence curves.
    """
    print(f"\n🏋️ Starting training model: {model_file} (Data Path: {data_dir})")
    sys.stdout.flush()
    
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Training transforms with random augmentations to improve generalization
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), norm,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)
    class_names = train_ds.classes
    print(f"   Class list: {class_names}")
    print(f"   Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")
    sys.stdout.flush()
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Compute inverse class frequency weights for CrossEntropyLoss to address class imbalance
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(class_names)).float()
    weights = (counts.sum() / (len(class_names) * counts)).to(device)

    # Initialize model using timm, criterion, AdamW optimizer, and Cosine Annealing scheduler
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_bal": []}
    best_score = 0.0
    best_epoch = 0

    for ep in range(NUM_EPOCHS):
        model.train()
        run_loss = 0.0
        run_ok = 0
        seen = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item() * y.size(0)
            run_ok += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
            
            # Print batch loss periodically
            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                print(f"   [Epoch {ep+1}/{NUM_EPOCHS}] Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                sys.stdout.flush()
                
        scheduler.step()

        # Validation phase
        model.eval()
        ok = n = 0
        val_loss_total = 0.0
        per_ok = [0] * len(class_names)
        per_n = [0] * len(class_names)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss_total += criterion(out, y).item() * y.size(0)
                pred = out.argmax(1)
                ok += (pred == y).sum().item(); n += y.size(0)
                for t, p in zip(y.tolist(), pred.tolist()):
                    per_n[t] += 1; per_ok[t] += int(t == p)
                    
        train_loss_epoch = run_loss / seen
        val_loss_epoch = val_loss_total / n
        train_acc_epoch = run_ok / seen
        val_acc_epoch = ok / n
        score = sum(o / t for o, t in zip(per_ok, per_n)) / len(per_n)

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["train_acc"].append(train_acc_epoch)
        history["val_acc"].append(val_acc_epoch)
        history["val_bal"].append(score)

        print(f"🎉 Completed Epoch [{ep+1}/{NUM_EPOCHS}] | Val Total Accuracy: {val_acc_epoch*100:.2f}% | Per-Class Average Accuracy: {score*100:.2f}%")
        sys.stdout.flush()

        # Save checkpoint if per-class average accuracy improves
        if score > best_score:
            best_score = score
            best_epoch = ep + 1
            torch.save(model.state_dict(), model_file)
            json.dump(class_names, open(class_file, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"    💾 Better model found! Saved to {model_file} (Score: {score*100:.2f}%)")
            sys.stdout.flush()

    print(f"🏆 [{model_file}] Training finished! Best Per-Class Average Accuracy: {best_score*100:.2f}% (Epoch {best_epoch})\n")
    sys.stdout.flush()

    # Plot and save training convergence curves
    epochs_range = list(range(1, NUM_EPOCHS + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs_range, history["train_loss"], "o-", label="train loss")
    ax1.plot(epochs_range, history["val_loss"], "o-", label="validation loss")
    ax1.set_title(f"{model_file} - Loss (lower is better)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")

    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], "o-", label="train accuracy")
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], "o-", label="validation accuracy")
    ax2.plot(epochs_range, [a * 100 for a in history["val_bal"]], "o--", label="per-class average")
    ax2.set_title(f"{model_file} - Accuracy % (higher is better)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy %")

    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="gray", linestyle=":", label=f"best epoch ({best_epoch})")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    curve_file = os.path.splitext(model_file)[0] + "_curves.png"
    plt.savefig(curve_file, dpi=150)
    plt.show()
    print(f"📈 Training convergence curves saved to: {curve_file}\n")
    sys.stdout.flush()

# Execute hierarchical dual-model training workflow
train_model("./data_m1", "m1_tomato.pth", "m1_classes.json")
train_model("./data_m2", "m2_late_blight.pth", "m2_classes.json")
print("🎉 Dual-model training and curve plotting completed successfully!\n")
sys.stdout.flush()
