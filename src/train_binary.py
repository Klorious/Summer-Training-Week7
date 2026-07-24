import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from dataset import RoadImageDataset
from models import get_binary_model

# ==========================================
# 0. 測試開關與路徑設定
# ==========================================
SMOKE_TEST = False  # 正式訓練時請改為 False
CONFIG_PATH = "configs/resnet18.yaml"
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV = "data/splits/val.csv"
IMG_DIR = "data/raw"

def main():
    # ==========================================
    # 1. 讀取設定檔與 W&B 初始化
    # ==========================================
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    # 動態產生符合規定的 run 名稱 (例: vgg16_pretrained_lr0.0001_bs32)
    run_name = f"{config['model_name']}_pretrained_lr{config['learning_rate']}_bs{config['batch_size']}"
    if SMOKE_TEST:
        run_name = "SMOKE_TEST_" + run_name

    wandb.init(
        project="training-Unit7-Binary",
        name=run_name,
        config=config
    )
    
    # 設置裝置與 Random Seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['random_seed'])
    
    # ==========================================
    # 2. 資料前處理與 DataLoader
    # ==========================================
    # 訓練集可使用隨機增強，驗證集必須固定
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(config['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RoadImageDataset(TRAIN_CSV, IMG_DIR, transform=train_transform)
    val_dataset = RoadImageDataset(VAL_CSV, IMG_DIR, transform=val_transform)
    
    if SMOKE_TEST:
        # Smoke Test: 只取前 16 張圖測試管線
        train_dataset = Subset(train_dataset, range(16))
        val_dataset = Subset(val_dataset, range(16))
        config['epochs'] = 2
        print("⚠️ 進入 Smoke Test 模式：僅使用 16 張圖片進行 2 個 Epoch 測試。")
        
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # ==========================================
    # 3. 載入模型、Loss 與 Optimizer
    # ==========================================
    model = get_binary_model(model_name=config['model_name'], pretrained=config['pretrained'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # ==========================================
    # 4. 訓練迴圈
    # ==========================================
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # ==========================================
        # 5. 驗證階段與指標計算
        # ==========================================
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # 計算基本任務要求的指標 (以 Good=1 為正類)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch [{epoch}/{config['epochs']}] "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {val_f1:.4f}")
        
        # W&B 動態紀錄
        wandb.log({
            "train/loss": epoch_train_loss,
            "train/accuracy": epoch_train_acc,
            "val/loss": epoch_val_loss,
            "val/accuracy": epoch_val_acc,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "val/f1": val_f1,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })
        
        # 儲存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"outputs/checkpoints/best_{config['model_name']}.pth")
            
    # ==========================================
    # 6. 結束與最佳結果紀錄
    # ==========================================
    # 紀錄 Confusion Matrix
    wandb.log({
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=["Bad", "Good"]
        )
    })
    
    wandb.finish()
    print("🎉 訓練結束！最佳模型已儲存。")

if __name__ == "__main__":
    main()
