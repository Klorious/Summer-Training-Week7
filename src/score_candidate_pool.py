import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings
from src.models import get_quality_model

warnings.filterwarnings("ignore")

# --- 設定區 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOL_CSV = "data/splits/candidate_pool.csv"
OUTPUT_DIR = "outputs/ranking"
MODELS = ["vgg16", "resnet18", "resnet50"]
TTA_RUNS = 5  # TTA 測試次數
PENALTY_WEIGHT = 0.5  # 信心修正的懲罰係數

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 資料前處理 ---
# 1. 基礎前處理 (計算 Raw Score 用)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. TTA 前處理 (計算 Adjusted Score 用，加入隨機性)
tta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CandidateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        return row['image_id'], img_path, self.transform(image), image

def get_z_scores(scores):
    """計算 Z-score 標準化分數"""
    scores = np.array(scores)
    return (scores - np.mean(scores)) / (np.std(scores) + 1e-8)

def main():
    print(f"使用裝置: {DEVICE}")
    df_pool = pd.read_csv(POOL_CSV)
    print(f"載入候選池圖片數量: {len(df_pool)}")
    
    # 建立一個存放所有分數的 DataFrame
    results_df = df_pool[['image_id', 'image_path']].copy()
    
    for model_name in MODELS:
        print(f"\n[{model_name}] 開始載入模型與權重...")
        model = get_quality_model(model_name, pretrained=False)
        weight_path = f"outputs/checkpoints/ranking_best_{model_name}.pth"
        
        # 讀取 Checkpoint 大禮包
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        # 拆開包裝：如果裡面有 model_state_dict，就只取這個部分
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint) # 防呆機制，兼容只存權重的版本
            
        model.to(DEVICE)

        model.eval() # 評估模式
        
        raw_scores = []
        adjusted_scores = []
        
        # 為了 TTA，需要讀取原始圖片
        dataset = CandidateDataset(POOL_CSV, transform=base_transform)
        
        for idx in tqdm(range(len(dataset)), desc=f"{model_name} 推論中"):
            image_id, img_path, base_tensor, raw_pil_img = dataset[idx]
            
            # 1. 計算 Raw Score (無隨機性)
            with torch.no_grad():
                base_tensor = base_tensor.unsqueeze(0).to(DEVICE)
                raw_score = model(base_tensor).item()
                raw_scores.append(raw_score)
                
            # 2. 計算 TTA 分數 (信心修正)
            tta_preds = []
            model.train() # 開啟 Dropout/BatchNorm 隨機性進行 TTA
            for _ in range(TTA_RUNS):
                tta_tensor = tta_transform(raw_pil_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    tta_preds.append(model(tta_tensor).item())
            
            tta_mean = np.mean(tta_preds)
            tta_std = np.std(tta_preds)
            adjusted_score = tta_mean - (PENALTY_WEIGHT * tta_std)
            adjusted_scores.append(adjusted_score)
            model.eval() # 恢復評估模式
            
        # 將該模型的分數存入對應欄位
        results_df[f'{model_name}_raw_score'] = raw_scores
        results_df[f'{model_name}_adjusted_score'] = adjusted_scores
        
        # 順便輸出單一模型的 CSV
        model_df = df_pool[['image_id', 'image_path']].copy()
        model_df['raw_score'] = raw_scores
        model_df['adjusted_score'] = adjusted_scores
        model_df.to_csv(f"{OUTPUT_DIR}/{model_name}_scores.csv", index=False)
        
    print("\n[Ensemble] 計算三模型集成分數 (Z-score 標準化)...")
    # 對三個模型的 Raw Score 進行 Z-score 標準化後平均
    z_vgg_raw = get_z_scores(results_df['vgg16_raw_score'])
    z_res18_raw = get_z_scores(results_df['resnet18_raw_score'])
    z_res50_raw = get_z_scores(results_df['resnet50_raw_score'])
    results_df['ensemble_raw_score'] = (z_vgg_raw + z_res18_raw + z_res50_raw) / 3.0
    
    # 對三個模型的 Adjusted Score 進行 Z-score 標準化後平均
    z_vgg_adj = get_z_scores(results_df['vgg16_adjusted_score'])
    z_res18_adj = get_z_scores(results_df['resnet18_adjusted_score'])
    z_res50_adj = get_z_scores(results_df['resnet50_adjusted_score'])
    results_df['ensemble_adjusted_score'] = (z_vgg_adj + z_res18_adj + z_res50_adj) / 3.0

    # 輸出最終總表
    final_csv_path = f"{OUTPUT_DIR}/all_model_scores.csv"
    results_df.to_csv(final_csv_path, index=False)
    print(f"\n✅ 推論完成！總表已儲存至: {final_csv_path}")
    print(f"總筆數: {len(results_df)} (應為 500)")

if __name__ == "__main__":
    main()