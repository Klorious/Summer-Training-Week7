import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from models import get_binary_model

# ==========================================
# 1. 設定檔案路徑
# ==========================================
REFERENCE_CSV = "data/annotations/binary_reference_100.csv"
IMG_DIR = "data/raw"
OUTPUT_CSV = "outputs/predictions/reference_results.csv"

# 確保輸出資料夾存在
os.makedirs("outputs/predictions", exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 讀取人工標註的 100 張參考集
    df = pd.read_csv(REFERENCE_CSV)
    
    # 驗證集必須使用固定前處理
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定義要測試的模型與對應的最佳權重路徑
    models_to_test = {
        "VGG16": ("vgg16", "outputs/checkpoints/best_vgg16.pth"),
        "ResNet18": ("resnet18", "outputs/checkpoints/best_resnet18.pth"),
        "ResNet50": ("resnet50", "outputs/checkpoints/best_resnet50.pth")
    }

    all_results = []

    for display_name, (model_name, weight_path) in models_to_test.items():
        print(f"\n載入模型: {display_name} ...")
        
        # 建立模型並載入我們辛苦訓練出來的最佳權重
        model = get_binary_model(model_name=model_name, pretrained=False)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()

        match_count = 0

        # 開始逐張圖片推論
        with torch.no_grad():
            for idx, row in df.iterrows():
                # 組合圖片路徑並前處理
                img_path = os.path.join(IMG_DIR, row['image_path'].split('/')[-1])
                image = Image.open(img_path).convert('RGB')
                input_tensor = val_transform(image).unsqueeze(0).to(device)

                # 模型預測
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                score_bad = probs[0]
                score_good = probs[1]
                pred_class = 1 if score_good > score_bad else 0
                pred_label_str = "Good" if pred_class == 1 else "Bad"
                
                # 比對人工標籤 (確保 CSV 中有 human_label 欄位)
                human_label_str = row['human_label']
                is_match = 1 if human_label_str == pred_label_str else 0
                match_count += is_match

                # 記錄結果 (依照作業規定格式)
                all_results.append({
                    "image_id": row['image_id'],
                    "human_label": human_label_str,
                    "model": display_name,
                    "predicted_label": pred_label_str,
                    "score_good": round(score_good, 4),
                    "score_bad": round(score_bad, 4),
                    "match": is_match
                })
        
        print(f"[{display_name}] 在 100 張人工參考集中的一致數量 (Matches): {match_count}/100")

    # 將結果輸出成 CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 所有推論完成！詳細結果已存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
