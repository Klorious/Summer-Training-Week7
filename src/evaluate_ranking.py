import os
import pandas as pd
from sklearn.metrics import average_precision_score

# --- 設定區 ---
SCORES_CSV = "outputs/ranking/all_model_scores.csv"
REFERENCE_CSV = "data/annotations/ranking_reference_top100.csv"
OUTPUT_DIR = "outputs/ranking"

# 8 種計分方式的欄位名稱
SCORE_COLUMNS = [
    "vgg16_raw_score", "vgg16_adjusted_score",
    "resnet18_raw_score", "resnet18_adjusted_score",
    "resnet50_raw_score", "resnet50_adjusted_score",
    "ensemble_raw_score", "ensemble_adjusted_score"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("載入分數總表與人工參考集...")
    df_scores = pd.read_csv(SCORES_CSV)
    df_ref = pd.read_csv(REFERENCE_CSV)
    
    # 建立 Ground Truth：若圖片在人工 Top-100 中，則標記為 1，否則為 0
    ref_set = set(df_ref['image_id'].astype(str))
    df_scores['is_human_top100'] = df_scores['image_id'].astype(str).apply(lambda x: 1 if x in ref_set else 0)
    
    metrics_records = []

    for col in SCORE_COLUMNS:
        print(f"正在結算 [{col}]...")
        # 1. 依分數由高至低排序
        df_sorted = df_scores.sort_values(by=col, ascending=False).reset_index(drop=True)
        
        # 2. 取出前 100 名並存檔
        df_top100 = df_sorted.head(100)
        top100_csv_path = f"{OUTPUT_DIR}/{col}_top100.csv"
        df_top100[['image_id', 'image_path', col]].to_csv(top100_csv_path, index=False)
        
        # 3. 計算進階指標
        model_top100_set = set(df_top100['image_id'].astype(str))
        
        # 共同選中數量 (Intersection)
        intersection = len(ref_set.intersection(model_top100_set))
        
        # 重疊率 (Overlap) = 交集 / 100
        overlap = intersection / 100.0
        
        # Jaccard 指標 = 交集 / (聯集) = 交集 / (200 - 交集)
        jaccard = intersection / (200 - intersection)
        
        # 平均精度 (AP) = 針對 500 張圖的連續分數與 Ground Truth 計算
        ap = average_precision_score(df_scores['is_human_top100'], df_scores[col])
        
        metrics_records.append({
            "ranking_method": col,
            "intersection": intersection,
            "overlap": f"{overlap:.4f}",
            "jaccard": f"{jaccard:.4f}",
            "ap": f"{ap:.4f}"
        })

    # 4. 輸出評估報告表
    df_metrics = pd.DataFrame(metrics_records)
    metrics_csv_path = f"{OUTPUT_DIR}/ranking_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    
    print("\n✅ 所有排序與指標計算完成！")
    print(f"最終指標已儲存至: {metrics_csv_path}")
    print("\n--- 預覽評估結果 ---")
    print(df_metrics.to_markdown(index=False))

if __name__ == "__main__":
    main()