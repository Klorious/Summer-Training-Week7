import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class RoadImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: train.csv 或 val.csv 的路徑
        root_dir: data/raw/ 的路徑
        transform: PyTorch 的資料前處理 (torchvision.transforms)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # 作業二元標籤規則已固定：Bad -> 0, Good -> 1
        self.label_map = {"Bad": 0, "Good": 1}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 組合圖片的完整路徑 (對應 data/raw/<檔名>)
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'].split('/')[-1])
        
        # 讀取圖片並確保轉換為 RGB 色彩空間
        image = Image.open(img_name).convert('RGB')
        
        # 讀取字串標籤並轉換為數字 0 或 1
        str_label = self.data_frame.iloc[idx]['human_label']
        label = self.label_map[str_label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class QualityRegressionDataset(Dataset):
    """Dataset for the advanced image-quality regression task.

    The CSV must contain ``image_path`` and ``quality_score`` columns.
    It intentionally does not read the binary label or the locked human
    Top-100 membership, so those fields cannot leak into model training.
    """

    REQUIRED_COLUMNS = {"image_path", "quality_score"}

    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        missing_columns = self.REQUIRED_COLUMNS.difference(self.data_frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_file} is missing required columns: {missing}")

        scores = pd.to_numeric(self.data_frame["quality_score"], errors="coerce")
        if scores.isna().any():
            bad_rows = scores[scores.isna()].index.tolist()
            raise ValueError(
                f"{csv_file} contains invalid quality_score values at rows: {bad_rows[:10]}"
            )
        self.data_frame["quality_score"] = scores.astype("float32")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        image_name = os.path.basename(str(row["image_path"]).replace("\\", "/"))
        image_path = os.path.join(self.root_dir, image_name)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.transform:
            image = self.transform(image)

        score = torch.tensor(float(row["quality_score"]), dtype=torch.float32)
        return image, score
