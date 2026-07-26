# Overleaf 報告使用方式

## 1. 上傳方式

將整個 `report/` 資料夾壓縮成 ZIP，登入 Overleaf 後選擇：

1. New Project
2. Upload Project
3. 上傳 ZIP

主檔為 `main.tex`。

## 2. 編譯器

在新版 Overleaf：

1. 點畫面左下角的齒輪 `Settings`。
2. 在彈出視窗選 `Compiler` 分頁。
3. Main document 選 `main.tex`。
4. Compiler 選 `XeLaTeX`。
5. TeX Live version 使用最新版本。
6. 關閉設定視窗並按 Recompile。

舊版介面則是左上角 `Menu` → `Settings` → `Compiler`。
專案也包含 `latexmkrc`，用來向 latexmk 指定 XeLaTeX；但仍建議在
Overleaf 專案設定中明確選擇 XeLaTeX。

報告使用 Overleaf 內建的 `Noto Serif CJK TC` 與
`Noto Sans CJK TC`，以繁體中文字形顯示正文、圖表標題與章節名稱。

## 3. W&B 圖檔

下列九張報告用圖片已放在 `figures/wandb/`：

- `binary_loss_curves.png`
- `binary_accuracy_curves.png`
- `binary_resnet18_lr_comparison.png`
- `binary_confusion_matrices.png`
- `binary_runs_table.png`
- `ranking_loss_curves.png`
- `ranking_validation_metrics.png`
- `ranking_runs_table.png`
- `ranking_final_metrics.png`

若自行替換圖片，請保留相同檔名。報告會在圖檔缺失時顯示占位框。

### Overleaf 免費方案編譯逾時

本專案中的報告圖片已縮至適合 A4 報告的解析度，案例圖片也已轉為
JPEG，以降低 XeLaTeX 的記憶體與編譯時間。若從舊專案更新檔案後仍
顯示逾時，請使用 `Recompile from scratch` 清除舊快取；最穩妥方式是
以最新 ZIP 建立新的 Overleaf project，再選擇 XeLaTeX 編譯。

## 4. 必須人工補上的內容

- 封面的姓名與學號。
- 確認所有 W&B 與 GitHub 連結可由未登入瀏覽器開啟。
- 若實際觀察與案例草稿不同，修正案例說明。

進階 validation 表的 best epoch、MAE、RMSE 與 Spearman 已由
三個正式 W&B run 的完整歷史紀錄精確填入。

## 5. 圖片與資料

`figures/cases/` 放置報告使用的 25 張案例圖片。
`tables/` 放置結果 CSV；Top-100 完整表位於 `tables/top100/`。
