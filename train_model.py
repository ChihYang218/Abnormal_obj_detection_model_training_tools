from ultralytics import YOLO
from ultralytics.utils.loss import FocalLoss

import torch

def custom_loss(preds, targets):
    # preds['cls'] 是分類 logits
    return focal(preds['cls'], targets['cls'])

def train_yolo_model(model):
    """
    執行 YOLO11 (或 v8/v9) 模型的訓練。
    """
    print("🤖 開始載入模型...")

    print("🔥 開始訓練模型...")
    
    # 執行訓練
    results = model.train(
        data='configs/data_conf.yaml',  # 指向您的資料集設定檔
        epochs=1500,                  # 訓練週期 (建議 100-300)
        imgsz=640,                   # 影像大小 (例如 640 或 1280)
        batch=20,                    # 批次大小 (根據您的 GPU VRAM 調整)
        device=0,
        name='v5',       # 訓練結果將存放在 'runs/detect/*'
        lr0=0.01,
        lrf=0.01,
        save_period=10,
        patience=1500, # 保證不 earlystopping
    )
    
    print("✅ 訓練完成！")
    print(f"📈 訓練結果存放在: {results.save_dir}")

    # (可選) 訓練完成後，自動使用驗證集進行驗證
    print("📊 開始驗證模型 (使用 val set)...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")


if __name__ == '__main__':
    # 確保在虛擬環境中執行
    print("--- YOLO 訓練腳本啟動 ---")
    model = YOLO('./models/v5.yaml') 
    
    alpha = torch.tensor([0.17, 0.62, 0.21], dtype=torch.float32)
    gamma = 2.0
    
    for m in model.model.modules():
        if m.__class__.__name__ == "Detect":
            m.cls_loss = FocalLoss(gamma=gamma, alpha=alpha)
            train_yolo_model(model)
    # train_yolo_model()
