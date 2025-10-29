from ultralytics import YOLO

def train_yolo_model():
    """
    執行 YOLO11 (或 v8/v9) 模型的訓練。
    """
    print("🤖 開始載入模型...")
    
    model = YOLO('yolo11n.pt') 

    print("🔥 開始訓練模型...")
    
    # 執行訓練
    results = model.train(
        data='configs/data_conf.yaml',  # 指向您的資料集設定檔
        epochs=100,                  # 訓練週期 (建議 100-300)
        imgsz=640,                   # 影像大小 (例如 640 或 1280)
        batch=16,                    # 批次大小 (根據您的 GPU VRAM 調整)
        device=0,                    # 指定 GPU 0 (如果有多張卡)
        name='out',       # 訓練結果將存放在 'runs/detect/out'
        patience=30                  # 提早停止 (Early stopping) 的等待週期
    )
    
    print("✅ 訓練完成！")
    print(f"📈 訓練結果存放在: {results.save_dir}")

    # (可選) 訓練完成後，自動使用驗證集進行驗證
    print("📊 開始驗證模型 (使用 val set)...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

    # (可選) 將最佳模型複製到 'model' 資料夾
    # best_model_path = results.save_dir / 'weights' / 'best.pt'
    # shutil.copy(best_model_path, 'model/best_model_yolov10n.pt')
    # print(f"🏆 最佳模型已複製到 'model/' 資料夾")


if __name__ == '__main__':
    # 確保在虛擬環境中執行
    print("--- YOLO 訓練腳本啟動 ---")
    train_yolo_model()

