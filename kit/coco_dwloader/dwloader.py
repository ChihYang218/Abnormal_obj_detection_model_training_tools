import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# --- 全域設定 ---
# 1. 如果你的 json 檔在 annotations 資料夾內，請改成 'annotations/instances_train2017.json'
#    如果你把 json 檔拿出來跟程式放在一起，就維持現狀。
JSON_DIR = '.'  # json 檔所在的資料夾路徑
TARGET_CLASSES = ['bird', 'person']

# 2. 類別 ID 映射 (確保 train 和 val 的 ID 一致)
CLASS_MAPPING = {
    'bird': 0,
    'person': 1,
}

# 3. 測試模式開關
# True = 只跑 10 張 (測試用)
# False = 跑全部 (正式用，會花很多時間)
IS_TEST_RUN = False  

# --- YOLO 座標轉換函式 ---
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)

# --- 核心處理函式 ---
def process_dataset(data_type, json_filename):
    print(f"\n{'='*20} 開始處理: {data_type} {'='*20}")
    
    ann_file = os.path.join(JSON_DIR, json_filename)
    if not os.path.exists(ann_file):
        print(f"錯誤: 找不到檔案 {ann_file}")
        return

    # 設定存檔路徑
    save_img_dir = f'./coco2017_dataset/{data_type}/images'
    save_label_dir = f'./coco2017_dataset/{data_type}/labels'
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    # 初始化 COCO
    coco = COCO(ann_file)

    # 取得類別 ID 與建立查找表
    coco_catIds = coco.getCatIds(catNms=TARGET_CLASSES)
    id_lookup = {}
    for catId in coco_catIds:
        cat_info = coco.loadCats(catId)[0]
        if cat_info['name'] in CLASS_MAPPING:
            id_lookup[catId] = CLASS_MAPPING[cat_info['name']]
    
    print(f"ID Mapping: {id_lookup}")

    # 取得圖片 ID
    img_ids_set = set()
    for catId in coco_catIds:
        img_ids_set.update(coco.getImgIds(catIds=[catId]))
    
    final_img_ids = list(img_ids_set)
    print(f"共找到 {len(final_img_ids)} 張圖片")

    # 測試模式判斷
    process_list = final_img_ids[:10] if IS_TEST_RUN else final_img_ids
    
    # 開始下載與轉換
    for img_id in tqdm(process_list, desc=f"下載 {data_type}"):
        img_info = coco.loadImgs(img_id)[0]
        img_w = img_info['width']
        img_h = img_info['height']
        filename = img_info['file_name']

        # 處理標籤
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=coco_catIds, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        yolo_lines = []
        for ann in anns:
            bbox = ann['bbox']
            # 過濾掉極小的框 (有些壞掉的資料 w或h 為 0)
            if bbox[2] < 1 or bbox[3] < 1: 
                continue
                
            yolo_box = convert_to_yolo((img_w, img_h), bbox)
            my_class_id = id_lookup.get(ann['category_id'])

            if my_class_id is not None:
                line = f"{my_class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
                yolo_lines.append(line)

        # 存檔 (只有當這張圖有我們需要的標籤時)
        if yolo_lines:
            # A. 寫入 Label
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            with open(os.path.join(save_label_dir, txt_filename), 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            # B. 下載 Image (若已存在則跳過)
            img_path = os.path.join(save_img_dir, filename)
            if not os.path.exists(img_path):
                try:
                    img_data = requests.get(img_info['coco_url'], timeout=10).content
                    with open(img_path, 'wb') as handler:
                        handler.write(img_data)
                except Exception as e:
                    print(f"下載失敗 {filename}: {e}")

# --- 主程式進入點 ---
if __name__ == '__main__':
    # 1. 處理 Train
    process_dataset('train', 'instances_train2017.json')
    
    # 2. 處理 Val
    process_dataset('valid', 'instances_val2017.json')
    
    print("\n全部處理完成！")