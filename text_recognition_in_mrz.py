from paddleocr import PaddleOCR
from pathlib import Path
from ultralytics import YOLO

import cv2
import json
import numpy 
import os


ocr = PaddleOCR(
    use_doc_orientation_classify = False,
    use_doc_unwarping = False,
    use_textline_orientation = False,
    lang = 'en', 
    ocr_version = 'PP-OCRv5',
    enable_mkldnn = False
)

model = YOLO('mrz.pt')

script_dir = Path(__file__).parent
images = script_dir / 'images'

crop_dir = script_dir / 'crops'
crop_dir.mkdir(exist_ok = True)

mrz_list = []
res_list = []

def get_list_with_lines_from_mrz(
    crop_dir: str | Path, 
    images: str | Path, 
    model: YOLO, 
    mrz_list: list, 
    ocr: PaddleOCR
    ) -> list[list[str]]:
    """Возвращает список списков, содержащих две строки из машиночитаемой зоны паспорта.
    
    Args:
        crop_dir: путь к папке, в которой будут сохраняться обрезанные фото с областью mrz.
        images: путь к папке с исходными фотографиями.
        model: модель YOLO, которая находит нужную область.
        mrz_list: пустой список, где будет храниться результат.
        ocr: объект PaddleOCR, извлекает текст.
    
    Returns: 
        mrz_list: список списков, который содержат две строки из машиночитаемой зоны паспорта.
    """
    
    for img in sorted(images.iterdir()):
        image = cv2.imread(str(img))
        results = model(image)
        #results[0].show()
        obb_boxes = results[0].obb

        for i, obb in enumerate(obb_boxes):
            corners = obb.xyxyxyxy.cpu().numpy()[0]
            
            x_min = int(corners[:, 0].min())
            y_min = int(corners[:, 1].min())
            x_max = int(corners[:, 0].max())
            y_max = int(corners[:, 1].max())
            
            mrz_crop = image[y_min:y_max, x_min:x_max]
            
            saving_path = crop_dir / f'{img.stem}_{i}.jpg'
            cv2.imwrite(str(saving_path), mrz_crop)

            ocr_result = ocr.ocr(mrz_crop)
            
            texts = ocr_result[0]['rec_texts']
            l1 = texts[0]
            l2 = texts[1]
            
            mrz_list.append([l1, l2])
            
    return mrz_list

def mrz_parsing(l1: str, l2: str) -> dict[str, str]:
    """Разложение по полям стандарта MRZ.
    
    Args:
        l1: первая строка машиночитаемой зоны документа.
        l2: вторая строка машиночитаемой зоны докуммента.
        
    Returns:
        data: словарь с разложением по полям."""
    
    data = {}
    
    data['document_type'] = l1[0]
    data['country'] = l1[2:5]

    names = l1[5:].split('<<')

    data['surname'] = names[0].replace('<', ' ').strip()

    name_parts = []
    for n in names[1:]:
        if n.strip():
            name_parts.append(n.replace('<', '').strip())

    data['name'] = ' '.join(name_parts)
    data['passport_number'] = l2[0:9].replace('<', '')
    data['nationality'] = l2[10:13]
    data['birth_date'] = l2[13:15] + '-' + l2[15:17] + '-' + l2[17:19]
    data['gender'] = l2[20]
    data['expiry_date'] = l2[21:23] + '-' + l2[23:25] + '-' + l2[25:27]
    data['personal_number'] = l2[28:42].replace('<', '')

    return data

mrz_list = get_list_with_lines_from_mrz(crop_dir, images, model, mrz_list, ocr)

for i, j in mrz_list:
    res = mrz_parsing(i, j)
    res_list.append(res)
    
with open('result_of_rec.json', 'w', encoding='utf-8') as f:
    json.dump(res_list, f, indent = 4)
