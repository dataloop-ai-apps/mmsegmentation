import dtlpy as dl
from mmseg.apis import inference_model, init_model
import os
import numpy as np

class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        os.system("mim download mmsegmentation --config pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512 --dest .")
        print("mmsegmentation downloaded successfully")
        with open("./labels.txt", "r") as file:
            self.coco_labels = [line.replace('\n', '') for line in file.readlines()]

    def detect_obj(self, item):
        image_path = item.download()

        config_file = './config_dir/pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512.py'
        checkpoint_file = 'pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth'

        model = init_model(config_file, checkpoint_file, device='cpu')
        builder = item.annotations.builder()

        result = inference_model(model, image_path).pred_sem_seg.cpu().data[0]
        ids = np.unique(result)
        columns_num = len(result[0])
        for id in ids:
            mask = []
            for pred_row in result:
                row = []
                for columns_i in range(columns_num):
                    row.append(1 if int(pred_row[columns_i]) == id else 0)
                mask.append(row)
            builder.add(annotation_definition=dl.Segmentation(geo=np.array(mask), label=self.coco_labels[id]))
        item.annotations.upload(builder)
