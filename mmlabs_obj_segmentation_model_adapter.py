import os
from mmseg.apis import inference_model, init_model

import dtlpy as dl
import numpy as np
import logging

logger = logging.getLogger('MMSegmentation')


@dl.Package.decorators.module(description='Model Adapter for mmlabs object segmentation',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        config_file = 'pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512.py'
        checkpoint_file = 'pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth'

        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            os.system("mim download mmsegmentation --config pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512 --dest .")

        print("mmsegmentation downloaded successfully")
        with open("./labels.txt", "r") as file:
            self.coco_labels = [line.replace('\n', '') for line in file.readlines()]

        self.model = init_model(config_file, checkpoint_file, device='cpu')

    def predict(self, batch, **kwargs):
        batch_annotations = list()
        for image in batch:
            image_annotations = dl.AnnotationCollection()
            result = inference_model(self.model, image).pred_sem_seg.cpu().data[0]
            ids = np.unique(result)
            columns_num = len(result[0])
            for id in ids:
                mask = []
                for pred_row in result:
                    row = []
                    for columns_i in range(columns_num):
                        row.append(1 if int(pred_row[columns_i]) == id else 0)
                    mask.append(row)
                image_annotations.add(
                    annotation_definition=dl.Segmentation(geo=np.array(mask),
                                                          label=self.coco_labels[id]))
                batch_annotations.append(image_annotations)
        return batch_annotations


if __name__ == '__main__':
    model = dl.models.get(model_id='65a9723545ad6d4b305a8546')
    adapter = Adapter(model_entity=model)
    item = dl.items.get(item_id='659dad958625651d89ca8e25')
    adapter.predict_items(items=[item])
