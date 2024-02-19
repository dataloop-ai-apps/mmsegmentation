import os
from mmseg.apis import inference_model, init_model
import torch
import dtlpy as dl
import numpy as np
import logging
import subprocess

logger = logging.getLogger('MMSegmentation')


@dl.Package.decorators.module(description='Model Adapter for mmlabs object segmentation',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class MMSegmentation(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        self.model = None
        self.confidence_thr = model_entity.configuration.get('confidence_thr', 0.4)
        self.device = model_entity.configuration.get('device', None)
        super(MMSegmentation, self).__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        model_name = self.model_entity.configuration.get('model_name', 'pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512')
        config_file = self.model_entity.configuration.get('config_file',
                                                          'pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512.py')
        checkpoint_file = self.model_entity.configuration.get('checkpoint_file',
                                                              'pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth')
        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            logger.info("Downloading MMSegmentation artifacts")
            download_status = subprocess.Popen(f"mim download mmsegmentation --config {model_name} --dest .",
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               shell=True)
            download_status.wait()
            (out, err) = download_status.communicate()
            if download_status.returncode != 0:
                raise Exception(f'Failed to download MMSegmentation artifacts: {err}')
            logger.info(f"MMSegmentation artifacts downloaded successfully, Loading Model {out}")

        if self.device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model on device {self.device}")
        self.model = init_model(config_file, checkpoint_file, device=self.device)
        logger.info("Model Loaded Successfully")

    def predict(self, batch, **kwargs):
        batch_annotations = list()
        for image in batch:
            image_annotations = dl.AnnotationCollection()
            result = inference_model(self.model, image).pred_sem_seg.cpu().data[0].numpy()
            unique_ids = np.unique(result)
            for unique_id in unique_ids:
                mask = np.zeros_like(result)
                mask[result == unique_id] = 1
                image_annotations.add(
                    annotation_definition=dl.Segmentation(geo=mask,
                                                          label=self.model_entity.labels[unique_id]))
            batch_annotations.append(image_annotations)
        return batch_annotations
