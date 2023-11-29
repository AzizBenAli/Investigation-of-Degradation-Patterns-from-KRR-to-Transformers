import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt


ROOT_DIR = r'C:\Users\benal\Mask_RCNN'  
sys.path.append(ROOT_DIR)


from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize


MODEL_DIR = os.path.join(ROOT_DIR, "logs")


COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") 

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


model.load_weights(COCO_MODEL_PATH, by_name=True)


image_path = '/path/to/your/image.jpg'  
image = skimage.io.imread(image_path)


results = model.detect([image], verbose=1)
class_names = ['BG', 'car']
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

plt.show()
