from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2, os
from  matplotlib import pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from resnet import build_delta_resnet_backbone
import torch
from deltacnn import DCConv2d, DCBackend
from torch.utils.tensorboard import SummaryWriter
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1000
# os.environ['DCConv2d.backend'] = "DCBackend.deltacnn"
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
DCConv2d.backend = DCBackend.cudnn

im = cv2.imread("./input.jpg")

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.MODEL.ROI_HEADS.NAME = 'DeltaRes5ROIHeads'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.PRECISION = 'float32'
cfg.MODEL.DEVICE='cuda'

cfg.MODEL.BACKBONE.NAME = 'build_delta_resnet_backbone'
# cfg.MODEL.WEIGHTS = "D:/Samsung/Delta-Detectron2/instance_segmentation_resnet50_channel_last.pkl"

cfg.MODEL.WEIGHTS = "D:/Samsung/Delta-Detectron2/instance_segmentation_resnet50.pkl"
predictor = DefaultPredictor(cfg)

torch.cuda.empty_cache()
# with torch.cuda.amp.autocast():
outputs = predictor(im)

print(outputs)