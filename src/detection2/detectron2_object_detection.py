import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

setup_logger()

# Load an image
im = cv2.imread("./input.jpg")

# Create a Detectron2 config
cfg = get_cfg()

# Use a pre-trained model from the model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# Set the model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# Make prediction
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Visualize the prediction
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('image', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
