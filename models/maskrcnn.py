from tinygrad.tensor import Tensor
from tinygrad import nn
from extra.utils import get_child, download_file
import numpy as np

from pathlib import Path
import re
from math import prod

from models.resnet import ResNet
from models.retinanet import ResNetFPN


class MaskRCNN:
  def __init__(self, backbone: ResNet):
    self.backbone = ResNetFPN(backbone, out_channels=256, returned_layers=[1, 2, 3, 4])
    self.rpn = RPN(self.backbone.out_channels)
    self.roi_heads = RoIHeads(self.backbone.out_channels, 91)

  def __call__(self, x):
    features = self.backbone(x)
    proposals = self.rpn(features)
    detections = self.roi_heads(features, proposals)
    return detections

  def load_from_pretrained(self):
    fn = Path(__file__).parent.parent / "weights/maskrcnn.pt"
    download_file("https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth", fn)

    import torch
    with open(fn, "rb") as f:
      state_dict = torch.load(f, map_location=torch.device("cpu"))["model"]

    for k, v in state_dict.items():
      if "module." in k:
        k = k.replace("module.", "")
      if "stem." in k:
        k = k.replace("stem.", "")
      if "fpn_inner" in k:
        block_index = int(re.search(r"fpn_inner(\d+)", k).group(1))
        k = re.sub(r"fpn_inner\d+", f"inner_blocks.{block_index - 1}", k)
      if "fpn_layer" in k:
        block_index = int(re.search(r"fpn_layer(\d+)", k).group(1))
        k = re.sub(r"fpn_layer\d+", f"layer_blocks.{block_index - 1}", k)
      print(k)
      get_child(self, k).assign(v.numpy()).realize()


class RPN:
  def __init__(self, in_channels):
    self.anchor_generator = AnchorGenerator()
    self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location()[0])

  def __call__(self, x):
    pass

class AnchorGenerator:
  def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), strides=(4, 8, 16, 32, 64)):
    anchors = [generate_anchors(stride, (size,), aspect_ratios) for stride, size in zip(strides, sizes)]
    self.cell_anchors = [Tensor(a) for a in anchors]

  def __call__(self, image_list, feature_maps):
    pass

  def num_anchors_per_location(self):
    return [cell_anchors.shape[0] for cell_anchors in self.cell_anchors]

# anchor generation code below is from the reference implementation here: https://github.com/mlcommons/training/blob/master/object_detection/pytorch/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
  return _generate_anchors(stride, np.array(sizes, dtype=np.float32) / stride, np.array(aspect_ratios, dtype=np.float32))

def _generate_anchors(base_size, scales, aspect_ratios):
  anchor = np.array([1, 1, base_size, base_size], dtype=np.float32) - 1
  anchors = _ratio_enum(anchor, aspect_ratios)
  anchors = np.vstack(
    [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
  )
  return anchors

def _whctrs(anchor):
  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((
    x_ctr - 0.5 * (ws - 1),
    y_ctr - 0.5 * (hs - 1),
    x_ctr + 0.5 * (ws - 1),
    y_ctr + 0.5 * (hs - 1),
  ))
  return anchors

def _ratio_enum(anchor, ratios):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

def _scale_enum(anchor, scales):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

class RPNHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    self.cls_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
    self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

  def __call__(self, x):
    logits = []
    bbox_reg = []
    for feature in x:
      t = self.conv(feature).relu()
      logits.append(self.cls_logits(t))
      bbox_reg.append(self.bbox_pred(t))

class RoIHeads:
  def __init__(self, in_channels, num_classes):
    self.box = RoIBoxHead(in_channels)

  def __call__(self, features, proposals):
    box_features = self.box(features, proposals)
    return box_features

class RoIBoxHead:
  def __init__(self, in_channels):
    self.feature_extractor = RoIBoxFeatureExtractor(in_channels)
    self.predictor = Predictor(1024, 2)
    self.post_processor = PostProcessor()

  def __call__(self, features, proposals):
    x = self.feature_extractor(features, proposals)
    class_logits, box_regression = self.predictor(x)
    return self.post_processor(class_logits, box_regression, proposals)

class RoIBoxFeatureExtractor:
  def __init__(self, in_channels):
    self.pooler = Pooler(7, 1/16)

class Pooler:
  def __init__(self, output_size, scales):
    self.output_size = output_size
    self.scales = scales
