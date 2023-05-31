from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.helpers import IMAGE
from extra.utils import get_child, download_file
import numpy as np

from pathlib import Path
import re
from math import prod

from models.resnet import ResNet
from models.retinanet import ResNetFPN

assert IMAGE

class MaskRCNN:
  def __init__(self, backbone: ResNet):
    self.transform = RcnnTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    self.backbone = ResNetFPN(backbone, out_channels=256, returned_layers=[1, 2, 3, 4])
    self.rpn = RPN(self.backbone.out_channels)
    self.roi_heads = RoIHeads(self.backbone.out_channels, 91)

  def __call__(self, x):
    transformed_img = self.transform(x)
    features = self.backbone(transformed_img)
    proposals = self.rpn(features)
    detections = self.roi_heads(features, proposals)
    return detections

  def load_from_pretrained(self):
    fn = Path(__file__).parent.parent / "weights/maskrcnn.pt"
    
    model_urls = {0: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth", # faithful the the original paper https://arxiv.org/abs/1703.06870
                  1: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth", # improvement based on Detection Transfer Learning with ViT https://arxiv.org/abs/2111.11429
                  2: "https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth",
                  }
    download_file(model_urls[2], fn)

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


# based on https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
# TODO: make it inline
def bilinear_resize_vectorized(image: Tensor, height, width, grad: Tensor):
  """
  `image` is a 2-D array, holding the input image
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  `grad` is a 2-D array of shape [height, width], holding the gradient to be
    backpropped to `image`.
  """
  img_height, img_width = image.shape

  image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  grad = grad.ravel()

  # gradient wrt `a`, `b`, `c`, `d`
  d_a = (1 - x_weight) * (1 - y_weight) * grad
  d_b = x_weight * (1 - y_weight) * grad
  d_c = y_weight * (1 - x_weight) * grad
  d_d = x_weight * y_weight * grad

  # [4 * height * width]
  grad = np.concatenate([d_a, d_b, d_c, d_d])
  # [4 * height * width]
  indices = np.concatenate([y_l * img_width + x_l,
                            y_l * img_width + x_h,
                            y_h * img_width + x_l,
                            y_h * img_width + x_h])

  # we must route gradients in `grad` to the correct indices of `image` in
  # `indices`, e.g. only entries of indices `y_l * img_width + x_l` in `image`
  # gets the gradient backpropped from `a`.

  # use numpy's broadcasting rule to generate 2-D array of shape
  # [4 * height * width, img_height * img_width]
  indices = (indices.reshape((-1, 1)) ==
              np.arange(img_height * img_width).reshape((1, -1)))
  d_image = np.apply_along_axis(lambda col: grad[col].sum(), 0, indices)

  return d_image.reshape((img_height, img_width))


# based on https://github.com/pytorch/vision/blob/01b9faa16cfeacbb70aa33bd18534de50891786b/torchvision/models/detection/transform.py
class RcnnTransform:
  def __init__(self, min_size: int, max_size: int, image_mean: List[float], image_std: List[float]):
    self.min_size = min_size
    self.max_size = max_size
    self.image_mean = image_mean
    self.image_std = image_std

  def forward(self, images:List[Tensor]):
    for i in range(len(images)):
      # inplace normalize&resize?
      self.normalize(images[i])
      images[i] = self.resize(images[i])

    #batching?

  # this happens inplace?
  def normalize(self, image: Tensor):
    mean = Tensor(self.image_mean, dtype=image.dtype, device=image.device)
    std = Tensor(self.image_std, dtype=image.dtype, device=image.device)
    (image - mean[:, None, None]) / std[:, None, None]

  # make it happen inplace
  def resize(self, image: Tensor):
    im_shape = Tensor(image.shape[-2:])
    im_min_dim = min(im_shape)
    im_max_dim = max(im_shape)
    scale_factor = min(self.min_size / im_min_dim, self.max_size / im_max_dim)
    resized_im_shape = (scale_factor * x for x in im_shape)
    return bilinear_resize_vectorized(image, *resized_im_shape, grad)

  
  
  
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
