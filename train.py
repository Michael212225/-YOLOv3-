# -*- coding: utf-8 -*-
import sys, os, time, json, random
import numpy as np
import cv2, paddle
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.image import imread

paddle.device.set_device("gpu")
INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

def get_insect_names():
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i
    return insect_category2id

def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def get_annotations(cname2cid, datadir):
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')
        tree = ET.parse(fpath)
        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])
        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), ), dtype=np.int32)
        is_crowd = np.zeros((len(objs), ), dtype=np.int32)
        difficult = np.zeros((len(objs), ), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(im_w - 1, x2); y2 = min(im_h - 1, y2)
            gt_bbox[i] = [(x1+x2)/2.0, (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
            is_crowd[i] = 0; difficult[i] = _difficult
        voc_rec = {
            'im_file': img_file, 'im_id': im_id, 'h': im_h, 'w': im_w,
            'is_crowd': is_crowd, 'gt_class': gt_class, 'gt_bbox': gt_bbox,
            'gt_poly': [], 'difficult': difficult}
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records

def get_bbox(gt_bbox, gt_class):
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2

def get_img_data_from_file(record):
    im_file = record['im_file']
    h, w = record['h'], record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']
    img = imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert img.shape[0] == int(h), f"image height of {im_file} inconsistent"
    assert img.shape[1] == int(w), f"image width of {im_file} inconsistent"
    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)
    gt_boxes[:, 0] /= float(w); gt_boxes[:, 1] /= float(h)
    gt_boxes[:, 2] /= float(w); gt_boxes[:, 3] /= float(h)
    return img, gt_boxes, gt_labels, (h, w)

def random_distort(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)
    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)
    img = Image.fromarray(img)
    img = ops[0](img); img = ops[1](img); img = ops[2](img)
    return np.asarray(img)

def random_expand(img, gtboxes, max_ratio=4., fill=None, keep_ratio=True, thresh=0.5):
    if random.random() > thresh or max_ratio < 1.0:
        return img.copy(), gtboxes.copy()
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    ratio_y = ratio_x if keep_ratio else random.uniform(1, max_ratio)
    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)
    out_img = np.zeros((oh, ow, c), dtype=img.dtype)
    if fill is not None and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = int(fill[i] * 255.0)
    out_img[off_y:off_y+h, off_x:off_x+w, :] = img
    new_gtboxes = gtboxes.copy()
    new_gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / ow
    new_gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / oh
    new_gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    new_gtboxes[:, 3] = gtboxes[:, 3] / ratio_y
    return out_img, new_gtboxes

def multi_box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4 and box2.shape[-1] == 4
    b1_x1 = box1[:, 0] - box1[:, 2]/2; b1_y1 = box1[:, 1] - box1[:, 3]/2
    b1_x2 = box1[:, 0] + box1[:, 2]/2; b1_y2 = box1[:, 1] + box1[:, 3]/2
    b2_x1 = box2[:, 0] - box2[:, 2]/2; b2_y1 = box2[:, 1] - box2[:, 3]/2
    b2_x2 = box2[:, 0] + box2[:, 2]/2; b2_y2 = box2[:, 1] + box2[:, 3]/2
    inter_x1 = np.maximum(b1_x1, b2_x1); inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2); inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area + b2_area - inter_area
    return inter_area / (union + 1e-6)

def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = crop
    im_w, im_h = img_shape
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * im_w
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * im_w
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * im_h
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * im_h
    crop_xyxy = np.array([x, y, x+w, y+h])
    centers = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
    mask = np.logical_and(
        (crop_xyxy[:2] <= centers), (centers <= crop_xyxy[2:])).all(axis=1)
    boxes_xyxy[:, :2] = np.maximum(boxes_xyxy[:, :2], crop_xyxy[:2])
    boxes_xyxy[:, 2:] = np.minimum(boxes_xyxy[:, 2:], crop_xyxy[2:])
    boxes_xyxy[:, :2] -= crop_xyxy[:2]
    boxes_xyxy[:, 2:] -= crop_xyxy[:2]
    valid = (boxes_xyxy[:, :2] < boxes_xyxy[:, 2:]).all(axis=1)
    mask = np.logical_and(mask, valid)
    crop_boxes = np.zeros_like(boxes)
    crop_boxes[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / w
    crop_boxes[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / h
    crop_boxes[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / w
    crop_boxes[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / h
    crop_boxes *= np.expand_dims(mask.astype(np.float32), axis=1)
    crop_labels = labels.astype(np.float32) * mask.astype(np.float32)
    crop_labels = crop_labels.astype(labels.dtype)
    return crop_boxes, crop_labels, mask.sum()

def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if len(boxes) == 0:
        return img.copy(), boxes.copy(), labels.copy()
    if constraints is None:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.0, 1.0)]
    img_pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img.copy()
    im_w, im_h = img_pil.size
    crops = [(0, 0, im_w, im_h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1/max_ratio, scale**2), min(max_ratio, 1/(scale**2)))
            crop_h = int(im_h * scale / np.sqrt(aspect_ratio))
            crop_w = int(im_w * scale * np.sqrt(aspect_ratio))
            if crop_w >= im_w or crop_h >= im_h:
                continue
            crop_x = random.randint(0, im_w - crop_w)
            crop_y = random.randint(0, im_h - crop_h)
            crop_xywh = np.array([
                (crop_x + crop_w/2)/im_w, (crop_y + crop_h/2)/im_h,
                crop_w/im_w, crop_h/im_h]).reshape(1, 4)
            iou = multi_box_iou_xywh(crop_xywh, boxes)
            if min_iou <= iou.min() and iou.max() <= max_iou:
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break
    while crops:
        crop = crops.pop(np.random.randint(len(crops)))
        x, y, w, h = crop
        cropped_img = np.asarray(img_pil.crop((x, y, x+w, y+h)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (im_w, im_h))
        if box_num >= 1:
            return cropped_img, crop_boxes, crop_labels
    return np.asarray(img_pil), boxes, labels

def random_interp(img, size, interp=None):
    interp_method = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method)-1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    return cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)

def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :].copy()
        gtboxes = gtboxes.copy()
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes

def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate([gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]

def image_augment(img, gtboxes, gtlabels, size, means=None):
    img = random_distort(img)
    img, gtboxes = random_expand(img, gtboxes, fill=means)
    img, gtboxes, gtlabels = random_crop(img, gtboxes, gtlabels)
    img = random_interp(img, size)
    img, gtboxes = random_flip(img, gtboxes)
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)
    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')

def get_img_data(record, size=640):
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,-1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1,1,-1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))
    return img, gt_boxes, gt_labels, scales

# === Network ===
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, groups=1, padding=0, act="leaky"):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=ch_in, out_channels=ch_out,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02)),
            bias_attr=False)
        self.batch_norm = paddle.nn.BatchNorm2D(
            num_features=ch_out,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0., 0.02), regularizer=paddle.regularizer.L2Decay(0.)),
            bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0), regularizer=paddle.regularizer.L2Decay(0.)))
        self.act = act
    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = paddle.nn.functional.leaky_relu(out, negative_slope=0.1)
        return out

class DownSample(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, inputs):
        return self.conv_bn_layer(inputs)

class BasicBlock(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        return paddle.add(inputs, conv2)

class LayerWarp(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, count):
        super().__init__()
        self.basicblock0 = BasicBlock(ch_in, ch_out)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer(f"basic_block_{i}", BasicBlock(ch_out*2, ch_out))
            self.res_out_list.append(res_out)
    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y

DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}

class DarkNet53_conv_body(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.stages = DarkNet_cfg[53][0:5]
        self.conv0 = ConvBNLayer(ch_in=3, ch_out=32, kernel_size=3, stride=1, padding=1)
        self.downsample0 = DownSample(ch_in=32, ch_out=64)
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(f"stage_{i}", LayerWarp(32*(2**(i+1)), 32*(2**i), stage))
            self.darknet53_conv_block_list.append(conv_block)
        for i in range(len(self.stages)-1):
            downsample = self.add_sublayer(f"stage_{i}_downsample", DownSample(ch_in=32*(2**(i+1)), ch_out=32*(2**(i+2))))
            self.downsample_list.append(downsample)
    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]

class YoloDetectionBlock(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        assert ch_out % 2 == 0
        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv1 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNLayer(ch_in=ch_out*2, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
        self.route = ConvBNLayer(ch_in=ch_out*2, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.tip = ConvBNLayer(ch_in=ch_out, ch_out=ch_out*2, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out); out = self.conv2(out); out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip

class Upsample(paddle.nn.Layer):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
    def forward(self, inputs):
        return paddle.nn.functional.interpolate(inputs, scale_factor=self.scale, mode="NEAREST")

class YOLOv3(paddle.nn.Layer):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        for i in range(3):
            yolo_block = self.add_sublayer(f"yolo_detecton_block_{i}",
                YoloDetectionBlock(ch_in=512//(2**i)*2 if i==0 else 512//(2**i)*2 + 512//(2**i), ch_out=512//(2**i)))
            self.yolo_blocks.append(yolo_block)
            num_filters = 3 * (num_classes + 5)
            block_out = self.add_sublayer(f"block_out_{i}",
                paddle.nn.Conv2D(in_channels=512//(2**i)*2, out_channels=num_filters, kernel_size=1, stride=1, padding=0))
            self.block_outputs.append(block_out)
            if i < 2:
                route = self.add_sublayer(f"route2_{i}",
                    ConvBNLayer(ch_in=512//(2**i), ch_out=256//(2**i), kernel_size=1, stride=1, padding=0))
                self.route_blocks_2.append(route)
            self.upsample = Upsample()
    def forward(self, inputs):
        outputs = []
        blocks = self.block(inputs)
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            block_out = self.block_outputs[i](tip)
            outputs.append(block_out)
            if i < 2:
                route = self.route_blocks_2[i](route)
                route = self.upsample(route)
        return outputs
    def get_loss(self, outputs, gtbox, gtlabel, gtscore=None,
                 anchors=[10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326],
                 anchor_masks=[[6,7,8],[3,4,5],[0,1,2]], ignore_thresh=0.7, use_label_smooth=False):
        self.losses = []
        downsample = 32
        for i, out in enumerate(outputs):
            anchor_mask_i = anchor_masks[i]
            loss = paddle.vision.ops.yolo_loss(
                x=out, gt_box=gtbox, gt_label=gtlabel, gt_score=gtscore,
                anchors=anchors, anchor_mask=anchor_mask_i, class_num=self.num_classes,
                ignore_thresh=ignore_thresh, downsample_ratio=downsample, use_label_smooth=False)
            self.losses.append(paddle.mean(loss))
            downsample //= 2
        return sum(self.losses)

class TrainDataset(paddle.io.Dataset):
    def __init__(self, datadir, mode='train'):
        self.datadir = datadir
        cname2cid = get_insect_names()
        records = get_annotations(cname2cid, datadir)
        self.records = records
        self.img_size = 640
    def __getitem__(self, idx):
        record = self.records[idx]
        img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=self.img_size)
        return img, gt_bbox, gt_labels, np.array(im_shape)
    def __len__(self):
        return len(self.records)

def get_lr(base_lr=0.0001, lr_decay=0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr*lr_decay, base_lr*lr_decay*lr_decay]
    return paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

# === MAIN TRAINING ===
ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
IGNORE_THRESH = .7
NUM_CLASSES = 7

# 使用相对路径以增强代码的可移植性
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAINDIR = os.path.join(current_dir, 'work', 'train')
VALIDDIR = os.path.join(current_dir, 'work', 'val')

if __name__ == '__main__':
    # 进一步降低 batch_size 到 1，并添加显存优化配置以解决显存不足问题
    batch_size = 1
    # 设置 PaddlePaddle 显存配置，尝试限制显存占用
    os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.80' 
    
    train_dataset = TrainDataset(TRAINDIR, mode='train')
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, use_shared_memory=False)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, use_shared_memory=False)

    model = YOLOv3(num_classes=NUM_CLASSES)
    learning_rate = get_lr()
    opt = paddle.optimizer.Momentum(learning_rate=learning_rate, momentum=0.9, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())

    MAX_EPOCH = 80
    train_log_path = os.path.join(current_dir, 'epoch_metrics.csv')
    with open(train_log_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,valid_loss\n')
    print('Start training...')
    for epoch in range(MAX_EPOCH):
        train_loss_sum = 0.0
        train_loss_count = 0
        for i, data in enumerate(train_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = paddle.ones(gt_labels.shape, dtype='float32')
            outputs = model(img)
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors=ANCHORS, anchor_masks=ANCHOR_MASKS, ignore_thresh=IGNORE_THRESH, use_label_smooth=False)
            loss.backward()
            opt.step()
            opt.clear_grad()
            train_loss_sum += float(loss.numpy().item())
            train_loss_count += 1
            if i % 10 == 0:
                print(f'[TRAIN] epoch {epoch}, iter {i}, loss: {loss.numpy().item():.4f}')
        # Validation
        model.eval()
        valid_loss_sum = 0.0
        valid_loss_count = 0
        for i, data in enumerate(valid_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = paddle.ones(gt_labels.shape, dtype='float32')
            outputs = model(img)
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors=ANCHORS, anchor_masks=ANCHOR_MASKS, ignore_thresh=IGNORE_THRESH, use_label_smooth=False)
            valid_loss_sum += float(loss.numpy().item())
            valid_loss_count += 1
            if i == 0:
                print(f'[VALID] epoch {epoch}, loss: {loss.numpy().item():.4f}')
        model.train()
        train_loss_epoch = train_loss_sum / max(1, train_loss_count)
        valid_loss_epoch = valid_loss_sum / max(1, valid_loss_count)
        with open(train_log_path, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{train_loss_epoch:.6f},{valid_loss_epoch:.6f}\n')
        paddle.save(model.state_dict(), f'yolo_epoch{epoch}.pdparams')
        print(f'Saved yolo_epoch{epoch}.pdparams')
    print('Done!')
