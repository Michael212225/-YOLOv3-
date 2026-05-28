import cv2
import numpy as np
import paddle
import os
import matplotlib.pyplot as plt

# 从 train 导入需要的网络结构和常用参数
from train import YOLOv3, INSECT_NAMES, NUM_CLASSES, ANCHORS, ANCHOR_MASKS

def decode_yolo_outputs(outputs, orig_h, orig_w, conf_thresh=0.5):
    """将YOLO的多层特征图输出通过官方算子解码为具体的坐标和置信度"""
    boxes_list = []
    scores_list = []
    downsample = 32
    
    # yolo_box算子需要的原图尺寸（用于把预测框映射回原图级别）
    img_size = paddle.to_tensor([[orig_h, orig_w]], dtype='int32')
    
    for i, out in enumerate(outputs):
        # 获取当前层对应的 anchors
        anchor_mask = ANCHOR_MASKS[i]
        masked_anchors = []
        for m in anchor_mask:
            masked_anchors.extend([ANCHORS[2*m], ANCHORS[2*m+1]])
            
        # paddle.vision.ops.yolo_box 将特征图解码并筛选出满足 conf_thresh 的候选框
        boxes, scores = paddle.vision.ops.yolo_box(
            x=out,
            img_size=img_size,
            anchors=masked_anchors,
            class_num=NUM_CLASSES,
            conf_thresh=conf_thresh,
            downsample_ratio=downsample
        )
        boxes_list.append(boxes)
        scores_list.append(scores)
        downsample //= 2
        
    # 拼接3个尺度的预测框 [N, total_boxes, 4] 和置信度 [N, total_boxes, class_num]
    yolo_boxes = paddle.concat(boxes_list, axis=1) 
    yolo_scores = paddle.concat(scores_list, axis=1) 
    
    return yolo_boxes[0].numpy(), yolo_scores[0].numpy()

def non_max_suppression(boxes, scores, score_thresh=0.5, nms_thresh=0.45):
    """手写多类别 NMS（非极大值抑制）"""
    results = []
    for c in range(NUM_CLASSES):
        cls_scores = scores[:, c]
        valid_idx = cls_scores > score_thresh
        if not np.any(valid_idx):
            continue
        
        valid_boxes = boxes[valid_idx]
        valid_scores = cls_scores[valid_idx]
        
        # 按置信度降序排序
        order = valid_scores.argsort()[::-1]
        valid_boxes = valid_boxes[order]
        valid_scores = valid_scores[order]
        
        while order.size > 0:
            # 保留置信度最高的框
            best_box = valid_boxes[0]
            results.append((best_box, valid_scores[0], c))
            
            if order.size == 1:
                break
            
            # 计算其余框与当前最高分框的 IoU
            xx1 = np.maximum(best_box[0], valid_boxes[1:, 0])
            yy1 = np.maximum(best_box[1], valid_boxes[1:, 1])
            xx2 = np.minimum(best_box[2], valid_boxes[1:, 2])
            yy2 = np.minimum(best_box[3], valid_boxes[1:, 3])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            area0 = (best_box[2] - best_box[0] + 1) * (best_box[3] - best_box[1] + 1)
            area1 = (valid_boxes[1:, 2] - valid_boxes[1:, 0] + 1) * (valid_boxes[1:, 3] - valid_boxes[1:, 1] + 1)
            iou = inter / (area0 + area1 - inter)
            
            # 剔除 IoU 大于阈值的冗余框
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]
            valid_boxes = valid_boxes[inds + 1]
            valid_scores = valid_scores[inds + 1]
            
    return results

def draw_results(image_path, results, out_path="predict_result.jpg"):
    """绘制框并保存/显示图片"""
    # 修复使用 cv2.imread 读取包含中文路径会报错的问题
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 为不同类别准备不同颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (100, 100, 255)]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    ax = plt.gca()
    
    # 绘制每一个筛选后的检测框
    for box, score, cls_id in results:
        x1, y1, x2, y2 = box
        name = INSECT_NAMES[cls_id]
        color = np.array(colors[cls_id % len(colors)]) / 255.0
        
        # 画矩形框
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # 贴上文本标签（类别+置信度）
        text = f'{name} {score:.2f}'
        ax.text(x1, y1 - 3, text, bbox=dict(facecolor=color, alpha=0.5, pad=2), 
                fontsize=10, color='white')
        
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n✅ 可视化预测结果已生成！请在当前目录下查看图片文件: {out_path}")

if __name__ == '__main__':
    # 1. 实例化网络模型
    model = YOLOv3(num_classes=NUM_CLASSES)
    
    # 2. 加载训练好的权重参数 (使用刚才训练完成的第 79 轮模型)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 模型保存在根目录下，所以从 yolo3 目录查找父目录
    params_path = os.path.join(current_dir, '..', 'yolo_epoch79.pdparams')
    
    if os.path.exists(params_path):
        model.set_state_dict(paddle.load(params_path))
        print(f"✅ 成功加载预训练模型权重: {params_path}")
    else:
        print(f"❌ 找不到模型权重文件: {params_path}")
        exit()
    
    model.eval() # 切换到预测模式
    
    # 3. 指定测试图片 (使用项目内的相对路径)
    test_image_dir = os.path.join(current_dir, 'work', 'test', 'images')
    if not os.path.exists(test_image_dir):
        # 兜底：如果路径不对，尝试当前目录下的 data
        test_image_dir = os.path.join(current_dir, 'data', 'test', 'images')
    
    if os.path.exists(test_image_dir):
        test_images = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        test_images = []
    
    if len(test_images) > 0:
        preferred_image = '1838.jpeg'
        if preferred_image in test_images:
            test_img_path = os.path.join(test_image_dir, preferred_image)
        else:
            test_img_path = os.path.join(test_image_dir, test_images[0])
        print(f"\n开始测试图片: {test_img_path}")
        
        # 4. 图片预处理 (处理中文路径 cv2 读取为空的问题)
        orig_img = cv2.imdecode(np.fromfile(test_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        orig_h, orig_w = orig_img.shape[:2]
        
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,-1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1,1,-1))
        img = (img / 255.0 - mean) / std
        img = img.astype('float32').transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        img_tensor = paddle.to_tensor(img)
        
        # 5. 模型推理
        with paddle.no_grad():
            outputs = model(img_tensor)
            
        # 6. 利用 YOLO 解码和 NMS 提取最终边界框
        boxes, scores = decode_yolo_outputs(outputs, orig_h, orig_w, conf_thresh=0.05)
        results = non_max_suppression(boxes, scores, score_thresh=0.1, nms_thresh=0.45)
        
        print(f"预测完毕，图片中共检测到 {len(results)} 个有效虫子目标。")
        
        # 7. 可视化
        draw_results(test_img_path, results, out_path="predict_result.jpg")

