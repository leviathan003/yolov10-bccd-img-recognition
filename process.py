import os
import cv2
import time
import torch
from ultralytics import YOLOv10
from collections import defaultdict

weights_path = "./outputs/content/runs/detect/train/weights/last.pt"
model = YOLOv10(weights_path)

def process_results(results):
    classes = list(results[0].boxes.cls)
    dims = list(results[0].boxes.xywhn)

    return list(zip(classes, dims)) 

def detect_classes(img,out_path,filename):
    image = cv2.imread(img)
    if image is None:
        print(f"Error: Could not load image at {img}")
        return

    class_colors = {
        0: (0, 0, 255),  # Red for class 0 (RBC)
        1: (255, 0, 0),  # Blue for class 1 (WBC)
        2: (0, 255, 0),  # Green for class 2 (Platelets)
    }

    results = model.predict(image, conf=0.5)

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            label = f"{result.names[class_id]} {confidence:.2f}"
            color = class_colors.get(class_id, (255, 255, 255))

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, label, (x_min+10, y_max - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = out_path+filename+".jpg"
    try:
        cv2.imwrite(out_path, image)
        print(f"Updated image saved to {out_path}")
    except Exception as e:
        print(f"Failed to save image at {out_path}")
        return
    processed_results = process_results(results)
    return processed_results

def get_ground_truth(file_path):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return []
    
    tuples_list = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            tuples_list.append(tuple(map(float, values)))
    
    return tuples_list

def iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_area = max(0, inter_max_x - inter_min_x) * max(0, inter_max_y - inter_min_y)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    predictions = [(int(pred[0].item()), pred[1].tolist()) for pred in predictions]
    annotations = [[int(ann[0]), *ann[1:]] for ann in ground_truths]

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    matched_annotations = set()

    for pred_class, pred_box in predictions:
        matched = False
        for ann_idx, ann in enumerate(annotations):
            ann_class, ann_box = ann[0], ann[1:]
            if ann_idx in matched_annotations:
                continue

            if pred_class == ann_class:
                if iou(pred_box, ann_box) >= iou_threshold:
                    tp[pred_class] += 1
                    matched_annotations.add(ann_idx)
                    matched = True
                    break

        if not matched:
            fp[pred_class] += 1

    for ann_idx, ann in enumerate(annotations):
        ann_class = ann[0]
        if ann_idx not in matched_annotations:
            fn[ann_class] += 1

    class_metrics = {}
    all_tp = 0
    all_fp = 0
    all_fn = 0

    for cls in set(tp.keys()).union(fp.keys()).union(fn.keys()):
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        class_metrics[cls] = {"precision": precision, "recall": recall}
        all_tp += tp[cls]
        all_fp += fp[cls]
        all_fn += fn[cls]

    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0

    overall_metrics = {"precision": overall_precision, "recall": overall_recall}

    return class_metrics, overall_metrics



img_shape = (480,640)
filename = "BloodImage_00409"
img_ext = ".jpg"
txt_ext = ".txt"
in_img_path = "./data/images/train/"+filename+img_ext
out_img_path = "./results/"
annot_file = "./data/labels/train/"+filename+txt_ext
start_time = time.time()

pred = detect_classes(in_img_path,out_img_path,filename)
ground_truth = get_ground_truth(annot_file)
class_metrics, overall_metrics = calculate_metrics(pred, ground_truth)

print("Class Metrics:", class_metrics)
print("Overall Metrics:", overall_metrics)
end_time = time.time()

print(f"\nExecution completed in {end_time-start_time}s")

