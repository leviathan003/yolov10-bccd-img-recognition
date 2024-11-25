import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict

# Function to process the results of the object detected image to make it compatible for metric calculation
def process_results(results):
    classes = list(results[0].boxes.cls)
    dims = list(results[0].boxes.xywhn)

    return list(zip(classes, dims)) 

# Function to get the ground truth of the annotations from the annotation file of that image
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

# Function to calculate intersection over union essential to find the precision and recall
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

# Function to calculate the precison and recall metrics
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

# Calling function to detect objects
def detect_classes(image,out_path,filename,weights='last'):

    if isinstance(image, Image.Image):  # Check if it's a PIL image
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif isinstance(image, str): # Check if it's a path
        image = cv2.imread(image)

    if image is None:
        print(f"Error: Could not load image at {image}")
        return

    class_colors = {
        0: (0, 0, 255),  # Red for class 0 (RBC)
        1: (255, 0, 0),  # Blue for class 1 (WBC)
        2: (21, 51, 10),  # Green for class 2 (Platelets)
    }

    weights_path = f"./outputs/content/runs/detect/train/weights/{weights}.pt"
    model = YOLO(weights_path)
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

    # out_path = out_path+filename+".jpg"
    # try:
    #     cv2.imwrite(out_path, image)
    #     print(f"Updated image saved to {out_path}")
    # except Exception as e:
    #     print(f"Failed to save image at {out_path}")
    #     return
    
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(annotated_image)

    processed_results = process_results(results)
    ground_truth = get_ground_truth("./data/labels/train/"+filename+".txt")
    class_metrics, overall_metrics = calculate_metrics(processed_results, ground_truth)

    return class_metrics, overall_metrics, pil_image