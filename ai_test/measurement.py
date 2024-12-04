import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(ground_truth_labels, predicted_labels):
    """
    Calculates accuracy, precision, recall, and F1-score.

    Parameters:
    ground_truth_labels (list): The ground truth labels.
    predicted_labels (list): The predicted labels.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    # Align labels based on the shorter length
    matched_ground_truth = ground_truth_labels[:len(predicted_labels)]

    # Compute metrics
    accuracy = accuracy_score(matched_ground_truth, predicted_labels)
    precision = precision_score(matched_ground_truth, predicted_labels, average="weighted")
    recall = recall_score(matched_ground_truth, predicted_labels, average="weighted")
    f1 = f1_score(matched_ground_truth, predicted_labels, average="weighted")

    # Return metrics as a dictionary
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1-Score": f1}


def convert_to_absolute_coords(label, img_width, img_height):
    """
    Convert label in normalized coordinates to absolute pixel coordinates.

    Args:
    - label (list): [class_id, x_center, y_center, width, height]
    - img_width (int): Image width
    - img_height (int): Image height

    Returns:
    - list: [x_min, y_min, x_max, y_max]
    """
    class_id, x_center, y_center, width, height = label
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    x_max = (x_center + width / 2) * img_width
    y_max = (y_center + height / 2) * img_height
    return [x_min, y_min, x_max, y_max]


def compute_iou(pred_box, gt_box):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
    - pred_box (list): [x_min, y_min, x_max, y_max] for predicted box
    - gt_box (list): [x_min, y_min, x_max, y_max] for ground truth box

    Returns:
    - float: IoU value
    """
    x1_intersection = max(pred_box[0], gt_box[0])
    y1_intersection = max(pred_box[1], gt_box[1])
    x2_intersection = min(pred_box[2], gt_box[2])
    y2_intersection = min(pred_box[3], gt_box[3])

    intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)

    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    union_area = pred_area + gt_area - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def calculate_precision_recall(true_labels, predicted_boxes, iou_threshold=0.5):
    """
    Calculate Precision and Recall for object detection predictions.

    Args:
    - true_labels (list): List of ground truth bounding boxes and their class ids.
    - predicted_boxes (list): List of predicted bounding boxes with confidence scores.
    - iou_threshold (float): Threshold for considering a prediction as valid.

    Returns:
    - precision (float): Precision for the predictions.
    - recall (float): Recall for the predictions.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(true_labels)

    for pred in predicted_boxes:
        best_iou = 0
        best_gt = None
        for gt in true_labels:
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_threshold:
            true_positives += 1
            false_negatives -= 1
            true_labels.remove(best_gt)  # Remove the matched ground truth
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


def calculate_map(true_labels, predicted_labels, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP).

    Args:
    - true_labels (list): List of ground truth labels with bounding boxes.
    - predicted_labels (list): List of predicted labels with bounding boxes and confidence scores.
    - iou_threshold (float): Threshold for considering a prediction as valid.

    Returns:
    - float: The mean Average Precision (mAP).
    """
    precisions = []
    recalls = []

    # Sort predictions by confidence score
    predicted_labels.sort(key=lambda x: x['confidence'], reverse=True)

    for pred in predicted_labels:
        precision, recall = calculate_precision_recall(true_labels.copy(), [pred], iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    ap = np.mean(precisions)  # Average of precision values for AP
    return ap


