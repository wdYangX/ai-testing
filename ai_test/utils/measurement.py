import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from constant import cfg


def calculate_metrics(ground_truth_labels, predicted_labels, predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    """
    Calculates accuracy, precision, recall, and F1-score using confusion matrix.

    Parameters:
    ground_truth_labels (list): The ground truth labels.
    predicted_labels (list): The predicted labels.
    predicted_boxes (list): The predicted bounding boxes.
    ground_truth_boxes (list): The ground truth bounding boxes.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    # Synchronize data: remove invalid values from both predicted and ground truth
    min_length = min(len(ground_truth_labels), len(predicted_labels))
    valid_indices = [
        i for i, (gt_label, pred_label) in enumerate(zip(ground_truth_labels, predicted_labels))
        if gt_label is not None and pred_label is not None and not np.isnan(gt_label) and not np.isnan(pred_label)
    ]

    # Filter both lists based on the valid indices
    ground_truth_labels = [ground_truth_labels[i] for i in valid_indices][:min_length]
    predicted_labels = [predicted_labels[i] for i in valid_indices][:min_length]
    predicted_boxes = [predicted_boxes[i] for i in valid_indices][:min_length]
    ground_truth_boxes = [ground_truth_boxes[i] for i in valid_indices][:min_length]

    # Convert labels to consistent types if necessary (e.g., both to integers)
    ground_truth_labels = list(map(int, ground_truth_labels))  # or map to string if that's your desired format
    predicted_labels = list(map(int, predicted_labels))  # same here for predicted labels

    # Now we can proceed to calculate the metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_pred_labels = []
    all_gt_labels = []

    # Match predictions to ground truth
    for i in range(len(ground_truth_labels)):
        pred_label = predicted_labels[i]
        gt_label = ground_truth_labels[i]
        pred_box = predicted_boxes[i]
        gt_box = ground_truth_boxes[i]

        # Compute IoU between predicted box and ground truth box
        iou = compute_iou(pred_box, gt_box)

        # True Positive: If the predicted label matches the ground truth label and IoU >= threshold
        if pred_label == gt_label and iou >= iou_threshold:
            true_positives += 1
            all_pred_labels.append(pred_label)
            all_gt_labels.append(gt_label)
        else:
            if iou < iou_threshold:
                false_positives += 1
                all_pred_labels.append(pred_label)
                all_gt_labels.append(-1)  # No match in ground truth
            else:
                false_negatives += 1
                all_pred_labels.append(-1)
                all_gt_labels.append(gt_label)

    # Compute confusion matrix
    confusion = confusion_matrix(all_gt_labels, all_pred_labels, labels=list(set(all_gt_labels + all_pred_labels)))
    # If confusion matrix is for binary classification (2x2 matrix), unpack it
    if confusion.shape == (2, 2):
        TN, FP, FN, TP = confusion.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # For multiclass, handle metrics per class
        accuracy = accuracy_score(all_gt_labels, all_pred_labels)
        precision = precision_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0)
        recall = recall_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0)
        f1 = f1_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0)

    # Return the results as a dictionary
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1-Score": f1}



def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Parameters:
    box1 (list): The first bounding box [x1, y1, x2, y2].
    box2 (list): The second bounding box [x1, y1, x2, y2].

    Returns:
    float: The IoU score between the two boxes.
    """
    # Calculate the area of the first and second box
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    # Calculate the intersection area
    x1_intersection = max(x1, x1_gt)
    y1_intersection = max(y1, y1_gt)
    x2_intersection = min(x2, x2_gt)
    y2_intersection = min(y2, y2_gt)

    intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)

    # Calculate the union area
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou
