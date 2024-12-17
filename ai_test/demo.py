import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(ground_truth_labels, predicted_labels, predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    """
    Calculates accuracy, precision, recall, and F1-score.

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

    # Trim excess elements
    ground_truth_labels = ground_truth_labels[:min_length]
    predicted_labels = predicted_labels[:min_length]
    predicted_boxes = predicted_boxes[:min_length]
    ground_truth_boxes = ground_truth_boxes[:min_length]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_predictions = 0
    total_predictions = 0
    all_pred_labels = []
    all_gt_labels = []

    # Match predictions to ground truth
    for i in range(min_length):
        pred_label = predicted_labels[i]
        gt_label = ground_truth_labels[i]
        pred_box = predicted_boxes[i]
        gt_box = ground_truth_boxes[i]

        # Compute IoU between predicted box and ground truth box
        iou = compute_iou(pred_box, gt_box)

        # True Positive: If the predicted label matches the ground truth label and IoU >= threshold
        if pred_label == gt_label and iou >= iou_threshold:
            true_positives += 1
            correct_predictions += 1
            all_pred_labels.append(pred_label)
            all_gt_labels.append(gt_label)
        else:
            if iou < iou_threshold:
                false_positives += 1
                all_pred_labels.append(pred_label)
                all_gt_labels.append(None)  # No match in ground truth
            else:
                false_negatives += 1
                all_pred_labels.append(None)
                all_gt_labels.append(gt_label)

        total_predictions += 1  # Total number of predictions

    # Filter both labels simultaneously to ensure matching lengths
    filtered_labels = [(pred, gt) for pred, gt in zip(all_pred_labels, all_gt_labels) if
                       pred is not None and gt is not None]

    # If no valid pairs remain, return zeros for metrics
    if not filtered_labels:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1-Score": 0}

    valid_pred_labels, valid_gt_labels = zip(*filtered_labels)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(valid_gt_labels, valid_pred_labels)
    precision = precision_score(valid_gt_labels, valid_pred_labels, average='macro', zero_division=0)
    recall = recall_score(valid_gt_labels, valid_pred_labels, average='macro', zero_division=0)
    f1 = f1_score(valid_gt_labels, valid_pred_labels, average='macro', zero_division=0)
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


ground_truth_labels = [0, 11, 14, 6, 8, 7, 14]
predicted_labels = [4.0, 7.0, 5.0, 7.0, 0.0, 5.0, 9.0]
predicted_boxes = [[42, 4, 870, 306], [711, 303, 857, 431], [847, 144, 914, 166], [865, 65, 934, 137], [992, 6, 1061, 84], [846, 164, 911, 176], [1012, 18, 1046, 49]]
ground_truth_boxes = [[59, 0, 845, 313], [994, 7, 1054, 86], [1014, 15, 1034, 50], [847, 158, 908, 168], [1014, 22, 1034, 37], [948, 3, 957, 13], [943, 3, 960, 55]]


metrics = calculate_metrics(ground_truth_labels, predicted_labels, predicted_boxes, ground_truth_boxes)
print(metrics)
