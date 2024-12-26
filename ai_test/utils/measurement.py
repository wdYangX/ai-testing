import numpy as np
import torch
from pathlib import Path
from ultralytics.utils.metrics import box_iou, smooth, compute_ap, plot_mc_curve, plot_pr_curve



def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names=(), eps=1e-16,
                 prefix=''):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    if tp.ndim == 1:
        tp = tp.reshape(-1, 1)  # Ensure shape is (N, 1)

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)

    nc = unique_classes.shape[0]  # number of classes, number of detections
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):

        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        if len(tp) > 1:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
        else:
            fpc = (1 - tp).cumsum(0)  # If tp has only one element, use it directly
            tpc = tp.cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, save_dir / f'{prefix}PR_curve.png', names, on_plot=on_plot)
        plot_mc_curve(px, f1, save_dir / f'{prefix}F1_curve.png', names, ylabel='F1', on_plot=on_plot)
        plot_mc_curve(px, p, save_dir / f'{prefix}P_curve.png', names, ylabel='Precision', on_plot=on_plot)
        plot_mc_curve(px, r, save_dir / f'{prefix}R_curve.png', names, ylabel='Recall', on_plot=on_plot)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def process_iou_in_batches(pred_boxes, gt_boxes, pred_labels, gt_labels, iou_th, batch_size=1000):
    tp_list = []
    for start_idx in range(0, len(pred_boxes), batch_size):
        end_idx = min(start_idx + batch_size, len(pred_boxes))

        # Slice the predictions and ground truths for this batch
        batch_pred_boxes = pred_boxes[start_idx:end_idx]
        batch_pred_labels = pred_labels[start_idx:end_idx]

        # Compute IoU for the current batch
        iou_matrix = box_iou(batch_pred_boxes, gt_boxes)

        # Initialize the true positives for this batch
        tp_batch = np.zeros(len(batch_pred_boxes), dtype=bool)
        iou_threshold = iou_th

        # Find matching predictions and ground truths based on IoU threshold
        match_indices = torch.where(iou_matrix > iou_threshold)

        for pred_idx, gt_idx in zip(match_indices[0], match_indices[1]):
            if batch_pred_labels[pred_idx] == gt_labels[gt_idx]:
                tp_batch[pred_idx] = True

        # Append the true positives for this batch to the list
        tp_list.extend(tp_batch)

    return np.array(tp_list)


def calculate_metrics_by_bboxs(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_th, names):
    """
    Computes the average precision (AP) and mean average precision (mAP) for object detection evaluation
    based on bounding boxes, labels, and IoU threshold.

    Args:
        pred_boxes (torch.Tensor): A tensor of predicted bounding boxes of shape (N, 4), where N is the number of predictions.
            Each box is represented as [xmin, ymin, xmax, ymax].
        pred_scores (torch.Tensor): A tensor of predicted confidence scores of shape (N,), corresponding to each predicted box.
        pred_labels (torch.Tensor): A tensor of predicted labels of shape (N,), where each value represents the class ID of the predicted box.
        gt_boxes (torch.Tensor): A tensor of ground truth bounding boxes of shape (M, 4), where M is the number of ground truth boxes.
            Each box is represented as [xmin, ymin, xmax, ymax].
        gt_labels (torch.Tensor): A tensor of ground truth labels of shape (M,), where each value represents the class ID of the ground truth box.
        iou_th (float): The Intersection over Union (IoU) threshold for determining whether a detection is a true positive or false positive.
        names (list of str): A list of class names corresponding to the label IDs. Used for mapping class IDs to class names.

    Returns:
        tuple: A tuple containing the following values:
            - map50 (float): Mean average precision at IoU threshold 0.5.
            - ap (numpy.ndarray): Average precision for each class at different IoU thresholds, array of shape (C, 1) where C is the number of classes.
            - class_id (numpy.ndarray): An array of unique class IDs that have non-zero average precision values.

    Example:
        pred_boxes = torch.tensor([[50, 50, 200, 200], [30, 30, 180, 180]])
        pred_scores = torch.tensor([0.9, 0.75])
        pred_labels = torch.tensor([0, 1])
        gt_boxes = torch.tensor([[50, 50, 210, 210], [30, 30, 180, 180]])
        gt_labels = torch.tensor([0, 1])

        names = {
            0: "Forklift",
            1: "Hand pallet jack",
            2: "Electric pallet jack",
            3: "Reach truck",
            4: "Truck",
            5: "Pallet",
            6: "Product box",
            7: "Product package",
            8: "Fallen package",
            9: "Person",
            10: "Person wear visible clothes",
            11: "Person using phone",
            12: "Person eating or drinking",
            13: "Person carrying object",
            14: "Person pull object",
            15: "Alcohol testing tool",
            16: "Firefighting equipment",
            17: "Wheel Chocks",
            18: "Beacon light",
            19: "No_beacon light",
            20: "Security person"
        }

        iou_th = 0.5

        map50, ap, class_id = calculate_metrics_by_bboxs(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_th, names)

        # map50: Mean AP at IoU threshold 0.5
        # ap: Array of average precision values per class at different IoU thresholds
        # class_id: Array of class IDs that have non-zero average precision
    """
    tp = process_iou_in_batches(pred_boxes, gt_boxes, pred_labels, gt_labels, iou_th, batch_size=1000)
    tp = np.array([[val] for val in tp])

    # Assuming `ap_per_class` is a predefined function
    tp, fp, p, r, f1, ap, class_id = ap_per_class(tp, pred_scores, pred_labels, gt_labels, names=names)
    map50 = ap[:, 0].mean() if len(ap) else 0.0

    return map50, ap, class_id, p, r, f1


if __name__ == "__main__":
    pass
    # from constant import cfg
    # # # Sample predictions and ground truths
    # predictions = [{'boxes': [[3, 10, 856, 367], [794, 233, 885, 275], [969, 72, 1058, 238], [866, 65, 932, 135], [719, 328, 858, 432], [845, 152, 909, 176], [797, 286, 878, 302], [714, 398, 852, 451], [987, 112, 1031, 146], [867, 130, 929, 141], [794, 273, 878, 291], [890, 41, 941, 50]],
    #                 'scores': [0.92578125, 0.9248046875, 0.916015625, 0.8720703125, 0.8671875, 0.8447265625, 0.8056640625, 0.73876953125, 0.6689453125, 0.64697265625, 0.61669921875, 0.5078125],
    #                 'labels': [4.0, 5.0, 0.0, 7.0, 7.0, 5.0, 5.0, 5.0, 9.0, 5.0, 5.0, 5.0]}][0]
    #
    # ground_truths = [{'boxes': [[724, 343, 847, 429], [868, 134, 924, 143], [715, 433, 835, 454], [0, 41, 189, 135], [891, 45, 936, 53], [969, 70, 1052, 240], [987, 121, 1029, 148], [999, 105, 1017, 121], [595, 12, 794, 372], [799, 261, 874, 278], [800, 286, 874, 303], [800, 275, 874, 291], [0, 213, 520, 630], [0, 124, 617, 444], [844, 166, 908, 178], [969, 70, 1052, 240], [987, 105, 1029, 148], [865, 56, 935, 134]],
    #                   'labels': [5, 6, 6, 13, 6, 11, 8, 7, 0, 6, 6, 6, 13, 13, 6, 1, 14, 5]}][0]
    # iou_th = 0.5
    # pred_boxes = torch.tensor(predictions['boxes'])
    # pred_scores = torch.tensor(predictions['scores'])
    # pred_labels = torch.tensor(ground_truths['labels'])
    # gt_boxes = torch.tensor(ground_truths['boxes'])
    # gt_labels = torch.tensor(predictions['labels'])
    # map50, ap, class_id = calculate_metrics_by_bboxs(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_th, cfg.AI_CONFIG)
    # print(map50, ap, class_id)