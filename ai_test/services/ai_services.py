import os
import zipfile
import torch
import numpy as np
import requests
import tempfile

from constant import cfg
from utils.measurement import calculate_metrics_by_bboxs

session = requests.session()


def validation_model_api(zip_file):
    headers = {"accept": "application/json", }
    files = {"file": (zip_file.name, zip_file.getvalue(), "application/x-zip-compressed"), }
    data = {"model_name": "yolov8m_warehouse_trt", "confidence_threshold": "0.5", "iou_threshold": "0.5", }
    response = session.post(cfg.API_ENDPOINT, headers=headers, files=files, data=data)
    return response.json()


def format_inp(data):
    new_data = {"image_name": [], "scores": [], "boxes": [], "predict": [], 'ground_truth': [], 'accuracy': [],
                'precision': [], 'recall': [], 'f1-Score': [], "g_boxes": []}
    for image_data in data:
        new_data['image_name'] += [image_data['image_name']]
        new_data['scores'] += [image_data['scores']]
        new_data['boxes'] += [image_data['boxes']]
        new_data['g_boxes'] += [image_data['g_boxes']]
        new_data['f1-Score'] += [image_data['f1-Score']]
        new_data['predict'] += [image_data['labels']]
        new_data['ground_truth'] += [image_data['ground_truth']]
        new_data['accuracy'] += [image_data['accuracy']]
        new_data['precision'] += [image_data['precision']]
        new_data['recall'] += [image_data['recall']]
    return new_data


def extract_and_convert_labels(zip_file_path):
    # Create a temporary directory to extract the files
    file_name = zip_file_path.name.split(".")[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the ZIP file into the temporary directory
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                print(f"Extracted files from {zip_file_path} to {temp_dir}")
                # List the files in the extracted folder
                file_list = os.listdir(os.path.join(temp_dir, file_name))
                print("Files in extracted folder:", file_list)

        except Exception as e:
            print(f"Error extracting {zip_file_path}: {e}")
            return None
        # Check for .txt files in the temporary directory
        label_files = [f for f in file_list if f.endswith('.txt')]
        if len(label_files) == 0:
            print("Error: No .txt label files found in the extracted files.")
            return None

        converted_labels = {}  # Dictionary to store converted labels
        # Convert labels in each file
        for label_file in label_files:
            file_path = os.path.join(temp_dir, file_name, label_file)
            try:
                with open(file_path, 'r') as file:
                    labels = {"labels": [], "boxes": []}
                    for line in file.readlines():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_min = int(float(parts[1]))
                        y_min = int(float(parts[2]))
                        x_max = int(float(parts[3]))
                        y_max = int(float(parts[4]))
                        labels['labels'] += [class_id]
                        labels['boxes'] += [[x_min, y_min, x_max, y_max]]
                # Store the converted labels in the dictionary
                converted_labels[label_file] = labels
            except Exception as e:
                print(f"Error reading {label_file}: {e}")

    return converted_labels


def process_label(zip_file):
    return extract_and_convert_labels(zip_file)


def avg_metric(metric):
    metric = np.array(metric)
    return round(np.average(metric), 2)


def remap_labels(labels, mapping):
    """Remap labels according to the provided mapping dictionary."""
    return [mapping[label] if label in mapping else label for label in labels]


def remove_duplicates(data):
  """
  Removes duplicate dictionaries from a list of dictionaries based on 'image_name'.

  Args:
    data: A list of dictionaries, where each dictionary has keys like 'image_name', 'boxes', 'scores', 'labels'.

  Returns:
    A new list of dictionaries with duplicates removed.
  """
  seen = set()
  result = []
  for item in data:
    if item['image_name'] not in seen:
      seen.add(item['image_name'])
      result.append(item)
  return result


def model_metrics(data_file, label_file):
    predictions = validation_model_api(data_file)
    predictions = remove_duplicates(predictions)
    labels = process_label(label_file)
    avg_accuracy = []
    avg_precision = []
    avg_recall = []
    avg_f1_score = []
    avg_map50 = []
    for predict in predictions:
        name = predict['image_name']
        label = labels[name.replace("jpg", 'txt')]["labels"]
        label = remap_labels(label, cfg.DATA_LABEL_MAPPING)
        predict_lb = predict["labels"]
        predict['ground_truth'] = label
        predict['g_boxes'] = labels[name.replace("jpg", 'txt')]["boxes"]
        pred_boxes = torch.tensor(predict['boxes'])
        pred_scores = torch.tensor(predict['scores'])
        pred_labels = torch.tensor(predict_lb)
        gt_boxes = torch.tensor(predict['g_boxes'])
        gt_labels = torch.tensor(label)
        iou_th = cfg.IOU_THRESHOLD
        names = cfg.AI_CONFIG
        map50, ap, class_id, p, r, f1 = calculate_metrics_by_bboxs(pred_boxes, pred_scores, pred_labels, gt_boxes,
                                                                   gt_labels, iou_th, names)
        metric = {"accuracy": ap.mean(), "precision": p.mean(), 'recall': r.mean(), 'f1-Score': f1.mean()}
        predict.update(metric)
        avg_accuracy.append(metric['accuracy'])
        avg_precision.append(metric['precision'])
        avg_recall.append(metric['recall'])
        avg_f1_score.append(metric['f1-Score'])
        avg_map50.append(map50)

    return format_inp(predictions), (
        avg_metric(avg_accuracy), avg_metric(avg_precision), avg_metric(avg_recall), avg_metric(avg_f1_score),
        avg_metric(avg_map50))
