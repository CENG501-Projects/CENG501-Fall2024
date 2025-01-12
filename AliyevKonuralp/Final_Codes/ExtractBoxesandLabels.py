import json
import numpy as np
from PIL import Image


def extract_bboxes_from_instanceIds(instance_path, instance_classes, json_path, class_to_category, category_map):
    
    # Initialize lists to store the results
    boxes = []
    labels = []
    category_labels = []

    # Load the JSON annotations
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Iterate through the objects in the JSON file
    for obj in json_data.get('objects', []):
        # Extract the class label from JSON
        label = obj.get('label', '')

        # Ensure the label exists in the instance_classes mapping
        if label and label in instance_classes:
            class_id = instance_classes[label]

            # Extract the polygon and compute the bounding box
            polygon = obj.get('polygon', [])
            if polygon:
                # Convert polygon points to numpy array
                polygon_np = np.array(polygon)
                x_min, y_min = np.min(polygon_np, axis=0)
                x_max, y_max = np.max(polygon_np, axis=0)

                # Append the bounding box and associated labels
                boxes.append([x_min, y_min, x_max, y_max])  # [xmin, ymin, xmax, ymax]
                category_label = class_to_category.get(label, "void")  # Map to category
                category_labels.append(category_label)
                labels.append(category_map.get(category_label, -1))  # Map to numeric category ID

    # If no boxes were found, return an empty target
    if len(boxes) == 0:
        return {"boxes": np.array([]), "labels": np.array([]), "category_labels": []}

    # Convert the results to the expected format
    target = {
        "boxes": np.array(boxes, dtype=np.float32),  # Ensure consistent dtype
        "labels": np.array(labels, dtype=np.int64),  # Ensure consistent dtype
        "category_labels": category_labels,  # Keep as a list of strings
    }

    return target


def extract_bboxes_from_instanceIdsBdd100k(image_labels, instance_classes, class_to_category,category_map):
    boxes = []
    labels = []
    category_labels = []

    for obj in image_labels:
        if obj['category']  in class_to_category:
            polygon = obj.get('box2d', [])
            if polygon:
                x_min, y_min,x_max,y_max= polygon['x1'], polygon['y1'], polygon['x2'], polygon['y2']
                boxes.append([x_min, y_min, x_max, y_max])
                category_label = class_to_category.get(obj['category'], "void")
                category_labels.append(category_label)
                labels.append(category_map.get(category_label, -1))
                
        else:
            polygon = obj.get('box2d', [])
            if polygon:
                x_min, y_min,x_max,y_max= polygon['x1'], polygon['y1'], polygon['x2'], polygon['y2']
                category_labels.append("void")
                labels.append(category_map.get("void", -1))
    if len(boxes) == 0:
        return {"boxes": np.array([]), "labels": np.array([]), "category_labels": []}
    target = {
        "boxes": np.array(boxes, dtype=np.float32),  # Ensure consistent dtype
        "labels": np.array(labels, dtype=np.int64),  # Ensure consistent dtype
        "category_labels": category_labels,  # Keep as a list of strings
    }

    return target