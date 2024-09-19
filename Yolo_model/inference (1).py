import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def load_models(person_model_path, ppe_model_path):
    print(f"Loading person model from {person_model_path}")
    print(f"Loading PPE model from {ppe_model_path}")
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)
    return person_model, ppe_model

def perform_inference(image_path, person_model, ppe_model):
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Perform person detection
    person_results = person_model(image)[0]

    # Process each detected person
    for person_box in person_results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, person_box[:4])

        # Crop the person from the image
        person_crop = image[y1:y2, x1:x2]

        # Perform PPE detection on the cropped image
        ppe_results = ppe_model(person_crop)[0]

        # Convert PPE detections back to full image coordinates
        # Convert PPE detections back to full image coordinates
        for ppe_box in ppe_results.boxes.data:
            ppe_x1, ppe_y1, ppe_x2, ppe_y2, conf, cls = map(int, ppe_box.tolist())
            full_ppe_x1 = x1 + ppe_x1
            full_ppe_y1 = y1 + ppe_y1
            full_ppe_x2 = x1 + ppe_x2
            full_ppe_y2 = y1 + ppe_y2

            # Draw PPE bounding box
            cv2.rectangle(original_image, (full_ppe_x1, full_ppe_y1), (full_ppe_x2, full_ppe_y2), (0, 255, 0), 2)

            # Add label and confidence
            label = ppe_model.names[cls]
            label_text = f"{label}"
            cv2.putText(original_image, label_text, (full_ppe_x1, full_ppe_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw person bounding box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(original_image, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return original_image


def process_directory(input_dir, output_dir, person_model, ppe_model):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"inference_{filename}")

            result_image = perform_inference(input_path, person_model, ppe_model)
            if result_image is not None:
                cv2.imwrite(output_path, result_image)
                print(f"Processed: {filename}")
            else:
                print(f"Skipping: {filename} due to errors")

def main():
    parser = argparse.ArgumentParser(description='Run object detection and PPE detection on images.')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, help='Directory to save output images with bounding boxes')
    parser.add_argument('person_det_model', type=str, help='Path to the person detection model')
    parser.add_argument('ppe_detection_model', type=str, help='Path to the PPE detection model')
    args = parser.parse_args()

    # Load models
    person_model, ppe_model = load_models(args.person_det_model, args.ppe_detection_model)

    # Process images
    process_directory(args.input_dir, args.output_dir, person_model, ppe_model)

if __name__ == '__main__':
    main()
