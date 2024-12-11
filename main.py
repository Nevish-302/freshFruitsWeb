import os
import cv2
import numpy as np
import json
import logging
from datetime import datetime
import pandas as pd
import re
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
from collections import defaultdict
import torch
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask App
app = Flask(__name__, template_folder='template')
CORS(app)

# Logger for error handling
logger = logging.getLogger("Integrated_Detection")
logging.basicConfig(level=logging.INFO)

# Ensure necessary directories exist
os.makedirs("outputs", exist_ok=True)


class IntegratedDetection:
    def __init__(self):
        # Define model paths with absolute paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")

        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)

        # Define model paths
        self.brand_model_path = ("my_model.keras")
        self.expiry_model_path = ("date_detection_model1.keras")
        self.yolo_path = ("Item_count_yolov8.pt")

        # Verify model files exist
        if not os.path.exists(self.brand_model_path):
            raise FileNotFoundError(f"Brand model not found at: {self.brand_model_path}")
        if not os.path.exists(self.expiry_model_path):
            raise FileNotFoundError(f"Expiry model not found at: {self.expiry_model_path}")
        if not os.path.exists(self.yolo_path):
            raise FileNotFoundError(f"YOLO model not found at: {self.yolo_path}")

        # Load YOLO model for object detection
        self.yolo_model = YOLO(self.yolo_path)

        # Load Detectron2 instance segmentation model
        try:
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            self.cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
            self.detectron_predictor = DefaultPredictor(self.cfg)
            logger.info("Detectron2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Detectron2 model: {e}")
            raise

        # Load Keras models WITHOUT disabling eager execution
        try:
            self.brand_model = tf.keras.models.load_model(self.brand_model_path, compile=False)
            self.expiry_model = tf.keras.models.load_model(self.expiry_model_path, compile=False)
            logger.info("Keras models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Keras models: {e}")
            raise

        # Load Qwen2 VL model
        try:
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            self.qwen_model.eval()
            logger.info("Qwen2 VL model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen2 VL model: {e}")

        # Load class indices
        class_indices_path = os.path.join(self.base_dir, "class_indices.json")
        if not os.path.exists(class_indices_path):
            raise FileNotFoundError(f"Class indices file not found at: {class_indices_path}")

        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        self.class_indices_reversed = {v: k for k, v in self.class_indices.items()}

        # Add frame processing parameters
        self.frame_skip = 2  # Process every nth frame
        self.process_width = 640  # Reduced processing width
        self.process_height = 480  # Reduced processing height
        self.frame_count = 0
        self.current_brand = "Unknown"  # Add this line to store current brand

    def preprocess_image_for_model(self, image, target_size=(224, 224)):
        # Optimize preprocessing by avoiding redundant conversions
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(image.astype("float32") / 255.0, axis=0)

    def detect_brand(self, image):
        try:
            preprocessed_image = self.preprocess_image_for_model(image)
            # Use numpy instead of direct predict
            predictions = self.brand_model(preprocessed_image, training=False).numpy()
            predicted_class = np.argmax(predictions, axis=1)[0]
            return self.class_indices_reversed.get(predicted_class, "Unknown")
        except Exception as e:
            logger.error(f"Error in brand detection: {e}")
            return "Unknown"

    def extract_dates_from_text(self, text):
        date_patterns = [
            r'\b\d{2}/\d{2}/\d{4}\b', r'\b\d{4}/\d{2}/\d{2}\b',
            r'\b\d{2}-\d{2}-\d{4}\b', r'\b\d{4}-\d{2}-\d{2}\b',
        ]
        detected_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            detected_dates.extend(matches)
        return detected_dates

    def save_to_excel(self, data, file_path):
        try:
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
            else:
                df = pd.DataFrame(columns=data[0].keys())
            new_df = pd.DataFrame(data)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_excel(file_path, index=False)
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

    def process_frame(self, frame):
        """
        Optimized frame processing with multiple detection methods
        """
        # Resize frame for faster processing
        frame = cv2.resize(frame, (self.process_width, self.process_height),
                           interpolation=cv2.INTER_AREA)

        object_counts = defaultdict(int)
        combined_detections = []
        output_data = []

        # Detect brand only every 5th frame to reduce load
        if self.frame_count % 5 == 0:
            try:
                self.current_brand = self.detect_brand(frame)
            except Exception as e:
                logger.error(f"Error in brand detection: {e}")
                # Keep previous brand if detection fails
                pass

        # YOLO detection
        try:
            yolo_results = self.yolo_model(frame)
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = self.yolo_model.names[class_id]
                    conf = box.conf[0]

                    combined_detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "confidence": conf,
                        "source": "YOLO"
                    })
                    object_counts[f"{class_name} (YOLO)"] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")

        # Detectron2 detection
        try:
            detectron_results = self.detectron_predictor(frame)
            instances = detectron_results["instances"]

            if len(instances) > 0:
                masks = instances.pred_masks.cpu().numpy()
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                scores = instances.scores.cpu().numpy()

                for i, box in enumerate(boxes):
                    if scores[i] < 0.5:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    mask = masks[i]
                    class_name = f"Class_{classes[i]}"

                    combined_detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "confidence": scores[i],
                        "source": "Detectron2"
                    })
                    object_counts[f"{class_name} (Detectron2)"] += 1

                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    colored_mask[mask] = (0, 0, 255)
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_name} {scores[i]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            logger.error(f"Error in Detectron2 detection: {e}")

        # Always use the stored brand name
        try:
            cv2.putText(frame, f"Brand: {self.current_brand}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            logger.error(f"Error drawing brand text: {e}")

        # Add total count to frame
        try:
            y_offset = 90
            for obj_name, count in object_counts.items():
                cv2.putText(frame, f"{obj_name}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        except Exception as e:
            logger.error(f"Error drawing object counts: {e}")

        # Prepare output data
        output_data.append({
            "Timestamp": datetime.now(),
            "Brand": self.current_brand,
            "Object_Counts": dict(object_counts)
        })

        # Update Excel less frequently
        if self.frame_count % 30 == 0:  # Update every 30 frames
            try:
                self.save_to_excel(output_data, "outputs/results.xlsx")
            except Exception as e:
                logger.error(f"Error saving to Excel: {e}")

        self.frame_count += 1
        return frame, object_counts


# Initialize IntegratedDetection class
detection_system = IntegratedDetection()

# Global variable to store object counts
current_object_counts = defaultdict(int)


def generate_frames():
    """
    Optimized frame capture and processing
    """
    global current_object_counts
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        logger.error("Failed to open video capture")
        return

    # Optimize video capture settings
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0

    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                logger.error("Failed to read frame")
                break

            try:
                # Process every nth frame
                if frame_count % detection_system.frame_skip == 0:
                    processed_frame, object_counts = detection_system.process_frame(frame)
                    current_object_counts = object_counts
                else:
                    processed_frame = frame

                frame_count += 1

                # Encode frame with optimized parameters
                _, buffer = cv2.imencode(".jpg", processed_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
    finally:
        video_capture.release()


@app.route('/')
def index():
    """
    Renders the HTML frontend.
    """
    return render_template("index.html")


@app.route('/object_summary')
def object_summary():
    """
    Returns the current object counts as JSON for the frontend.
    """
    return jsonify(current_object_counts)


@app.route('/video_feed')
def video_feed():
    """
    Streams processed video frames.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)