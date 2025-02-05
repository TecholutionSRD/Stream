"""
This file contains the codes for Google's Gemini Model Inference. 
"""
import asyncio
import time
import os
import threading
from typing import List, Dict, Optional, Tuple
from PIL import Image
import json
import json_repair
import numpy as np
import google.generativeai as genai
from pathlib import Path
import cv2
import sys
from PIL import Image
from dotenv import load_dotenv


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config.config import load_config
from RAIT.cameras.recevier import CameraReceiver
from utils.utils_main import deproject_pixel_to_point, transform_coordinates

class Gemini_Inference:
    """
    Gemini Inference is the class used for various inference tasks using Google's Gemini Model.
    Mainly the task is of object detection.
    
    Args:
        api_key (str): API key for accessing the Gemini model.
        recording_dir (Path): Directory containing the video recording.
        inference_mode (bool): Flag to control whether inference should be performed.
        target_classes (List[str]): List of target classes for detection.
    """
    def __init__(self, config, inference_mode: bool = False):
        self.config = config.get('Gemini', {})
        self.configure_gemini(gemini_api_key)
        self.model = genai.GenerativeModel(model_name=self.config["model_name"])
        self.recording_dir = Path(self.config['recording_dir'])
        self.inference_mode = inference_mode
        self.lock = threading.Lock()
        self.detection_results = None
        self.process_results = None
        self.boxes = None
        self.target_classes = []
        self.default_prompt = (
            "Return bounding boxes for objects in the format:\n"
            "```json\n{'<object_name>': [xmin, ymin, xmax, ymax]}\n```\n"
            "Include multiple instances of objects as '<object_name>_1', '<object_name>_2', etc."
        )
        self.detection_prompt = (
            "You are a world-class computer vision expert. Analyze this image carefully and detect "
            "all objects with detailed, specific descriptions. For example, use 'red soda can' instead of just 'can'. "
            "Include color, brand names, or distinctive features when possible. "
            "Return bounding boxes in the following JSON format:\n"
            "{'<detailed_object_name>': [xmin, ymin, xmax, ymax]}\n"
            "For multiple instances of similar objects, append numbers like '<detailed_object_name>_1', '<detailed_object_name>_2'.\n"
            "Focus on accuracy and detail in object descriptions."
        )
        self.capture_state = False
    
    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configure Gemini API."""
        genai.configure(api_key=api_key)

    def set_target_classes(self, target_classes: List[str]) -> None:
        """
        Set the target classes for detection.
        
        Args:
            target_classes (List[str]): List of target classes.
        """
        self.target_classes = target_classes

    def process_frame(self, image: Image.Image):
        """
        Process a single frame for object detection.
        
        Args:
            image (Image.Image): The input image.
        """
        prompt = self.default_prompt
        if self.target_classes:
            prompt += "\nDetect the following classes: " + ", ".join(self.target_classes if self.target_classes else ["everything"])

        response = self.model.generate_content([image, prompt])
        try:
            detection_results = json.loads(json_repair.repair_json(response.text))
        except ValueError as e:
            detection_results = {}
            print(f"Error parsing detection results: {e}")

        with self.lock:
            self.process_results = detection_results
        
    def get_process_frame_results(self) -> Optional[Dict]:
        """
        Get the results of the processed frame.
        
        Returns:
            dict: The processed frame results.
        """
        with self.lock:
            return self.process_results  
        
    def get_object_detection_results(self) -> Optional[Dict]:
        """
        Get the latest object detection results.
        
        Returns:
            dict: The object detection results.
        """
        with self.lock:
            return self.detection_results
    
    def set_inference_state(self, state: bool):
        """
        Enable or disable inference mode.
        
        Args:
            state (bool): True to enable inference, False to disable.
        """
        self.inference_mode = state

    def set_capture_state(self, state:bool):
        """
        Enable or disable capture mode.
        
        Args:
            state (bool): True to enable capture, False to disable.
        """
        self.capture_state = state
    
    def get_object_center(self, image: Image.Image, target_class: str) -> Optional[Dict]:
        """
        Get the center and bounding box of a detected object.
        
        Args:
            image (Image.Image): The input image.
            target_class (str): The target class name.
        
        Returns:
            dict: A dictionary containing the center coordinates, bounding box, and confidence score.
        """

        self.process_frame(image)
        results = self.get_process_frame_results()
        print("-"*100)
        print(results)
        if not results or target_class not in results:
            return None

        box = results[target_class]
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        detection_results = {"center": (center_x, center_y), "box": box, "confidence": 100}
        self.detection_results = detection_results
        return detection_results
    
    def get_object_centers(self, im_folderpath:str = None , im: Image = None, target_classes: List[str] = None) -> Dict[str, Tuple[Optional[int], Optional[int], Optional[np.ndarray], Optional[float]]]:
        """
        Get the centers of the detected objects for the given target classes.
        
        Args:
            im: PIL Image
            target_classes (List[str]): List of object classes to detect
                
        Returns:
            Dict[str, Tuple[Optional[int], Optional[int], Optional[np.ndarray], Optional[float]]]: 
                Dictionary with target class as key and tuple of center coordinates, bounding box, confidence score as value. 
                All None if detection fails for a class.
        """
        centers = {}
        unscaled_boxes = None

        if not im:
            # Open the latest image path for the given folder
            image_files = sorted(Path(f'{im_folderpath}').glob('*.jpg'), key=os.path.getmtime)
            if image_files:
                im = Image.open(image_files[-1])
            else:
                raise FileNotFoundError("No .jpg files found in the specified directory.")

        if target_classes:
            self.set_target_classes(target_classes=target_classes)
        
        self.process_frame(im)
        unscaled_boxes = self.get_process_frame_results()

        self.boxes = unscaled_boxes
        print("visualizing detections")
        self.visualize_detections(im, unscaled_boxes, self.recording_dir)
        boxes = self.get_real_boxes()
        print(f"Boxes: {boxes}")

        for target_class, box in boxes.items():
            confidence = 100
        
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            
            centers[target_class] = {"center":[center_x, center_y], "bboxes":box, "confidence":confidence}
            print(f"Center for {target_class}: {centers[target_class]}")
        
        self.detection_results = centers
        return centers

    async def detect(self, camera, target_class: List[str] = ['red soda can']):
        """
        Processes the captured images to detect an object and calculate its real-world coordinates.
        
        :param camera: Camera instance
        :param target_class: List of target objects to detect
        :return: Transformed real-world coordinates of the detected object
        """
        recording_dir = self.config.get("recording_dir")
        save_path = f"{recording_dir}/{int(time.time())}"
        frames = await camera.capture_frames(save_path)  # Add await here
        color_frame_path = frames.get('rgb')
        depth_frame_path = frames.get('depth')
        intrinsics = camera._get_intrinsics(location='India', camera_name='D435I')
        
        self.set_target_classes(target_class)
        color_image = Image.open(color_frame_path)
        print("Gemini Inference: Processing frame...")
        output = self.get_object_center(color_image, target_class[0])
        pixel_center = output.get('center')
        if not pixel_center:
            print("No object detected.")
            return None
            
        depth_image = np.load(depth_frame_path)
        depth_center = deproject_pixel_to_point(pixel_center, depth_image, intrinsics=intrinsics)
        transformed_center = transform_coordinates(*depth_center)
        return transformed_center

    def detect_objects(self, rgb_frame: Image.Image) -> List[str]:
        """
        Run detection on a single RGB frame and return detected object names.
        
        Args:
            rgb_frame (Image.Image): RGB frame as PIL Image
            
        Returns:
            List[str]: List of detected object names
        """
        # Use detection_prompt instead of default_prompt
        prompt = self.detection_prompt
        if self.target_classes:
            prompt += "\nDetect the following classes: " + ", ".join(self.target_classes)
            
        response = self.model.generate_content([rgb_frame, prompt])
        try:
            results = json.loads(json_repair.repair_json(response.text))
        except ValueError as e:
            print(f"Error parsing detection results: {e}")
            return []

        with self.lock:
            self.process_results = results
            
        if not results:
            return []
        
        object_names = []
        for key in results.keys():
            base_name = key.rsplit('_', 1)[0]  
            if base_name not in object_names:
                object_names.append(base_name)
                
        return object_names


if __name__ == "__main__":
    async def main():
        config = load_config("config/config.yaml")
        gemini_config = config.get("Gemini", {})
        camera = CameraReceiver(config)
        gemini = Gemini_Inference(config)
        recording_dir = gemini_config.get("recording_dir")
        save_path = f"{recording_dir}/{int(time.time())}"
        print(save_path)
        
        frames = await camera.capture_frames(save_path)
        color_frame_path = frames.get('rgb')
        
        if color_frame_path:
            color_image = Image.open(color_frame_path)
            detected_objects = gemini.detect_objects(color_image)
            print(f"Detected objects: {detected_objects}")
        else:
            print("No frames received or saved.")

    asyncio.run(main())
