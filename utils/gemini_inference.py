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
            # Get bounding box and confidence score
            confidence = 100
            
            # Calculate center coordinates
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            
            centers[target_class] = {"center":[center_x, center_y], "bboxes":box, "confidence":confidence}
            print(f"Center for {target_class}: {centers[target_class]}")
        
        self.detection_results = centers
        return centers

    # def visualize_detections(self, 
    #                        image: Image.Image, 
    #                        boxes: Dict, 
    #                        output_dir: str,
    #                        filename: str = 'detection_visualization.jpg') -> Tuple[int, int]:
    #     """
    #     Visualize detected objects with bounding boxes and save the result.
        
    #     Args:
    #         image_path (str): Path to the original image
    #         boxes (Dict): Dictionary of detected objects and their bounding boxes
    #         output_dir (str): Directory to save the visualization
    #         filename (str): Name of the output file
            
    #     Returns:
    #         Tuple[int, int]: Image dimensions (width, height)
    #     """
    #     im = image # Image.open(image_path)
    #     self._plot_bounding_boxes(im, list(boxes.items()))
        
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_path = os.path.join(output_dir, filename)
    #     im.save(output_path)
        
    #     return im.width, im.height
    # @staticmethod
    # def _plot_bounding_boxes(image: Image.Image, 
    #                        noun_phrases_and_positions: List[Tuple]) -> None:
    #     """Plot bounding boxes on the image."""
    #     plot_bounding_boxes(image, noun_phrases_and_positions)

    # def get_real_boxes(self):
    #     if self.boxes is None:
    #         return None
    #     return {i: normalize_box(j) for i, j in self.boxes.items()}


if __name__ == "__main__":
    async def main():
        config = load_config("config/config.yaml")
        camera_config = config.get("Stream", {})
        gemini_config = config.get("Gemini", {})
        camera = CameraReceiver(camera_config)
        gemini = Gemini_Inference(gemini_config)
        target_class = ['red soda can']
        gemini.set_target_classes(target_class)
        recording_dir = gemini_config.get("recording_dir")
        save_path = f"{recording_dir}/{int(time.time())}"
        print(save_path)
        frames = await camera.capture_frames(save_path)
        color_frame_path = frames.get('rgb')
        depth_frame_path = frames.get('depth')
        if color_frame_path and depth_frame_path:
            color_image = Image.open(color_frame_path)
            output = gemini.get_object_center(color_image, target_class[0])
            pixel_center = output.get('center')
            print(f"Pixel Center: {pixel_center}")
        else:
            print("No frames received or saved.")

        
    asyncio.run(main())