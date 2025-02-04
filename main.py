"""
This is the main entry point of the application.
It initializes the required components, captures frames from a camera, processes them to detect objects,
and calculates the real-world coordinates of the detected object.
"""

import numpy as np
import asyncio
import os
import sys
import logging
from PIL import Image
import time
from utils.gemini_inference import Gemini_Inference
from utils.utils import deproject_pixel_to_point, transform_coordinates

# Add project root directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from config.config import load_config
from RAIT.cameras.recevier import CameraReceiver

# Load configuration settings
config = load_config("config/config.yaml")
gemini_config = config.get('Gemini', {})

# Initialize camera receiver and Gemini inference model
camera_receiver = CameraReceiver(config)
gemini = Gemini_Inference(config)

async def main():
    """
    Main function that performs the following steps:
    1. Connects to the camera receiver.
    2. Captures RGB and depth frames.
    3. Identifies the target object in the RGB frame.
    4. Calculates the object's real-world coordinates using depth data.
    5. Transforms the coordinates and logs the results.
    6. Cleans up the camera receiver connection.
    """
    await camera_receiver.connect()
    logging.info("Starting application...")
    
    # Define the target object class
    target_class = ['red soda can']
    gemini.set_target_classes(target_class)
    
    # Define the directory for storing captured frames
    recording_dir = gemini_config.get("recording_dir")
    save_path = f"{recording_dir}/{int(time.time())}"
    print(save_path)
    
    # Capture frames from the camera
    frames = await camera_receiver.capture_frames(save_path)
    color_frame_path = frames.get('rgb')
    depth_frame_path = frames.get('depth')
    
    if color_frame_path and depth_frame_path:
        # Process RGB frame to detect the target object
        color_image = Image.open(color_frame_path)
        output = gemini.get_object_center(color_image, target_class[0])
        pixel_center = output.get('center')
        print(f"Pixel Center: {pixel_center}")
    else:
        print("No frames received or saved.")
    
    if pixel_center:
        # Load depth data and retrieve camera intrinsics
        depth_image = np.load(depth_frame_path)
        intrinsics = camera_receiver._get_intrinsics(location='India', camera_name='D435I')
        print(f"Camera Intrinsics: {intrinsics}")
        
        # Convert pixel coordinates to real-world coordinates
        depth_center = deproject_pixel_to_point(pixel_center, depth_image, intrinsics=intrinsics)
        print(f"Real World Center: {depth_center}")
        
        # Apply coordinate transformation
        transformed_center = transform_coordinates(*depth_center)
        print(f"Transformed Center: {transformed_center}")
    else:
        print("No pixel center found.")
    
    # Cleanup camera receiver resources
    await camera_receiver.cleanup()
    logging.info("Application ended.")

if __name__ == "__main__":
    asyncio.run(main())