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
from utils.video_recorder import VideoRecorder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config.config import load_config
from RAIT.cameras.recevier import CameraReceiver
#------------------------------------------------------------------------------------------------------------#
config = load_config("config/config.yaml")
gemini_config = config.get('Gemini', {})
camera_receiver = CameraReceiver(config)
gemini = Gemini_Inference(config)
# video_recorder = VideoRecorder(config)
#------------------------------------------------------------------------------------------------------------#
def initialize_camera():
    """
    Connects to the camera receiver and prepares it for capturing frames.
    """
    return camera_receiver.connect()
#------------------------------------------------------------------------------------------------------------#
async def main():
    """
    Main function to initialize the camera and process frames.
    """
    await initialize_camera()
    logging.info("Camera initialized. Starting application...")

    #

    await camera_receiver.cleanup()
    logging.info("Application ended.")

#------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    asyncio.run(main())