import cv2
import os
import time
import asyncio
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from RAIT.cameras.recevier import CameraReceiver
from config.config import load_config

class VideoRecorder:
    """
    Class to record an 8-second video at 30 FPS and save frames at 2 FPS,
    while also capturing initial RGB and depth images before each recording.
    """
    def __init__(self, receiver, config):
        """
        Initializes the VideoRecorder with the provided CameraReceiver instance.
        
        Args:
            receiver (CameraReceiver): Instance of CameraReceiver to get frames.
            config (dict): Configuration dictionary for video recording.
        """
        self.config = config
        self.receiver = receiver
        self.output_dir = self.config['data_path']
        self.sample_count = 0
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_sample_directory(self):
        """
        Creates a new sample directory structure for storing video and frames.
        """
        self.sample_count += 1
        sample_folder = os.path.join(self.output_dir, f"sample_{self.sample_count}")
        os.makedirs(sample_folder, exist_ok=True)
        os.makedirs(os.path.join(sample_folder, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(sample_folder, "depth"), exist_ok=True)
        os.makedirs(os.path.join(sample_folder, "initial_frames"), exist_ok=True)
        return sample_folder
    
    async def capture_initial_frames(self, sample_folder):
        """
        Captures a single RGB and depth frame and saves them before video recording.
        
        Args:
            sample_folder (str): Path to the current sample folder.
        """
        color_frame, depth_frame = await self.receiver.decode_frames()
        if color_frame is not None:
            cv2.imwrite(os.path.join(sample_folder, "initial_frames", "image_0.png"), color_frame)
            cv2.imshow("Initial RGB Frame", color_frame)
            cv2.waitKey(1)
        if depth_frame is not None:
            np.save(os.path.join(sample_folder, "initial_frames", "image_0.npy"), depth_frame)
        print("Initial frames captured.")
    
    async def record_video(self):
        """
        Records videos based on the number of samples in the configuration.
        """
        for _ in range(self.config['num_samples']):
            sample_folder = self.create_sample_directory()
            await self.capture_initial_frames(sample_folder)
            
            for i in range(5, 0, -1):
                print(f"Waiting for {i} seconds before starting recording...")
                await asyncio.sleep(1)
            
            print("Recording video...")
            video_path = os.path.join(sample_folder, f"video.{self.config['video_format']}")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(video_path, fourcc, self.config['video_fps'], (self.config['height'], self.config['width']))
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < self.config['video_duration']:
                color_frame, depth_frame = await self.receiver.decode_frames()
                
                if color_frame is not None:
                    out.write(color_frame)
                    cv2.imshow("Live RGB Feed", color_frame)
                    cv2.waitKey(1)
                    if frame_count % self.config['processor_fps'] == 0: 
                        frame_filename = f"image_{int(frame_count // self.config['processor_fps'])}.png"
                        cv2.imwrite(os.path.join(sample_folder, "rgb", frame_filename), color_frame)
                
                if depth_frame is not None and frame_count % 15 == 0:
                    np.save(os.path.join(sample_folder, "depth", f"image_{int(frame_count // self.config['processor_fps'])}.npy"), depth_frame)
                
                frame_count += 1
                await asyncio.sleep(1/self.config['video_fps'])
            
            out.release()
            cv2.destroyAllWindows()
            print(f"Video and frames saved in {sample_folder}")
    
    async def display_live_feed(self):
        """
        Displays the live feed from the receiver after recording completes.
        """
        print("Displaying live feed...")
        while True:
            color_frame, _ = await self.receiver.decode_frames()
            if color_frame is not None:
                cv2.imshow("Live RGB Feed", color_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    config = load_config("config/config.yaml")
    receiver = CameraReceiver(config.get('Stream', {}))
    video_config = load_config("config/config.yaml")['Video_Recorder']
    async def main():
        await receiver.connect()
        recorder = VideoRecorder(receiver, video_config)
        await recorder.record_video()
        await recorder.display_live_feed()
    
    asyncio.run(main())