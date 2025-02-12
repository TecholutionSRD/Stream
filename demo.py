# import cv2
# import pyrealsense2 as rs
# import numpy as np

# def display_realsense_video():
#     # Set up the RealSense pipeline
#     pipeline = rs.pipeline()
#     config = rs.config()
    
#     # Configure the pipeline to stream color and depth data
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
#     # Start the pipeline
#     pipeline.start(config)
    
#     try:
#         while True:
#             # Wait for a new set of frames from the camera
#             frames = pipeline.wait_for_frames()
            
#             # Get color and depth frames
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
            
#             # Convert images to numpy arrays
#             color_image = np.asanyarray(color_frame.get_data())
#             depth_image = np.asanyarray(depth_frame.get_data())
            
#             # Display the color image
#             cv2.imshow('RealSense Color Frame', color_image)
            
#             # Optional: You can display the depth image as well
#             # cv2.imshow('RealSense Depth Frame', depth_image)
            
#             # Break the loop if the user presses the 'q' key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Stop the pipeline and close OpenCV windows
#         pipeline.stop()
#         cv2.destroyAllWindows()

# # Call the function to display video
# display_realsense_video()

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Paths to the images
color_frame_path = "/home/shreyas/Desktop/Stream/Realtime-WebRTC/data/captured__frames/1739379914/rgb/image_0.jpg"
depth_frame_path = "/home/shreyas/Desktop/Stream/Realtime-WebRTC/data/captured__frames/1739379914/depth/image_0.npy"

# Load the color image
color_image = cv2.imread(color_frame_path)

# Check if the color image is loaded properly
if color_image is None:
    print(f"Failed to load color image from {color_frame_path}")
else:
    print(f"Color image loaded successfully with shape: {color_image.shape}")

# Load the depth image
depth_image = np.load(depth_frame_path)

# Check if the depth image is loaded properly
if depth_image is None:
    print(f"Failed to load depth image from {depth_frame_path}")
else:
    print(f"Depth image loaded successfully with shape: {depth_image.shape}")

# Pixel coordinates to check
pixel_x, pixel_y = 314, 495

# Bounding box coordinates (top-left, bottom-right)
bbox_x1, bbox_y1 = 212, 458  # top-left corner of bbox
bbox_x2, bbox_y2 = 419, 528  # bottom-right corner of bbox

# Get the dimensions of the images
color_height, color_width = color_image.shape[:2]  # Color image has 3 channels, but we only need the height and width
depth_height, depth_width = depth_image.shape

# Check if the pixel is within the valid range for both images
if 0 <= pixel_x < color_width and 0 <= pixel_y < color_height:
    print(f"Pixel ({pixel_x}, {pixel_y}) is within the color image range.")
else:
    print(f"Pixel ({pixel_x}, {pixel_y}) is out of bounds for the color image.")

if 0 <= pixel_x < depth_width and 0 <= pixel_y < depth_height:
    print(f"Pixel ({pixel_x}, {pixel_y}) is within the depth image range.")
else:
    print(f"Pixel ({pixel_x}, {pixel_y}) is out of bounds for the depth image.")

# Plot the color image with the pixel and bounding box marked
fig, ax = plt.subplots(figsize=(8, 6))

# Convert the BGR color image to RGB for displaying with matplotlib
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# Plot the color image
ax.imshow(color_image_rgb)

# Mark the point on the image
ax.plot(pixel_x, pixel_y, 'ro', label=f'Pixel ({pixel_x}, {pixel_y})')

# Add a label for the point
ax.text(pixel_x + 5, pixel_y - 5, f'({pixel_x}, {pixel_y})', color='white', fontsize=12)

# Draw the bounding box (top-left and bottom-right)
rect = plt.Rectangle((bbox_x1, bbox_y1), bbox_x2 - bbox_x1, bbox_y2 - bbox_y1,
                     linewidth=2, edgecolor='blue', facecolor='none', label='Bounding Box')
ax.add_patch(rect)

# Add title and legend
ax.set_title("Color Image with Pixel and Bounding Box")
ax.legend()

# Save the plotted image to a file
plt.savefig('/home/shreyas/Desktop/Stream/Realtime-WebRTC/data/captured__frames/1739379914/rgb/image_with_pixel_and_bbox.png')

# Display the plot
plt.show()

# Save the depth image with the pixel and bounding box marked
depth_image_with_point_and_bbox = cv2.cvtColor(depth_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Mark the point on the depth image (Red color)
if 0 <= pixel_x < depth_width and 0 <= pixel_y < depth_height:
    depth_image_with_point_and_bbox = cv2.circle(depth_image_with_point_and_bbox, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

# Draw the bounding box on the depth image
cv2.rectangle(depth_image_with_point_and_bbox, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 0), 2)

# Save the depth image with the point and bounding box marked
cv2.imwrite('/home/shreyas/Desktop/Stream/Realtime-WebRTC/data/captured__frames/1739379914/depth/image_with_pixel_and_bbox.png', depth_image_with_point_and_bbox)

print("Images saved with marked pixel and bounding box.")


