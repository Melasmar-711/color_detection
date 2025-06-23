#!/usr/bin/env python

# ====== PATCH TENSORRT NP.BOOL ISSUE ======
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tensorrt as trt
    # Create wrapper that maintains both dictionary and function interface
    class NpTypeWrapper:
        def __init__(self):
            self.mapping = {
                trt.DataType.FLOAT: np.float32,
                trt.DataType.HALF: np.float16,
                trt.DataType.INT8: np.int8,
                trt.DataType.INT32: np.int32,
                trt.DataType.BOOL: bool,  # Fix: Use bool instead of np.bool
            }
        
        def __call__(self, dtype):
            return self.mapping[dtype]
        
        def __getitem__(self, key):
            return self.mapping[key]
    
    if hasattr(trt, 'nptype'):
        trt.nptype = NpTypeWrapper()
except ImportError:
    pass
# ==========================================

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import time
from geometry_msgs.msg import PolygonStamped, Point32
class YOLOv8ROS:
    def __init__(self):
        rospy.init_node('yolov8_ros_processor')
        
        try:
            # Initialize YOLO model with explicit task
            self.model = YOLO("new.pt", task="detect")
            
            # Warmup GPU with proper error handling
            rospy.loginfo("Warming up GPU...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)  # Black image
            for _ in range(3):
                _ = self.model(dummy_img, verbose=False)  # Disable logging
            
            # ROS CV Bridge
            self.bridge = CvBridge()
            
            # FPS tracking
            self.frame_count = 0
            self.fps = 0
            self.last_time = time.time()
            
            # Subscribe to image topic with large buffer
            self.sub = rospy.Subscriber(
                "/camera/image_raw", 
                Image, 
                self.process_image,
                queue_size=1,
                buff_size=2**24
            )

            self.bbox_pub = rospy.Publisher('/detected_boxes', PolygonStamped, queue_size=10)
            rospy.loginfo("YOLOv8 TensorRT node ready!")
            
        except Exception as e:
            rospy.logerr(f"Initialization failed: {str(e)}")
            raise

    def process_image(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLOv8 inference (disable verbose output)
            #results = self.model(cv_image, verbose=False)
            # Run inference with optimized settings
            results = self.model(
            	   
		    cv_image,
		    #iou=0.45,           # Balanced speed/accuracy
		    conf=0.7,          # Filter weak detections
		    device=0,           # Use GPU 0
		    #agnostic_nms=True   # Faster NMS
		    half=True,
		    verbose=True
		)
            
            if results:  # Check if results exist
                annotated_frame = results[0].plot()

                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                bbox_msg = PolygonStamped()
                bbox_msg.header.stamp = msg.header.stamp  # Match image timestamp
                bbox_msg.header.frame_id = "camera_frame"

                for box in boxes:  # boxes in xyxy format
                 pts = [
			Point32(x=box[0], y=box[1], z=0),
			Point32(x=box[2], y=box[1], z=0),
			Point32(x=box[2], y=box[3], z=0),
			Point32(x=box[0], y=box[3], z=0)
		    ]
                 bbox_msg.polygon.points.extend(pts)
		    
                self.bbox_pub.publish(bbox_msg)          
                

                
                # Calculate FPS (smoother average)
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    now = time.time()
                    self.fps = 10 / (now - self.last_time)
                    self.last_time = now
                
                # Display FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {self.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                
                # Show output
                cv2.imshow("YOLOv8 TensorRT", annotated_frame)
                cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")


   

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down...")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = YOLOv8ROS()
        node.run()
    except Exception as e:
        rospy.logerr(f"Node crashed: {str(e)}")
        raise
