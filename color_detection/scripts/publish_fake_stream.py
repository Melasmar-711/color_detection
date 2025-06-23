#!/usr/bin/env python

import rospy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Header

class ImageFolderPublisher:
    def __init__(self):
        rospy.init_node('image_folder_publisher', anonymous=True)
        
        # Get parameters from ROS parameter server
        self.folder_path = rospy.get_param('~folder_path', './sim')
        self.topic_name = rospy.get_param('~topic_name', '/camera/image_raw')
        self.publish_rate = rospy.get_param('~publish_rate', 10)  # Hz
        self.display_duration = rospy.get_param('~display_duration', 3)  # seconds per image
        self.loop = rospy.get_param('~loop', True)
        
        # Expand user path if needed
        self.folder_path = os.path.expanduser(self.folder_path)
        
        # Initialize publisher and CV bridge
        self.image_pub = rospy.Publisher(self.topic_name, Image, queue_size=10)
        self.bridge = CvBridge()
        
        # Get list of image files
        self.image_files = self._get_image_files()
        if not self.image_files:
            rospy.logerr(f"No images found in folder: {self.folder_path}")
            return
            
        rospy.loginfo(f"Found {len(self.image_files)} images to publish")
        rospy.loginfo(f"Publishing each image for {self.display_duration} seconds")
        
    def _get_image_files(self):
        """Get sorted list of image files in the folder"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        files = [f for f in os.listdir(self.folder_path) 
                if os.path.splitext(f)[1].lower() in valid_extensions]
        files.sort()  # Sort alphabetically
        return [os.path.join(self.folder_path, f) for f in files]
        
    def publish_images(self):
        """Main publishing loop"""
        rate = rospy.Rate(self.publish_rate)
        index = 0
        
        while not rospy.is_shutdown() and self.image_files:
            # Load current image
            img_path = self.image_files[index]
            try:
                cv_image = cv2.imread(img_path)
                if cv_image is None:
                    rospy.logwarn(f"Could not read image: {img_path}")
                    continue
                    
                # Convert to ROS message with timestamp
                header = Header()
                header.stamp = rospy.Time.now()  # Critical timestamp addition
                header.frame_id = "camera_frame"
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                ros_image.header = header
                
                # Publish this image for the specified duration
                start_time = rospy.Time.now()
                while (rospy.Time.now() - start_time).to_sec() < self.display_duration:
                    if rospy.is_shutdown():
                        return
                        
                    # Update timestamp for each publication if needed
                    # ros_image.header.stamp = rospy.Time.now()  # Uncomment for unique timestamps
                    self.image_pub.publish(ros_image)
                    rospy.logdebug(f"Publishing: {img_path}")
                    rate.sleep()
                
                rospy.loginfo(f"Published {img_path} for {self.display_duration} seconds")
                
            except Exception as e:
                rospy.logerr(f"Error processing {img_path}: {str(e)}")
                
            # Move to next image
            index += 1
            if index >= len(self.image_files):
                if self.loop:
                    index = 0
                    rospy.loginfo("Looping back to first image")
                else:
                    rospy.loginfo("Finished publishing all images")
                    break

if __name__ == '__main__':
    try:
        publisher = ImageFolderPublisher()
        publisher.publish_images()
    except rospy.ROSInterruptException:
        pass

