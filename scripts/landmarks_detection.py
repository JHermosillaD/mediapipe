#!/usr/bin/env python
import numpy as np
import roslib
import rospy
import time

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from mediapipe import solutions
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

class mediapipe_detection:  

  def __init__(self):
    print('Loading model...', end='')
    start_time = time.time()
          
    self.base_options = python.BaseOptions(model_asset_path='/home/jhermosilla/Downloads/pose_landmarker_lite.task')
    self.options = vision.PoseLandmarkerOptions(base_options=self.base_options, output_segmentation_masks=True)
    self.detector = vision.PoseLandmarker.create_from_options(self.options)  
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Model ready! took {} seconds'.format(elapsed_time))

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.callbackDetection, queue_size=1)
    self.image_pub = rospy.Publisher("/mediapipe/image/compressed", CompressedImage, queue_size=1)

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
  
  def callbackDetection(self,ros_data):
    try:
      colorImage = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
    except CvBridgeError as e:
      print(e)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=colorImage)
    detection_result = self.detector.detect(image)
    annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)

    try:
      self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(annotated_image))
    except CvBridgeError as e:
      print(e)
      
def main():    
  rospy.init_node('yolo_detection', anonymous=True)
  ic = mediapipe_detection()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down...")

if __name__ == '__main__':
    main()
