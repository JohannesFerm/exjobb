import rclpy
from rclpy.node import Node
import cv2
import time
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
from math import atan2
from std_msgs.msg import String

class PoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        #Define angle limits
        self.lowerLimitArmpit = 50
        self.upperLimitArmpit = 130
        self.lowerLimitElbow = 130
        self.upperLimitElbow = 180

        #Mediapipe setup
        self.mpPose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        #Setup for intel camera
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.profile = self.pipe.start()

        #Publisher
        self.publisher = self.create_publisher(String, 'humanpose', 10)
        self.message = String()

    def angle(self, p1, p2, p3):
        a = atan2(p3[1] - p2[1], p3[0] - p2[0]) - atan2(p1[1] - p2[1], p1[0] - p2[0])
        a = np.rad2deg(a)
        a = abs(a)
        if a > 180:
            a = 360 - a
        return a

    def pointingDirection(self, laa, raa, lea, rea):
        if self.lowerLimitArmpit < laa < self.upperLimitArmpit and self.lowerLimitElbow < lea < self.upperLimitElbow and not self.lowerLimitArmpit < raa < self.upperLimitArmpit:
            return "left"
        elif self.lowerLimitArmpit < raa < self.upperLimitArmpit and self.lowerLimitElbow < rea < self.upperLimitElbow and not self.lowerLimitArmpit < laa < self.upperLimitArmpit:
            return "right"
        else:
            return None

    def run(self):
        with self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
            while rclpy.ok():

                if cv2.waitKey(1) == ord('q'):
                    break
        
                frames = self.pipe.wait_for_frames()
                cFrame = frames.get_color_frame()
                cImage = np.asanyarray(cFrame.get_data())
                cImage = cv2.cvtColor(cImage, cv2.COLOR_RGB2BGR)

                mpRes = pose.process(cImage)

                try:
                    landmarks = mpRes.pose_landmarks.landmark
                except:
                    continue

                self.mp_drawing.draw_landmarks(cImage, mpRes.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                           circle_radius=2),
                                               self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                           circle_radius=2)
                                               )

                #Get mediapipe landmarks
                leftShoulder = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                                         landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].y])
                rightShoulder = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                          landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].y])
                leftElbow = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                                      landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].y])
                rightElbow = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].x,
                                       landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].y])
                leftWrist = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].x,
                                      landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].y])
                rightWrist = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].x,
                                       landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].y])
                leftHip = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].y])
                rightHip = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].x,
                                     landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].y])
                
                #Calculate four necessary angles
                leftArmpitAngle = self.angle(leftHip, leftShoulder, leftElbow)
                rightArmpitAngle = self.angle(rightHip, rightShoulder, rightElbow)
                leftElbowAngle = self.angle(leftShoulder, leftElbow, leftWrist)
                rightElbowAngle = self.angle(rightShoulder, rightElbow, rightWrist)

                pointingDir = self.pointingDirection(leftArmpitAngle, rightArmpitAngle, leftElbowAngle, rightElbowAngle)

                #Run motor depending on the pointing direction
                if pointingDir is not None:
                    if pointingDir == "left":
                        self.message.data = "left"
                    else:
                        self.message.data = "right"

                else:
                    self.message.data = ""
                
                self.publisher.publish(self.message)
                cv2.imshow('Mediapipe Feed', cImage)

        self.pipe.stop()

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()
    node.run()
    node.destroy_node()
    rclpy.shutdown()
