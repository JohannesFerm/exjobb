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
        self.lowerLimitElbow = 100
        self.upperLimitElbow = 180

        #Mediapipe setup
        self.mpPose = mp.solutions.pose
        self.mpHands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        #Setup for intel camera
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.profile = self.pipe.start()

        #Publisher
        self.publisher = self.create_publisher(String, 'humanpose', 10)
        self.message = String()

        self.hand1open = False
        self.hand2open = False

    #Function to get angle from three points
    def angle(self, p1, p2, p3):
        a = atan2(p3[1] - p2[1], p3[0] - p2[0]) - atan2(p1[1] - p2[1], p1[0] - p2[0])
        a = np.rad2deg(a)
        a = abs(a)
        if a > 180:
            a = 360 - a
        return a

    #Function to get current gesture
    def getGesture(self, laa, raa, lea, rea, xvals, openhand):

        pointingLeft = (self.lowerLimitArmpit < laa < self.upperLimitArmpit and self.lowerLimitElbow < lea < self.upperLimitElbow) and not (self.lowerLimitArmpit < raa < self.upperLimitArmpit) and (xvals[2] > xvals[0] + 0.2)
        pointingRight = (self.lowerLimitArmpit < raa < self.upperLimitArmpit and self.lowerLimitElbow < rea < self.upperLimitElbow) and not (self.lowerLimitArmpit < laa < self.upperLimitArmpit) and (xvals[1] > xvals[3] + 0.2)
    
        if pointingLeft:
            return "left"
        elif pointingRight:
            return "right"
        elif openhand is not None:
            if openhand and not (pointingLeft or pointingRight):
                return "openhand"
            elif not openhand and not (pointingLeft or pointingRight):
                return "closedhand"
        else:
            return None
    
    #Function that checks if hand is open or not
    def isOpen(self, handLandmarks):
        
        #IDs for finger tips
        tips = [
            self.mpHands.HandLandmark.THUMB_TIP,
            self.mpHands.HandLandmark.INDEX_FINGER_TIP,
            self.mpHands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mpHands.HandLandmark.RING_FINGER_TIP,
            self.mpHands.HandLandmark.PINKY_TIP
        ]
        
        #IDs for mcps
        mcp = [
            self.mpHands.HandLandmark.THUMB_MCP,
            self.mpHands.HandLandmark.INDEX_FINGER_MCP,
            self.mpHands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mpHands.HandLandmark.RING_FINGER_MCP,
            self.mpHands.HandLandmark.PINKY_MCP
        ]
        
        #Wrist ID
        wristID = self.mpHands.HandLandmark.WRIST

        #Wrist position
        wrist = np.array([
            handLandmarks.landmark[wristID].x,
            handLandmarks.landmark[wristID].y,
            handLandmarks.landmark[wristID].z
        ])

        extendedFingers = 0

        #Loop through all tips and mcp
        for tip, mcp in zip(tips, mcp):
            tipPos = np.array([
                handLandmarks.landmark[tip].x,
                handLandmarks.landmark[tip].y,
                handLandmarks.landmark[tip].z
            ])
            
            mcpPos = np.array([
                handLandmarks.landmark[mcp].x,
                handLandmarks.landmark[mcp].y,
                handLandmarks.landmark[mcp].z
            ])
            
            #Get the vector from tips to wrist and from mcps to wrist
            wristToTip = tipPos - wrist
            wristToMcp = mcpPos - wrist
            
            #If the distance from wrist to tip is greater the finger is extended
            if np.linalg.norm(wristToTip) > np.linalg.norm(wristToMcp) + 0.05:
                extendedFingers += 1

        return extendedFingers >= 4 #Hand is open if 4 or more fingers are extended

    #Runs the node
    def run(self):
        with self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose, \
        self.mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while rclpy.ok():

                handOpen = None
                if cv2.waitKey(1) == ord('q'):
                    break
                
                #Collect frames
                frames = self.pipe.wait_for_frames()
                cFrame = frames.get_color_frame()
                cImage = np.asanyarray(cFrame.get_data())
                cImage = cv2.cvtColor(cImage, cv2.COLOR_RGB2BGR)

                #Get mediapipe results for pose and hand
                mpRes = pose.process(cImage)
                handRes = hands.process(cImage)

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
                                         landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].y,
                                         landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].z])
                rightShoulder = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                          landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                          landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].z])
                leftElbow = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                                      landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].y,
                                      landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value].z])
                rightElbow = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].x,
                                       landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].y,
                                       landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value].z])
                leftWrist = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].x,
                                      landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].y,
                                      landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value].z])
                rightWrist = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].x,
                                       landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].y,
                                       landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value].z])
                leftHip = np.array([landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].y,
                                    landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].z])
                rightHip = np.array([landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].x,
                                     landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].y,
                                     landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value].z])

                #Calculate four necessary angles
                leftArmpitAngle = self.angle(leftHip, leftShoulder, leftElbow)
                rightArmpitAngle = self.angle(rightHip, rightShoulder, rightElbow)
                leftElbowAngle = self.angle(leftShoulder, leftElbow, leftWrist)
                rightElbowAngle = self.angle(rightShoulder, rightElbow, rightWrist)

                #Get hand landmarks, find the right hand, which becomes the left irl
                if handRes.multi_hand_landmarks:
                    rightHand = None

                    for i, handLandmarks in enumerate(handRes.multi_hand_landmarks):
                        handedness = handRes.multi_handedness[i].classification[0].label
                        
                        if handedness == "Right":
                            rightHand = handLandmarks
                            break

                    if rightHand:
                        self.mp_drawing.draw_landmarks(cImage, rightHand, self.mpHands.HAND_CONNECTIONS)
                        handOpen = self.isOpen(rightHand)
                            
                gesture = self.getGesture(leftArmpitAngle, rightArmpitAngle, leftElbowAngle, rightElbowAngle,
                                     [leftShoulder[0], rightShoulder[0], leftWrist[0], rightWrist[0], leftElbow[0], rightElbow[0], leftHip[0], rightHip[0]],
                                     handOpen)

                #Run motor depending on the pointing direction
                if gesture is not None:
                    if gesture == "left":
                        self.message.data = "left"
                    elif gesture == "right":
                        self.message.data = "right"
                    elif gesture == "openhand":
                        self.message.data = "backward"
                    elif gesture == "closedhand":
                        self.message.data = "forward"
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
