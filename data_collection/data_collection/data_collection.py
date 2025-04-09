import rclpy
from rclpy.node import Node
import time
from flask import Flask, render_template, jsonify
from std_msgs.msg import String
import threading
import pyaudio
import wave
import os
import numpy as np
from hqv_public_interface.msg import MowerImu
import datetime 
import csv


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('DataCollectorNode')
        self.recording = False
        self.audioStream = None
        self.audioFile = None

        self.imuSub = self.create_subscription(MowerImu, '/hqv_mower/imu0/orientation', self.IMUCallback, 10)
        self.imuFile = None
        self.orientation = np.zeros(3)
        self.lastCallbackTime = time.time()
        self.sampleFolder = None

        #Setup flask server stuff
        self.app = Flask(__name__, template_folder="templates")

        @self.app.route("/")
        def home():
            return render_template("index.html")

        #For recording audio samples
        @self.app.route("/label/<label>", methods = ['GET'])
        def readLabel(label):
            #Start a new thread for the recording
            if not self.recording:
                self.recording = True
                thread = threading.Thread(target=self.record, args=(label,))
                thread.start()
            return jsonify({"status": f"Recording {label}"})

        #For stopping the audio recording
        @self.app.route("/stop", methods=['GET'])
        def stop_recording():
            self.recording = False 
            return jsonify({"status": "Recording stopped"})
        
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()

    def IMUCallback(self, msg):
        currentTime = time.time()

        
        if currentTime - self.lastCallbackTime >= 1.0:
            self.lastCallbackTime = currentTime 
            self.orientation = np.array([msg.roll, msg.pitch, msg.yaw])

            if self.imuFile:
                self.imuFile.write(f"{currentTime},{self.orientation[0]},{self.orientation[1]},{self.orientation[2]}\n")


    def run_flask(self):
        self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    def record(self, label):
        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512

        sampleId = f"sample_{int(time.time())}"
        self.sampleFolder = f"dataset/{label}/{sampleId}"
        os.makedirs(self.sampleFolder, exist_ok=True)

        audioPath = os.path.join(self.sampleFolder, f"audio{time.time()}.wav")
        imuPath = os.path.join(self.sampleFolder, f"imu{time.time()}.csv")

        self.audioFile = wave.open(audioPath, 'wb')
        self.audioFile.setnchannels(CHANNELS)
        self.audioFile.setsampwidth(audio.get_sample_size(FORMAT))
        self.audioFile.setframerate(RATE)

        self.imuFile = open(imuPath, 'w', newline='')
        imuWriter = csv.writer(self.imuFile)
        imuWriter.writerow(["timestamp", "roll", "pitch", "yaw"])

        self.audioStream = audio.open(format=FORMAT, channels=CHANNELS,
                                       rate=RATE, input=True,
                                       frames_per_buffer=CHUNK)

        while self.recording:
            data = self.audioStream.read(CHUNK)
            self.audioFile.writeframes(data)

        self.audioStream.stop_stream()
        self.audioStream.close()
        self.audioFile.close()
        self.imuFile.close()
        audio.terminate()

        self.audioStream = None
        self.audioFile = None
        self.imuFile = None

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()