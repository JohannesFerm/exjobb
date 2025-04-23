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
from hqv_public_interface.msg import MowerGnssPosition
import datetime 
import csv
import torch
from mower_node.model import MultimodalModel
import os
import librosa


class MowerNode(Node):
    def __init__(self):
        super().__init__('MowerNode')

        #For audio
        self.recording = False
        self.audioStream = None
        self.audioFile = None

        #For IMU
        self.imuSub = self.create_subscription(MowerImu, '/hqv_mower/imu0/orientation', self.IMUCallback, 10)
        self.imuFile = None
        self.orientation = np.zeros(3)
        self.lastCallbackTime = time.time()
        
        #For GPS
        self.gpsSub = self.create_subscription(MowerGnssPosition, '/hqv_mower/gnss/position', self.GPSCallback, 10)
        self.gpsFile = None
        self.pos = np.zeros(2)
        
        self.sampleFolder = None

        #For testing the model
        self.prediction = None
        self.testing = False
        self.model = None
        self.imuBuffer =  [[0.0, 0.0, 0.0] for _ in range(10)]

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
        
        #For testing the model
        @self.app.route("/start-test", methods=['GET'])
        def startTest():
            self.testing = True
            threading.Thread(target=self.test, daemon=True).start()
            return jsonify({"status": "Test started"})
        
        @self.app.route("/stop-test", methods=["GET"])
        def stopTest():
            self.testing = False
            return jsonify({"status": "Test stopped"})

        @self.app.route("/prediction", methods=["GET"])
        def getPrediction():
            if self.testing:
                return jsonify({"prediction": self.prediction, "status": "testing"})
            else:
                return jsonify({"prediction": "Not testing", "status": "idle"})

        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()

    #Collect IMU data, 1 second intervals
    def IMUCallback(self, msg):
        currentTime = time.time()
        if currentTime - self.lastCallbackTime >= 1.0:
            self.lastCallbackTime = currentTime 
            self.orientation = np.array([msg.roll, msg.pitch, msg.yaw])

            if self.imuFile:
                self.imuFile.write(f"{currentTime},{self.orientation[0]},{self.orientation[1]},{self.orientation[2]}\n")
            elif self.testing:
                self.imuBuffer.append(self.orientation)
                self.imuBuffer = self.imuBuffer[1:]

    #Collect GPS data, 1 second intervals (decided by the topic)
    def GPSCallback(self, msg):
        if self.gpsFile:
            self.gpsFile.write(f"{time.time()},{msg.latitude},{msg.longitude}\n")

    def run_flask(self):
        self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    def record(self, label):
        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512

        #Define directory
        sampleId = f"sample_{int(time.time())}"
        self.sampleFolder = f"dataset/{label}/{sampleId}"
        os.makedirs(self.sampleFolder, exist_ok=True)

        #Define the different data files
        audioPath = os.path.join(self.sampleFolder, f"audio{time.time()}.wav")
        imuPath = os.path.join(self.sampleFolder, f"imu{time.time()}.csv")
        gpsPath = os.path.join(self.sampleFolder, f"gps{time.time()}.csv")

        #Setup audio
        self.audioFile = wave.open(audioPath, 'wb')
        self.audioFile.setnchannels(CHANNELS)
        self.audioFile.setsampwidth(audio.get_sample_size(FORMAT))
        self.audioFile.setframerate(RATE)

        #Setup IMU
        self.imuFile = open(imuPath, 'w', newline='')
        imuWriter = csv.writer(self.imuFile)
        imuWriter.writerow(["timestamp", "roll", "pitch", "yaw"])

        #Setup GPS
        self.gpsFile = open(gpsPath, 'w', newline='')
        gpsWriter = csv.writer(self.gpsFile)
        gpsWriter.writerow(["timestamp", "latitude", "longitude"])

        #Record audio
        self.audioStream = audio.open(format=FORMAT, channels=CHANNELS,
                                       rate=RATE, input=True,
                                       frames_per_buffer=CHUNK)
        while self.recording:
            data = self.audioStream.read(CHUNK)
            self.audioFile.writeframes(data)

        #Clean up
        self.audioStream.stop_stream()
        self.audioStream.close()
        self.audioFile.close()
        self.imuFile.close()
        self.gpsFile.close()
        audio.terminate()

        self.audioStream = None
        self.audioFile = None
        self.imuFile = None
        self.gpsFile = None
    
    def test(self):

        #Load the model
        if self.model is None:

            #Find the weights
            model_path = "models/model_weights.pth"
            
            NUM_CLASSES = 4 #Should be 5
            self.model = MultimodalModel(numClasses=NUM_CLASSES)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()

        #Setup for audio
        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 2
        while self.testing:

            start = time.time()
            #Record audio (2 seconds)
            frames = []
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                       rate=RATE, input=True,
                                       frames_per_buffer=CHUNK)
            
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            #Preprocess audio, convert to spectrogram
            soundClip = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            soundClip = librosa.feature.melspectrogram(y=soundClip, sr=RATE, n_fft=2048, hop_length=512, n_mels=128)
            soundClip = librosa.power_to_db(soundClip, ref=np.max)

            #Tensors for the model
            audioTensor = torch.tensor(soundClip, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            imuTensor = torch.tensor(np.stack(self.imuBuffer), dtype=torch.float32).unsqueeze(0)
            
            output = self.model.forward(audioTensor, imuTensor)

            _, pred = torch.max(output, 1)
            self.prediction = pred.item() #Set output on the server

            stop = time.time()
            print(f"TIME: {start - stop}")
        
        audio.terminate()
        self.imuBuffer =  [[0.0, 0.0, 0.0] for _ in range(10)]
        self.prediction = None   

def main(args=None):
    rclpy.init(args=args)
    node = MowerNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()