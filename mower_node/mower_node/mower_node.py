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
import contextlib
from geopy.distance import geodesic

#Helper to avoid ALSA warnings
@contextlib.contextmanager
def suppressStderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr_fd = os.dup(2) 
        os.dup2(devnull.fileno(), 2) 
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)

class MowerNode(Node):
    def __init__(self):
        super().__init__('MowerNode')

        #For audio
        self.recording = False
        self.audioFile = None
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 512
        self.RECORD_SECONDS = 2

        #For IMU
        self.imuSub = self.create_subscription(MowerImu, '/hqv_mower/imu0/orientation', self.IMUCallback, 10)
        self.imuFile = None
        self.orientation = np.zeros(3)
        self.lastCallbackTime = time.time()
        
        #For GPS
        self.gpsSub = self.create_subscription(MowerGnssPosition, '/hqv_mower/gnss/position', self.GPSCallback, 10)
        self.gpsFile = None
        self.pos = None
        
        self.sampleFolder = None

        #For testing the model
        self.prediction = None
        self.testing = False
        self.model = None
        self.imuBuffer =  [[0.0, 0.0, 0.0] for _ in range(10)]

        #For map building
        self.cellWidth = 0.54 #Mowers width in meters
        self.cellHeight = 0.75 #Mowers height in meters
        self.mapDim = 50
        self.startPos = None
        self.mapBuilding = False
        self.map = np.full((self.mapDim, self.mapDim), fill_value=-1, dtype=int)


        #Load the model
        model_path = "models/model_weights.pth"
        NUM_CLASSES = 4 #Should be 5
        self.model = MultimodalModel(numClasses=NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        #Librosa uses numba, dummy call to avoid first (real) call taking longer
        dummyAudio = np.zeros(44100 * 2, dtype=np.float32) 
        _ = librosa.feature.melspectrogram(y=dummyAudio, sr=44100, n_fft=2048, hop_length=512, n_mels=128)

        #Setup flask server
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
            threading.Thread(target=self.runModel, daemon=True).start()
            return jsonify({"status": "Test started"})
        
        #For stopping the testing
        @self.app.route("/stop-test", methods=["GET"])
        def stopTest():
            self.testing = False
            return jsonify({"status": "Test stopped"})

        #For getting the current prediction to put on screen
        @self.app.route("/prediction", methods=["GET"])
        def getPrediction():
            if self.testing:
                return jsonify({"prediction": self.prediction, "status": "testing"})
            else:
                return jsonify({"prediction": "Not testing", "status": "idle"})

        #For map building
        @self.app.route("/start-map", methods = ["GET"])
        def startMapBuilding():
            self.mapBuilding = True
            self.testing = True
            threading.Thread(target=self.runModel, daemon=True).start()
            return jsonify({"status": "Map building started"})
        
        #For stopping map building
        @self.app.route("/stop-map",  methods = ["GET"])
        def stopMapBuilding():
            self.mapBuilding = False
            self.testing = False
            return jsonify({"status": "Map building stopped"})

        @self.app.route("/map", methods=["GET"])
        def getMap():
            processedMap = self.map.tolist()
            return jsonify({"map": processedMap})


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
        elif self.mapBuilding:
            self.pos = [msg.latitude, msg.longitude]

    def run_flask(self):
        self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    #Function for data collection
    def record(self, label):
        with suppressStderr():
            audio = pyaudio.PyAudio()

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
        self.audioFile.setnchannels(self.CHANNELS)
        self.audioFile.setsampwidth(audio.get_sample_size(self.FORMAT))
        self.audioFile.setframerate(self.RATE)

        #Setup IMU
        self.imuFile = open(imuPath, 'w', newline='')
        imuWriter = csv.writer(self.imuFile)
        imuWriter.writerow(["timestamp", "roll", "pitch", "yaw"])

        #Setup GPS
        self.gpsFile = open(gpsPath, 'w', newline='')
        gpsWriter = csv.writer(self.gpsFile)
        gpsWriter.writerow(["timestamp", "latitude", "longitude"])

        #Record audio
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                       rate=self.RATE, input=True,
                                       frames_per_buffer=self.CHUNK)
        while self.recording:
            data = stream.read(self.CHUNK)
            self.audioFile.writeframes(data)

        #Clean up
        stream.stop_stream()
        stream.close()
        self.audioFile.close()
        self.imuFile.close()
        self.gpsFile.close()
        audio.terminate()

        self.audioFile = None
        self.imuFile = None
        self.gpsFile = None
    
    #Function for testing the model
    def runModel(self):
        with suppressStderr():
            audio = pyaudio.PyAudio()

        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)

        while self.testing:
            #Record audio
            frames = []            
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

            #Start a thread for inference
            threading.Thread(target=self.useModel, args=(frames,)).start()
        
        #Clean up
        stream.stop_stream()
        stream.close()
        audio.terminate()
        self.imuBuffer =  [[0.0, 0.0, 0.0] for _ in range(10)]
        self.prediction = None   

    #Function for doing inference and map building
    def useModel(self, frames):

        #Set start position for map if haven't
        if self.mapBuilding:
            if self.startPos is None:
                self.startPos = self.pos

            currentPos = self.pos #Use this line so that the GPS position isn't affected (updated) by the inference taking half a second
                                    #If the timing is bad it might happen that the GPS position from the callback belongs to the next clip without this line

        soundClip = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        soundClip = librosa.feature.melspectrogram(y=soundClip, sr=self.RATE, n_fft=2048, hop_length=512, n_mels=128)
        soundClip = librosa.power_to_db(soundClip, ref=np.max)

        #Tensors for the model
        audioTensor = torch.tensor(soundClip, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        imuTensor = torch.tensor(np.stack(self.imuBuffer), dtype=torch.float32).unsqueeze(0)
        
        output = self.model.forward(audioTensor, imuTensor)

        _, pred = torch.max(output, 1)
        self.prediction = pred.item() #Set output on the server

        if self.mapBuilding and self.pos is not None:
            self.buildMap(currentPos, self.prediction)

    #Function that builds the map based on positions and model output
    def buildMap(self, pos, pred):
        #Get the difference between starting point and current point in meters
        xDiff = geodesic((self.startPos[0], self.startPos[1]), (self.startPos[0], pos[1])).meters
        yDiff = geodesic((self.startPos[0], self.startPos[1]), (pos[0], self.startPos[1])).meters
        xSign = 1 if pos[1] >= self.startPos[1] else -1
        ySign = 1 if pos[0] >= self.startPos[0] else -1
        xDiff *= xSign
        yDiff *= ySign

        #Get the index of current position, treat the middle of the grid as origin
        xIndex = int(round(xDiff / self.cellWidth + self.mapDim // 2))
        yIndex = int(round(yDiff / self.cellHeight + self.mapDim // 2))

        if 0 <= xIndex < self.mapDim and 0 <= yIndex < self.mapDim:
            self.map[xIndex, yIndex] = pred

def main(args=None):
    rclpy.init(args=args)
    node = MowerNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()