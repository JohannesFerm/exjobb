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
from hqv_public_interface.msg import MowerWheelSpeed
from geopy.distance import distance
import datetime 
import csv
import torch
from mower_node.model import MultimodalModel
import os
import librosa
import contextlib
from geopy.distance import geodesic
from multiprocessing import Process, Queue
import json
import pandas as pd

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

#Function that starts the data saving process
def initDataSaverProcess(dir = "map_logs"):
    os.makedirs(dir, exist_ok=True)
    queue = Queue()
    p = Process(target=dataSaver, args=(queue, dir))
    p.start()
    return queue, p

#Function that saves inference data when map building
def dataSaver(queue, dir):
    timestamp = int(time.time())
    sampleDir  = os.path.join(dir, f"sample_{timestamp}")
    os.makedirs(sampleDir, exist_ok=True)

    sampleFile = os.path.join(sampleDir, "data.csv")
    with open(sampleFile, 'w', newline="") as file:

        data = []
        while True:
            sample = queue.get()
            
            #If stop save the map
            if isinstance(sample, dict) and sample.get("type") == "STOP":
                mapData = sample.get("map")
                if mapData:
                    mapPath = os.path.join(sampleDir, "map.json")
                    with open(mapPath, "w") as mf:
                        json.dump(mapData, mf, indent=2)
                break
            
            #Save rows
            elif isinstance(sample, dict): 
                data.append(sample)
        
        df = pd.DataFrame(data)
        picklePath = os.path.join(sampleDir, "data.pkl")
        df.to_pickle(picklePath)

#Function that stops the data saving process
def stopDataSaverProcess(queue, process, map):
    stop = {"type": "STOP"}
    if map:
        stop["map"] = map
    queue.put(stop)
    queue.close()
    queue.join_thread()
    process.join()

#Function to start process that updates map
def initModelIdle(model):
    cQueue = Queue()
    mQueue = Queue()
    p = Process(target=useModelInIdle, args=(model, cQueue, mQueue))
    p.start()
    return cQueue, mQueue, p

#Function to update the map in the background when the UI is on the main page
def useModelInIdle(modelPath, commandQueue, mapQueue, dataPath = "map_logs"):
    print("START OF USEMODELINIDLE", flush=True)
    print(f"Model path: {modelPath}", flush = True)

    #Unfortunate code duplication here, but seems to be easiest way
    def buildMapProcess(mapArg, pos, pred, mapDim):
        closestI, closestJ = None, None
        minDiff = float('inf') 

        #Loop through map
        for i in range(mapDim):
            for j in range(mapDim):
                cell = mapArg[i][j]
                if cell is None:
                    continue

                latDiff = abs(pos[0] - cell["lat"])
                lonDiff = abs(pos[1] - cell["lon"])
                diff = latDiff + lonDiff

                #Find the cell with the most similar gps position
                if diff < minDiff:
                    minDiff = diff
                    closestI, closestJ = i, j

        #Insert prediction in the map
        if closestI is not None and closestJ is not None:
            mapArg[closestI][closestJ]["prediction"] = pred

    print("BEFORE MODEL LOADING OF USEMODELINIDLE", flush=True)

    #Load the model
    checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
    print("LOAD SUCCESSFULL", flush=True)
    NUM_CLASSES = len(checkpoint['label_mapping'])
    print("LOAD SUCCESSFULL2", flush=True)
    try:
        print("Instantiating model...", flush=True)
        model = MultimodalModel(numClasses=NUM_CLASSES)
        print("Model instantiated successfully", flush=True)
    except Exception as e:
        print("Error during model instantiation:", e, flush=True)
        import traceback
        traceback.print_exc()

    #model = MultimodalModel(numClasses=NUM_CLASSES)
    print("LOAD SUCCESSFULL3", flush=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("LOAD SUCCESSFULL4", flush=True)
    model.eval()

    print("AFTER MODEL LOADING OF USEMODELINIDLE", flush=True)

    df = None
    lastRow = 0
    while True:
        print("BEFORE COMMAND", flush=True)
        command = commandQueue.get()
        print("AFTER COMMAND", flush=True)
        
        #If new map data is coming in, use the latest log file
        if command == "newData":
            folders = [f for f in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, f))]
            latestFolder = max(folders, key=lambda folder: os.path.getmtime(os.path.join(dataPath, folder)))
            pickleFile = os.path.join(latestFolder, 'data.pkl')
            if os.path.exists(pickleFile):
                df = pd.read_pickle(pickleFile)
                lastRow = 0
            else:
                print("FILE NOT FOUND")
                continue

            mapPath = os.path.join(latestFolder, "map.json")
            if os.path.exists(mapPath):
                with open(mapPath, "r") as f:
                    mapData = json.load(f)
            else:
                print("FILE NOT FOUND")
                continue

        #If in idle state on the UI, start processing and updating the map
        elif command == "idle":
            print("IDLE COMMAND", flush=True)
            if df is not None:
                for index in  range(lastRow, len(df)):
                    if not commandQueue.empty():
                        command = commandQueue.get()
                        if command == "busy":
                            lastRow = index
                            if mapData is not None:
                                mapQueue.put(mapData)
                            break

                    row = df.iloc[index]
                    audio = row["spectrogram"]
                    imu = row["imu"]

                    audioTensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                    imuTensor = torch.tensor(np.stack(imu), dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        output = model(audioTensor, imuTensor)

                    probs = torch.softmax(output, dim=1)
                    confidence, pred = torch.max(probs, 1)
                    pred = pred.item()
                    if pred == row["label"]:
                        continue
                    if confidence > 0.7: #Use a high confidence, so only change/update map if prediction is high certainty
                        buildMapProcess(mapData, row["gps"], pred, len(mapData))
                else:
                    lastRow = len(df)

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
        self.predText = None
        self.testing = False
        self.model = None
        self.imuBuffer =  [[0.0, 0.0, 0.0] for _ in range(10)]

        #For map building
        self.cellWidth = 0.54 #Mowers width in meters
        self.cellHeight = 0.75 #Mowers height in meters
        self.mapDim = 50
        self.startPos = None
        self.mapBuilding = False
        self.map = [[{"lat": None, "lon": None, "prediction": -1} for _ in range(self.mapDim)] for _ in range(self.mapDim)]

        #Load the model
        modelPath = "models/model_with_labels.pth"
        checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
        NUM_CLASSES = len(checkpoint['label_mapping'])
        self.model = MultimodalModel(numClasses=NUM_CLASSES)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.labelMapping = checkpoint['label_mapping']

        #Librosa uses numba, dummy call to avoid first (real) call taking longer
        dummyAudio = np.zeros(44100 * 2, dtype=np.float32) 
        _ = librosa.feature.melspectrogram(y=dummyAudio, sr=44100, n_fft=2048, hop_length=512, n_mels=128)

        #For wheel speed
        self.wheel0Sub = self.create_subscription(MowerWheelSpeed, '/hqv_mower/wheel0/speed', self.wheel0Callback, 10)
        self.wheel1Sub = self.create_subscription(MowerWheelSpeed, '/hqv_mower/wheel1/speed', self.wheel1Callback, 10)
        self.wheel0buffer = []
        self.wheel1buffer = []

        #For the map idle updating process, start it
        self.commandQueue, self.mapQueue, self.mapProcess = initModelIdle(modelPath)
        self.commandQueue.put("idle")

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

                #Tell the map building process to stop and get new map if there is one
                self.commandQueue.put("busy")
                if not self.mapQueue.empty():
                    newMap = self.mapQueue.get()
                    self.map = newMap
                else:
                    print("Map queue empty")

                thread = threading.Thread(target=self.record, args=(label,))
                thread.start()
            return jsonify({"status": f"Recording {label}"})

        #For stopping the audio recording
        @self.app.route("/stop", methods=['GET'])
        def stop_recording():
            self.recording = False 
            self.commandQueue.put("idle")
            return jsonify({"status": "Recording stopped"})
        
        #For testing the model
        @self.app.route("/start-test", methods=['GET'])
        def startTest():
            self.testing = True

            #Tell the map building process to stop and get new map if there is one
            self.commandQueue.put("busy")
            if not self.mapQueue.empty():
                newMap = self.mapQueue.get()
                self.map = newMap
            else:
                print("Map queue empty")

            threading.Thread(target=self.runModel, daemon=True).start()
            return jsonify({"status": "Test started"})
        
        #For stopping the testing
        @self.app.route("/stop-test", methods=["GET"])
        def stopTest():
            self.testing = False
            self.commandQueue.put("idle")
            return jsonify({"status": "Test stopped"})

        #For getting the current prediction to put on screen
        @self.app.route("/prediction", methods=["GET"])
        def getPrediction():
            if self.testing:
                return jsonify({"prediction": self.predText, "status": "testing"})
            else:
                return jsonify({"prediction": "Not testing", "status": "idle"})

        #For map building
        @self.app.route("/start-map", methods = ["GET"])
        def startMapBuilding():
            self.mapBuilding = True
            self.testing = True

            #Tell the map building process to stop and get new map if there is one
            self.commandQueue.put("busy")
            if not self.mapQueue.empty():
                newMap = self.mapQueue.get()
                self.map = newMap
            else:
                print("Map queue empty")

            if self.startPos is None:
                self.startPos = self.pos
                self.createMap(self.startPos)
            self.queue, self.process = initDataSaverProcess() #Start the data saving process
            threading.Thread(target=self.runModel, daemon=True).start()
            return jsonify({"status": "Map building started"})
        
        #For stopping map building
        @self.app.route("/stop-map",  methods = ["GET"])
        def stopMapBuilding():
            self.mapBuilding = False
            self.testing = False
            stopDataSaverProcess(self.queue, self.process, self.map) #Stop data saving process
            self.commandQueue.put("newData")
            self.commandQueue.put("idle")
            return jsonify({"status": "Map building stopped"})

        @self.app.route("/map", methods=["GET"])
        def getMap():
            processedMap = []
            for i in range(self.mapDim):
                row = []  
                for j in range(self.mapDim):
                    prediction = self.map[i][j]["prediction"] if self.map[i][j]["prediction"] != -1 else -1
                    row.append(prediction)  
                processedMap.append(row) 
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

        self.pos = [msg.latitude, msg.longitude]

    #Callbacks for wheel speed, 25 ms
    def wheel0Callback(self, msg):
        if self.testing:
            self.wheel0buffer.append(msg.speed)

    def wheel1Callback(self, msg):
        if self.testing:
            self.wheel1buffer.append(msg.speed)

    def run_flask(self):
        self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    #Function for data collection
    def record(self, label):
        with suppressStderr():
            audio = pyaudio.PyAudio()

        #Define directory
        sampleId = f"sample_{int(time.time())}"
        self.sampleFolder = f"datasetNew/{label}/{sampleId}"
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
            data = stream.read(self.CHUNK, exception_on_overflow=False)
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
            self.wheel0buffer = []
            self.wheel1buffer = []
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
            currentPos = self.pos #Use this line so that the GPS position isn't affected (updated) by the inference taking half a second
                                    #If the timing is bad it might happen that the GPS position from the callback belongs to the next clip without this line
        imuBufferCur = self.imuBuffer #Avoid same issue as GPS

        numpyWheel0 = np.array(self.wheel0buffer)
        numpyWheel1 = np.array(self.wheel1buffer)

        #Sometimes one callback gets one sample ahead of the other, clip size of arrays to smaller one
        minLen = min(len(numpyWheel0), len(numpyWheel1))
        numpyWheel0 = numpyWheel0[:minLen]
        numpyWheel1 = numpyWheel1[:minLen]

        notMovingForward = (numpyWheel0 <= 0.01) & (numpyWheel1 <= 0.01)

        if np.sum(notMovingForward) / len(numpyWheel0) > 0.3:
            self.predText = "Not moving properly"
            return

        soundClip = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        soundClip = librosa.feature.melspectrogram(y=soundClip, sr=self.RATE, n_fft=2048, hop_length=512, n_mels=128)
        soundClip = librosa.power_to_db(soundClip, ref=np.max)

        #Tensors for the model
        audioTensor = torch.tensor(soundClip, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        imuTensor = torch.tensor(np.stack(imuBufferCur), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model.forward(audioTensor, imuTensor)

        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        self.prediction = pred.item() 

        if confidence.item() < 0.6:
            self.predText = "Low confidence"
            return
        else:
            self.predText = self.labelMapping[self.prediction]

        if self.mapBuilding and self.pos is not None:
            self.buildMap(currentPos, self.prediction)
            self.queue.put({
                "spectrogram": soundClip,
                "imu": imuBufferCur,                
                "gps": currentPos,                
                "label": self.prediction              
            })

    #Function to add prediction to right position in the map
    def buildMap(self, pos, pred):
        closestI, closestJ = None, None
        minDiff = float('inf') 

        #Loop through map
        for i in range(self.mapDim):
            for j in range(self.mapDim):
                cell = self.map[i][j]
                if cell is None:
                    continue

                latDiff = abs(pos[0] - cell["lat"])
                lonDiff = abs(pos[1] - cell["lon"])
                diff = latDiff + lonDiff

                #Find the cell with the most similar gps position
                if diff < minDiff:
                    minDiff = diff
                    closestI, closestJ = i, j

        #Insert prediction in the map
        if closestI is not None and closestJ is not None:
            self.map[closestI][closestJ]["prediction"] = pred

    #Function to build the map with gps coordinates
    def createMap(self, pos):
        #Loop through whole grid
        for i in range(self.mapDim):
            for j in range(self.mapDim):
                dy = (self.mapDim // 2 - i) * self.cellHeight
                dx = (self.mapDim // 2 - j) * self.cellWidth

                pointY = distance(meters=abs(dy)).destination(pos, bearing=0 if dy >= 0 else 180)
                point = distance(meters=abs(dx)).destination(pointY, bearing=90 if dx >= 0 else 270)

                self.map[i][j] = {"lat": point.latitude, "lon": point.longitude, "prediction": -1}


def main(args=None):
    rclpy.init(args=args)
    node = MowerNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()