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

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('DataCollectorNode')
        self.recording = False

        self.imuSub = self.create_subscription(MowerImu, '/hqv_mower/imu0/orientation', self.IMUCallback, 10)
        self.orientation = np.zeros(3)

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
        self.orientation = np.array([msg.roll, msg.pitch, msg.yaw])

    def run_flask(self):
        self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    def record(self, label):
        audio = pyaudio.PyAudio()

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 1

        sample = 0

        dirPath = f"dataset/{label}"
        os.makedirs(dirPath, exist_ok=True) 

        #Record audio and imu data
        while self.recording:
            
            sampleName = f"sample_{int(time.time() * 1000)}_{sample+1:03d}"
            sampleFolder = os.path.join(dirPath, sampleName)
            os.makedirs(sampleFolder, exist_ok=True)

            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            frames = []
            
            audioPath = os.path.join(sampleFolder, f"{sampleName}.wav")
            imuPath = os.path.join(sampleFolder, f"{sampleName}.npy")

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()

            #Save the audio data
            waveFile = wave.open(audioPath, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            #Save the IMU data
            np.save(imuPath, self.orientation)
            
            sample += 1    

        audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()