import os
from multiprocessing import Process, Queue
import json
import pandas as pd
import torch
from mower_node.model import MultimodalModel
import numpy as np

#Function to start process that updates map
def initModelIdle(model):
    cQueue = Queue()
    mQueue = Queue()
    p = Process(target=useModelInIdle, args=(model, cQueue, mQueue))
    p.start()
    return cQueue, mQueue, p

#Function to update the map in the background when the UI is on the main page
def useModelInIdle(modelPath, commandQueue, mapQueue, dataPath = "map_logs"):

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

    #Load the model
    checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
    NUM_CLASSES = len(checkpoint['label_mapping'])
    try:
        model = MultimodalModel(numClasses=NUM_CLASSES)
    except Exception as e:
        print("Error during model instantiation:", e, flush=True)
        import traceback
        traceback.print_exc()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = None
    lastRow = 0
    mapData = None
    while True:
        command = commandQueue.get()
        
        #If new map data is coming in, use the latest log file
        if command == "newData":
            folders = [f for f in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, f))]
            latestFolder = max(folders, key=lambda folder: os.path.getmtime(os.path.join(dataPath, folder)))
            pickleFile = os.path.join(dataPath, os.path.join(latestFolder, 'data.pkl'))

            if os.path.exists(pickleFile):
                df = pd.read_pickle(pickleFile)
                lastRow = 0
            else:
                print("DATA FILE NOT FOUND")
                continue

            mapPath = os.path.join(dataPath, os.path.join(latestFolder, "map.json"))
            if os.path.exists(mapPath):
                with open(mapPath, "r") as f:
                    mapData = json.load(f)
            else:
                print("MAP FILE NOT FOUND")
                continue

        #If in idle state on the UI, start processing and updating the map
        elif command == "idle":
            changed = False
            if df is not None:
                for index in  range(lastRow, len(df)):
                    if not commandQueue.empty():
                        command = commandQueue.get()
                        if command == "busy":
                            lastRow = index
                            if mapData is not None and changed:
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
                        changed = True
                else:
                    lastRow = len(df)
        
        elif command == "busy": #If we get a busy command but have finished processing the latest file, send the new map
            if mapData is not None and changed:
                mapQueue.put(mapData)