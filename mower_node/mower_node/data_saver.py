import os
from multiprocessing import Process, Queue
import time
import json
import pandas as pd

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