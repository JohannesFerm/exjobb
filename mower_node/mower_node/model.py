import torch.nn as nn
import timm
import torch

class MultimodalModel(nn.Module):
    def __init__(self, numClasses):
        super(MultimodalModel, self).__init__()
        
        #Define cnn, remove classification layer
        self.cnn = timm.create_model("efficientnet_lite0", pretrained=True, in_chans=1) 
        self.cnn.fc = nn.Identity() 

        #Define lstm
        self.lstm = nn.LSTM(input_size=3, hidden_size=512, num_layers=2, batch_first=True)

        #Fully connected layer
        self.fc = nn.Linear(1512, numClasses)
    
    def forward(self, audio, imu):
        cnnOut = self.cnn(audio)

        imuOut, (hn, cn) = self.lstm(imu)
        imuOut = hn[-1]

        #Put the features into a fully connected layer
        combined = torch.cat((cnnOut, imuOut), dim=1)
        return self.fc(combined)