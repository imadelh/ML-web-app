import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .network import Net
from PIL import Image

# Define model - ref CNN2

class MyModel:

    def __init__(self, model_weights:str, device:str):
        '''

        '''
        self.net = Net()
        self.weights = model_weights
        self.device = torch.device('cuda:0' if device=='cuda' else 'cpu')
        self.preprocess = transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                        ])
        self._initialize()


    def _initialize(self):
        # Load weights
        try:
            # Force loading on CPU if there is no GPU
            if(torch.cuda.is_available() == False):
                self.net.load_state_dict(torch.load(self.weights,map_location=lambda storage, loc: storage)["state_dict"])
            else:
                self.net.load_state_dict(torch.load(self.weights)["state_dict"])

        except IOError:
            print("Error Loading Weights")
            return None
        self.net.eval()

        # Move to specified device
        self.net.to(self.device)

    def predict(self,path):
        # Open the Image and resize
        img = Image.open(path).convert('L')

        # Convert to tensor on device
        with torch.no_grad():
            img_tensor = self.preprocess(img) # tensor in [0,1]
            img_tensor = 1 - img_tensor
            img_tensor = img_tensor.view(1, 28, 28, 1).to(self.device)

            # Do Inference
            probabilities = self.net(img_tensor)
            probabilities = F.softmax(probabilities, dim = 1)

        return probabilities[0].cpu().numpy()
