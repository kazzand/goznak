import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from datasets import CLFData, DenoiseData
from functions import spec2image



class Predict():
    def __init__(self, path_to_model, path_to_data, task):
        super(Predict, self).__init__()
        self.model = torch.load(path_to_model)
        self.path_to_data = path_to_data

    def predict_label(numpy_spec):
        image = spec2image(numpy_spec.T)
        if image.shape[1] <= 200:
            image_crop = np.zeros((80, 200))
            image_crop[:image.shape[0], :image.shape[1]] = image
        else:
            images = []
            for i in range(0, image.shape[1], 200):
                if i + 200 < image.shape[1]:
                    image_crop = image[:, i:i + 200]
                    images.append(torch.FloatTensor(image_crop))
            images.append(torch.FloatTensor(image_crop[:, image.shape[1] - 201:image.shape[1] - 1]))
        self.model.eval()
        with torch.no_grad():
            predict = self.model(images)
        return predict.detach().cpu().numpy().ravel()

