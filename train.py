import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from IPython.display import clear_output
from sklearn.metrics import roc_auc_score
from datasets import CLFData, DenoiseData

class Training():
    def __init__(self, path_to_train, path_to_val, task):
        super(Training, self).__init__()
        self.path_to_train = path_to_train
        self.path_to_val = path_to_val
        self.task = task
        self.model = None
        self.num_epochs = None
        self.loss = None
        self.train_loader = None
        self.val_loader = None

    def create_model(self):
        if self.task == 'classifier':
            model = models.resnet18()
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
            self.model = nn.Sequential(
                model,
                nn.Sigmoid()
            )
            self.num_epochs = 2
            self.loss = nn.BCELoss()
            train_data = CLFData(self.path_to_train)
            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
            val_data = CLFData(self.path_to_val)
            self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=True)
        elif self.task == 'denoise':
            class ResBlock(nn.Module):
                def __init__(self, ch):
                    super(ResBlock, self).__init__()
                    self.gamma = nn.Parameter(torch.zeros(1))

                    self.l1 = nn.Sequential(
                        nn.BatchNorm2d(ch),
                        nn.Conv2d(ch, ch, 3, padding=1)
                    )

                    self.l2 = nn.Sequential(
                        nn.BatchNorm2d(ch),
                        nn.Conv2d(ch, ch, 3, padding=1)
                    )

                def forward(self, x):
                    return x + self.gamma * self.l2(self.l1(x))

            base_model = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.LeakyReLU(),
                ResBlock(16),
                nn.Conv2d(16, 64, 3, padding=1),
                nn.LeakyReLU(),
                ResBlock(64),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LeakyReLU(),
                ResBlock(64),
                nn.Conv2d(64, 1, 1),
            )

            class SuperRes(nn.Module):
                def __init__(self, base_model):
                    super(SuperRes, self).__init__()
                    self.gamma = nn.Parameter(torch.zeros(1))
                    self.base_model = base_model

                def forward(self, x):
                    return x + self.gamma * self.base_model(x)

            self.model = SuperRes(base_model)
            self.num_epochs = 10
            self.loss = nn.MSELoss()
            train_data = DenoiseData(self.path_to_train)
            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
            val_data = DenoiseData(self.path_to_val)
            self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=True)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        for epoch in range(self.num_epochs):
            for X_batch, y_batch in tqdm(self.train_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                predict = model(X_batch)
                loss = self.loss(predict,y_batch)
                loss.backward()
                optimizer.step()
        torch.save(model, self.path_to_train+'model')


