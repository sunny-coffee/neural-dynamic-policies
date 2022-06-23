import numpy as np
import torch

# import matplotlib
# matplotlib.use('Agg')


from mnist.utils.dataset import TrainDataset, TestDataset
from mnist.model.elmo_model import Elmo
from torch.utils.data import DataLoader

import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ndp-il')
args = parser.parse_args()

data_path = './mnist/data/40x40-smnist.mat'
inds = np.arange(1024+128)
np.random.shuffle(inds)
test_inds = inds[1024:]
train_inds = inds[:1024]
train_dataset = TrainDataset(data_path, train_inds)
test_dataset = TestDataset(data_path, test_inds)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)

time = str(datetime.now())
time = time.replace(' ', '_')
time = time.replace(':', '_')
time = time.replace('-', '_')
time = time.replace('.', '_')
model_save_path = './mnist/data/' + args.name + '_' + time
os.mkdir(model_save_path)

learning_rate = 1e-4
num_epochs = 151

elmo = Elmo(state_dim=2, hidden_size=16)
optimizer = torch.optim.Adam(elmo.parameters(), lr=learning_rate)
pretrain_criterion = torch.nn.MSELoss(reduction = 'mean')
# print(torch.cat(train_dataset[0], dim=1))
# print(torch.cat(train_dataset[0], dim=1).shape)

for epoch in range(num_epochs):
    print(epoch)
    elmo.train()
    for i,data in enumerate(train_loader):
        x, y =data
        # with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        y_h = elmo.fit(x)
        # print(y_h.shape)
        # print(y.shape)
        loss = pretrain_criterion(y_h, y)
        # print(loss.shape)
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        elmo.eval()
        losses = []
        with torch.no_grad():
            for i,data in enumerate(train_loader):
                x, y =data
                y_h = elmo.fit(x)
                # print(y_h.data[0],y_h.data[1])
                test_loss = pretrain_criterion(y_h, y)
                losses.append(test_loss.item())
        mean_loss = sum(losses)/len(losses)
        torch.save(elmo.state_dict(), model_save_path + '/elmo_model.pt')
        print('Epoch: ' + str(epoch) + ', Test Error: ' + str(mean_loss))


