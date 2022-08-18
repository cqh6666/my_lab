# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_change_data
   Description:   ...
   Author:        cqh
   date:          2022/8/13 21:49
-------------------------------------------------
   Change Activity:
                  2022/8/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import sys

import pandas as pd
import numpy as np
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

from utils_api import get_train_test_data, get_shap_value, get_train_test_x_y


def load_data(all_data):
    """
    x: 标准化后的数据集x
    standardizer 标准器
    df_wine 数据集y
    """
    # read in from csv
    # replace nan with -99
    data_x = all_data.drop(drop_feature, axis=1) * init_similar_weight
    data_y = all_data[label_feature]
    data_x = data_x.values.reshape(-1, data_x.shape[1]).astype('float32')
    # stadardize values
    standardizer = preprocessing.StandardScaler()
    data_x = standardizer.fit_transform(data_x)
    return data_x, standardizer, data_y


def numpyToTensor(x):
    x_train = torch.from_numpy(x).to(device)
    return x_train


class DataBuilder(Dataset):
    def __init__(self, all_data):
        self.x, self.standardizer, self.wine = load_data(all_data)
        self.x = numpyToTensor(self.x)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


class Autoencoder(nn.Module):
    def __init__(self, D_in, H=50, H2=12, latent_dim=100):

        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        #         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        #         # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # self.decode(z) ist später recon_batch, mu ist mu und logvar ist logvar
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def train(epoch_):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    #        if batch_idx % log_interval == 0:
    #            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                epoch, batch_idx * len(data), len(trainloader.dataset),
    #                       100. * batch_idx / len(trainloader),
    #                       loss.item() / len(data)))
    if epoch_ % log_interval == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch_, train_loss / len(trainloader.dataset)))
        # train_losses.append(train_loss / len(trainloader.dataset))


def get_embedding(dataloader):
    mu_output = []
    logvar_output = []

    with torch.no_grad():
        for i, (data) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            mu_tensor = mu
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)

            logvar_tensor = logvar
            logvar_output.append(logvar_tensor)
            logvar_result = torch.cat(logvar_output, dim=0)

    return mu_result


if __name__ == '__main__':

    # 间隔多少输出
    log_interval = 100
    drop_feature = ['Label', 'ID']
    label_feature = 'Label'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"./result/new_data"
    """
    version=1 dim=100
    version=2 dim=20 50 100
    """
    dimension = int(sys.argv[1])
    version = sys.argv[2]
    # ======================================================
    train_data_file = os.path.join(save_path, f"train_data_dim{dimension}_v{version}.csv")
    test_data_file = os.path.join(save_path, f"test_data_dim{dimension}_v{version}.csv")
    # ======================================================
    # 数据路径
    # train_data_path = r"D:\dataset\data\data_process_df\mushroom_train_df.csv"
    # test_data_path = r"D:\dataset\data\data_process_df\mushroom_test_df.csv"
    # 读取数据
    train_data, test_data = get_train_test_data()
    print("get all data", train_data.shape, test_data.shape)
    # train_data = load_data(train_data_path)
    # test_data = load_data(test_data_path)

    init_similar_weight = get_shap_value()

    # 建立数据
    train_data_set = DataBuilder(train_data)
    trainloader = DataLoader(dataset=train_data_set, batch_size=1024)
    test_data_set = DataBuilder(test_data)
    testloader = DataLoader(dataset=test_data_set, batch_size=1024)

    # AutoEncoder
    D_in = train_data_set.x.shape[1]
    H = 50
    H2 = 12
    # 降维的维度
    model = Autoencoder(D_in, H, H2, latent_dim=dimension).to(device)
    # model.apply(weights_init_uniform_rule)
    # sae.fc4.register_forward_hook(get_activation('fc4'))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = customLoss()

    epochs = 5000
    val_losses = []
    train_losses = []
    # train
    for epoch in range(1, epochs + 1):
        train(epoch)

    # embedding
    train_result = get_embedding(trainloader)
    test_result = get_embedding(testloader)

    train_df = pd.DataFrame(train_result.numpy())
    train_df.to_csv(train_data_file)
    test_df = pd.DataFrame(test_result.numpy())
    test_df.to_csv(test_data_file)

    print("save success!", train_df.shape, test_df.shape)
