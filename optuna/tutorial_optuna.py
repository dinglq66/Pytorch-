import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms

import optuna


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 20
BATCH_SIZE = 128
N_TRAIN_EXAMPLES = BATCH_SIZE * 30
N_VALID_EXAMPLES = BATCH_SIZE * 30


# define the model
class ConvNet(nn.Module):
    def __init__(self, trial):
        # 我们将要优化卷积神经网络中的drop out率
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5, step=0.1)
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        fc2_input_dim = trial.suggest_int("fc2_input_dim", 32, 128, 32)
        self.fc1 = nn.Linear(32*7*7, fc2_input_dim)
        dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.3, step=0.1)
        self.drop2 = nn.Dropout2d(p=dropout_rate2)
        fc3_input_dim = trial.suggest_int("fc3_input_dim", 32, 128, 32)
        self.fc2 = nn.Linear(fc2_input_dim, fc3_input_dim)
        self.fc3 = nn.Linear(fc3_input_dim, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def get_mnist(train_dataset, batch_size):
    """
    得到MNIST数据集的数据加载器
    :param train_dataset:
    :param batch_size:
    :return: train_loader, val_lodaer
    """

    train_data, val_data = random_split(train_dataset, [m-int(m*0.2), int(m*0.2)])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader


def objective(trial):
    """
    定义目标函数，使用采样程序来选择每次试验的超参数值，并返回在该试验中获得的验证准确度
    :param trial:
    :return: accuracy
    """
    model = ConvNet(trial).to(DEVICE)
    # 尝试不同的优化器对结果的影响
    # 比较RMSprop和SGD，都具有momentum参数
    optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum)

    # 比较Adam和Adadelta
    '''
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    '''
    batch_size = trial.suggest_int("batch_size", 64, 128, step=64)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_mnist(train_dataset=train_dataset, batch_size=batch_size)
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx * batch_size >= N_TRAIN_EXAMPLES:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


        # 验证阶段，要防止梯度回传
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx_val, (images_val, labels_val) in enumerate(val_loader):
                if batch_idx_val * batch_size >= N_VALID_EXAMPLES:
                    break
                images_val, labels_val = images_val.to(DEVICE), labels_val.to(DEVICE)
                output_val = model(images_val)
                pred = output_val.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels_val.view_as(pred)).sum().item()

        accuracy = correct / len(val_loader.dataset)
        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy


if __name__ == '__main__':
    # create the dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST('classifier_data', train=True, download=True, transform=transform)
    m = len(train_dataset)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    trial = study.best_trial
    print('Accuracy:{}'.format(trial.value))
    print('Best hyperparameters:{}'.format(trial.params))

    # df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'], axis=1)
    # df.tail(5)

    # optuna.visualization.plot_optimization_history(study)
    # optuna.visualization.plot_contour(study, params=['batch_size', 'lr'])
    # optuna.visualization.plot_param_importances(study)


