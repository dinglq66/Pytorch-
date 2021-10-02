import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules import loss
torch.manual_seed(10)


# ============================ step 1/5 生成数据================================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)

x0 = torch.normal(mean_value*n_data, 1) + bias
y0 = torch.zeros(sample_nums)
x1 = torch.normal(-mean_value*n_data, 1) + bias
y1 = torch.ones(sample_nums)

train_x = torch.cat([x0, x1], 0)
train_y = torch.cat([y0, y1], 0)
# print('train_y shape:{}'.format(train_y.shape))

# ============================ step 2/5 选择模型================================
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()


# ============================ step 3/5 选择损失函数================================
loss_fn = nn.BCELoss()


# ============================ step 4/5 选择优化器================================
lr = 0.01
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)


# ============================ step 5/5 迭代训练================================
for iteration in range(1000):
    y_pred = lr_net(train_x)
    loss = loss_fn(y_pred.squeeze(), train_y)
    # if iteration % 200 == 0:
    #     print('y_pred shape:', y_pred.shape)
    loss.backward() # 反向传播
    optimizer.step() # 更新参数
    optimizer.zero_grad() # 清空梯度
    if iteration % 100 == 0:
        mask = y_pred.ge(0.5).float().squeeze()
        correct = (mask==train_y).sum()
        accuracy = correct / train_y.size(0)

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-5, 7)
        plt.plot(plot_x, plot_y)
        plt.text(-5, 5, 'Loss:%.4f'% loss.data.numpy(), fontdict={'size':20, 'color':'red'})
        plt.title('Iteration:{}\nw0:{}w1:{}b:{}accuracy:{}'.format(iteration, w0, w1, plot_b, accuracy))
        plt.legend()

        plt.show()
        plt.pause(0.5)
        if accuracy > 0.99:
            break