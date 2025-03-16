import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.autograd as autograd
from matplotlib import pyplot as plt

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.input_to_hidden = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            # 激活函数选择
            # ReLu() [0, ∞]
            # Tanh() [-1, 1]
            # Sigmoid() [0, 1]
            nn.ReLU()
        )

        self.hidden_to_z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        # 模拟logvar 因为不用激活函数，可正可负
        self.hidden_to_z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.z_to_hidden = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.hidden_to_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dim),
            # 限制输出范围，根据输出的情况来,这里假定的是输出的图片像素值
            nn.Sigmoid()
        )

    def encode(self, x_in):
        x_hidden = self.input_to_hidden(x_in)
        z_mean = self.hidden_to_z_mean(x_hidden)
        z_logvar = self.hidden_to_z_logvar(x_hidden)
        return z_mean, z_logvar

    def decode(self, z_in):
        z_hidden = self.z_to_hidden(z_in)
        z_out = self.hidden_to_output(z_hidden)
        return z_out

    def reparameterize(self, mean, logvar):
        # 重参数化技巧
        # logvar = ln(std)**2
        std = torch.exp(0.5 * logvar)
        # torch.randn_like: 生成与std形状相同的torch张量，每一个值都从标准正态分布里取值
        sample_normal_distribution = torch.randn_like(std)
        # X = (Z - mean)/std
        return mean + sample_normal_distribution * std

    def forward(self, input):
        mean, logvar = self.encode(input)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z)
        return output, mean, logvar

    @staticmethod
    def loss_function(x, decode_x, mean, logvar):
        # loss1 x和decode_x的差异
        # 使用二元交叉熵损失
        BCE = nn.functional.binary_cross_entropy(decode_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD


# 检查数据集是否已经下载
if not os.path.exists('./data/MNIST'):
    # 加载MNIST数据集
    # transforms.Compose 组合多个图像的变换工具
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# train_dataset中的内容:
# [a][0]第a个dataset的像素点内容
# [a][1]第a个dataset对应的真实数字
# 将像素值重构为一列，即为输出的维度
data_input_dim = len(train_dataset[0][0].view(-1))
data_batch_size = 128

# shuffle:在每一个epoch开始之前，数据是否会先被打乱
# shuffle = True是一种提高训练效率的方法

train_loader = DataLoader(train_dataset, batch_size=data_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=data_batch_size, shuffle=False)

"""""""""
# enumerate(x),遍历x的内容，并获得当前的i索引
# (data,_),表示只关心data的内容，对其余内容不关心
for i, (data, _) in enumerate(train_loader):
    # 只查看第一个 batch 的第一个 data 的形状
    single_data = data[0]
    print(f"Single data shape: {single_data.shape}")
    break  # 只查看第一个 batch，查看后立即退出循环
"""

# 实例化模型并移动到GPU
model = VAE(input_dim=data_input_dim, hidden_dim=400, z_dim=20).to(device)

# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(10):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        #print("数据维度_1", torch.squeeze(data, dim = 1).view(data_batch_size, -1).shape)
        # 注意不能删除通道维度(1, 28, 28) to (28, 28)
        # print("数据维度_2", data.view(-1, data_input_dim).shape)
        # print("数据维度_3",data.view(data_batch_size, -1).shape )
        data = data.view(-1, data_input_dim).to(device)
        # data = data.view(data_batch_size, -1).to(device)
        # 训练数据不足data_batch_size时会报错
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data)
        loss = model.loss_function(data, recon_batch, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")
    print("\n")

# 测试
model.eval()

"""""""""
test_loss = 0
with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
        data = data.view(-1, 784).to(device)
        recon_batch, mean, logvar = model(data)
        test_loss += model.loss_function(data, recon_batch, mean, logvar).item()

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
"""

# 随机从 test_loader 中抽取一批数据
with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
        # 取第一张图像
        data = data[0].view(-1, data_input_dim).to(device)
        # 通过模型进行重构
        recon_batch, _, _ = model(data)

        # 由于只需要第一张图像，停止循环
        break

# 将数据从 GPU 移到 CPU 并转换回原始形状
data = data.view(28, 28).cpu()
recon_batch = recon_batch.view(28, 28).cpu()

# 可视化原始图像和重构图像
fig, axes = plt.subplots(1, 2)
axes[0].imshow(data, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(recon_batch, cmap='gray')
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')

plt.show()
