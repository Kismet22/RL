import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torchvision import datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置设备：GPU 优先，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")


# 1. 创建 alpha 调度 (variance schedule)
def create_alpha_schedule(timesteps, start=0.999, end=0.9):
    return torch.linspace(start, end, timesteps, device=device)


# 2. 创建去噪神经网络ϵϕ方法
class DenoisingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(DenoisingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, z_t, t):
        t_embed = torch.ones_like(z_t[:, 0:1]) * t  # 批量时间嵌入
        _input = torch.cat([z_t, t_embed], dim=-1)
        return self.net(_input)


# 3. 创建前向扩散过程的类
class DiffusionModel:
    def __init__(self, timesteps, alpha_schedule):
        self.timesteps = timesteps
        self.alpha_schedule = alpha_schedule  # alpha_t
        self.alpha_bar = torch.cumprod(alpha_schedule, dim=0)  # _alpha_t
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)  # sqrt(_alpha_t)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)  # sqrt(1 - _alpha_t)

    def forward_process(self, z_0, t):
        noise = torch.randn_like(z_0, device=z_0.device)  # 生成高斯噪声
        alpha_t = self.sqrt_alpha_bar[t]
        one_minus_alpha_t = self.sqrt_one_minus_alpha_bar[t]
        z_t = alpha_t * z_0 + one_minus_alpha_t * noise
        return z_t, noise

    def backward_process(self, z_t, t, predict_model):
        z_i = z_t
        for i in range(t, 1, -1):
            noise_i = predict_model(z_i, i)
            step_noise = torch.randn_like(z_i)  # 标准正态分布的噪声
            beta_i = 1 - self.alpha_schedule[i]  # 噪声强度
            posterior_variance_i = beta_i * (1 - self.alpha_bar[i - 1]) / (1 - self.alpha_bar[i])
            z_i = (1 / torch.sqrt(self.alpha_schedule[i]) * (
                    z_i - (1 - self.alpha_schedule[i]) / self.sqrt_one_minus_alpha_bar[i] * noise_i)
                   + torch.sqrt(posterior_variance_i) * step_noise)
        return z_i


# 4. 图像数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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
"""""""""
# enumerate(x),遍历x的内容，并获得当前的i索引
# (data,_),表示只关心data的内容，对其余内容不关心
for i, (data, _) in enumerate(train_loader):
    # 只查看第一个 batch 的第一个 data 的形状
    single_data = data[0]
    print(f"Single data shape: {single_data.shape}")
    break  # 只查看第一个 batch，查看后立即退出循环
"""

train_loader = DataLoader(train_dataset, batch_size=data_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=data_batch_size, shuffle=False)

# 7. 参数设定
_T = 10  # 总时间步
_z_dim = 28 * 28  # 图像展开的维度
_hidden_dim = 128  # 隐藏层维度
num_epochs = 10  # 训练 epoch
_alpha_schedule = create_alpha_schedule(_T)

# 8. 初始化模型和优化器
_diffusion_model = DiffusionModel(_T, _alpha_schedule)
denoising_model = DenoisingNetwork(z_dim=_z_dim, hidden_dim=_hidden_dim).to(device)
optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4)


# 训练函数
def train_diffusion(predict_model, diffusion_model, optimizer, num_epochs, train_loader, timesteps, device):
    predict_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            # 将图像扁平化为向量并移动到设备
            # a.view(a.size(0), -1) 按a的第一个维度([x, y, z, w]中的x维度铺开，-1代表自动计算后三个维度大小，若不为-1，
            # 代表自定义指定的维度)
            z_start = images.view(images.size(0), -1).to(device)

            for t in range(timesteps):
                z_t, noise = diffusion_model.forward_process(z_start, t)
                predicted_noise = predict_model(z_t, t)
                loss = nn.MSELoss()(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(images):.6f}")
        print("\n")


# 开始训练
train_diffusion(denoising_model, _diffusion_model, optimizer, num_epochs, train_loader, _T, device)


# 定义函数：展示原始图像、加噪后的图像和去噪后的图像
def visualize_diffusion_process(original_images, noisy_images, denoised_images, num_images=6):
    fig, axes = plt.subplots(3, num_images, figsize=(12, 6))

    # 显示原始图像
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].view(28, 28).cpu().detach().numpy(), cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

    # 显示加噪后的图像
    for i in range(num_images):
        axes[1, i].imshow(noisy_images[i].view(28, 28).cpu().detach().numpy(), cmap="gray")
        axes[1, i].set_title("Noisy")
        axes[1, i].axis("off")

    # 显示去噪后的图像
    for i in range(num_images):
        axes[2, i].imshow(denoised_images[i].view(28, 28).cpu().detach().numpy(), cmap="gray")
        axes[2, i].set_title("Denoised")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()


# 测试扩散模型
def test_diffusion(predict_model, diffusion_model, test_loader, timesteps, device):
    predict_model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不进行梯度计算

        # 取一个 batch 的测试数据
        for images, _ in test_loader:
            original_images = images.view(images.size(0), -1).to(device)  # 将图像展平
            break  # 只用一个 batch 的数据进行可视化

        # 加噪过程
        t = timesteps - 1  # 使用最大时间步 t
        noisy_images, _ = diffusion_model.forward_process(original_images, t)

        # 反向去噪过程
        denoised_images = diffusion_model.backward_process(noisy_images, t, predict_model)

        # 可视化：原始图像、加噪后的图像和去噪后的图像
        visualize_diffusion_process(original_images, noisy_images, denoised_images)


# 在测试集上测试扩散模型
test_diffusion(denoising_model, _diffusion_model, test_loader, _T, device)
