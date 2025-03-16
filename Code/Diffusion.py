import torch
import torch.nn as nn
import torch.optim as optim

# 设置设备：GPU 优先，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")


# 创建 alpha 调度 (variance schedule)
# beta_t逐渐增大，alpha_t逐渐减小
def create_alpha_schedule(timesteps, start=0.999, end=0.9):
    output = torch.linspace(start, end, timesteps, device=device)
    return output


# 创建去噪神经网络ϵϕ方法
class DenoisingNetwork(nn.Module):
    def __init__(self, z_dim, u_dim, hidden_dim):
        super(DenoisingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + u_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, z_t, u0, t):
        t_embed = torch.tensor([t], dtype=torch.float32, device=z_t.device).unsqueeze(0)
        _input = torch.cat([z_t, u0, t_embed], dim=-1)
        return self.net(_input)


# 创建前向扩散过程的类
class DiffusionModel:
    def __init__(self, timesteps, alpha_schedule):
        self.timesteps = timesteps
        self.alpha_schedule = alpha_schedule  # alpha_t
        self.alpha_bar = torch.cumprod(alpha_schedule, dim=0)  # _alpha_t
        self.sqrt_alpha_bar = torch.sqrt(torch.cumprod(alpha_schedule, dim=0))  # sqrt(_alpha_t)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.sqrt_alpha_bar ** 2)  # sqrt(1 - _alpha_t)

    def forward_process(self, z_0, t):
        noise = torch.randn_like(z_0, device=z_0.device)  # 生成高斯噪声
        alpha_t = self.sqrt_alpha_bar[t]
        one_minus_alpha_t = self.sqrt_one_minus_alpha_bar[t]
        z_t = alpha_t * z_0 + one_minus_alpha_t * noise
        return z_t, noise

    """""""""
    def generate_process(self, z_t, u0, t, predict_model):
        noise_t = predict_model(z_t, u0, t)
        predict_z_0 = (z_t - self.sqrt_one_minus_alpha_bar[t] * noise_t) / self.sqrt_alpha_bar[t]
        return predict_z_0
    """

    def backward_process(self, z_t, u0, t, predict_model):
        # 逐步实现从 z_t 到 z_0 的反向过程
        z_i = z_t
        for i in range(t, 1, -1):
            noise_i = predict_model(z_i, u0, i)
            step_noise = torch.randn_like(z_i)  # 标准正态分布的噪声

            # 计算后验方差：posterior_variance
            beta_i = 1 - self.alpha_schedule[i]  # 噪声强度
            posterior_variance_i = beta_i * (1 - self.alpha_bar[i - 1]) / (1 - self.alpha_bar[i])

            # 更新 z_i 的公式：反向采样
            z_i = (1 / torch.sqrt(self.alpha_schedule[i]) * (
                    z_i - (1 - self.alpha_schedule[i]) / self.sqrt_one_minus_alpha_bar[i] * noise_i)
                   + torch.sqrt(posterior_variance_i) * step_noise)

        return z_i


# 训练函数
def train(predict_model, diffusion_model, _optimizer, _num_epochs, z0, u0, timesteps):
    predict_model.train()
    for epoch in range(_num_epochs):
        epoch_loss = 0.0
        for t in range(timesteps):
            z_t, noise = diffusion_model.forward_process(z0, t)
            predicted_noise = predict_model(z_t, u0, t)
            loss = nn.MSELoss()(predicted_noise, noise)
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{_num_epochs}, Loss: {epoch_loss / timesteps}")


# 测试函数
def test(predict_model, diffusion_model, z0, u0, T):
    predict_model.eval()
    with torch.no_grad():
        z_T, z_t_noise = diffusion_model.forward_process(z0, T)
        #predict_z0 = diffusion_model.backward_process(z_T, u0, T, predict_model)
        predicted_noise = predict_model(z_T, u0, T)
        print(f"实际添加的噪声{z_t_noise}")
        print(f"网络预测的噪声{predicted_noise}")
        loss = torch.norm(z_t_noise - predicted_noise, p=2)
        print(f"L2 距离: {loss.item()}")


# 设置超参数
_z_dim = 10
_u_dim = 0
_hidden_dim = 128
_T = 1000
num_epochs = 30

_alpha_schedule = create_alpha_schedule(timesteps=_T)

# 初始化 z_start 和 u0
z_start = torch.randn((1, _z_dim), device=device)
_u0 = torch.randn((1, _u_dim), device=device)
print("开始训练时的 z 为:", z_start)
print("已知的 u0 为:", _u0)

# 实例化模型和优化器，并将模型移动到指定设备
_diffusion_model = DiffusionModel(_T, _alpha_schedule)
denoising_model = DenoisingNetwork(z_dim=_z_dim, u_dim=_u_dim, hidden_dim=_hidden_dim).to(device)
optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4)

# 开始训练
train(denoising_model, _diffusion_model, optimizer, num_epochs, z_start, _u0, _T)

# 测试模型
test(denoising_model, _diffusion_model, z_start, _u0, 100)
