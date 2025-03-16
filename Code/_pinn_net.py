import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

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

"""""""""
# 定义Poisson方程中的N(u)
def N(x):
    return -torch.sin(np.pi * x)
"""


def exact_solution(d, w0, t):
    # 定义欠阻尼谐振子问题的解析解
    assert d < w0  # 确保阻尼系数d小于自然频率w0
    w = np.sqrt(w0**2 - d**2)  # 计算欠阻尼频率
    phi = np.arctan(-d/w)  # 计算初始相位
    A = 1/(2*np.cos(phi))  # 计算振幅
    cos = torch.cos(phi + w * t)  # 计算余弦项
    exp = torch.exp(-d * t)  # 计算指数衰减项
    u = exp * 2 * A * cos  # 计算解
    return u


# PINN神经网络模型
class PINN(nn.Module):
    def __init__(self, N_function=None):
        super(PINN, self).__init__()
        # F:u(x, t) + N(u) = 0
        # 定义N(u)
        self.N_function = N_function
        self.hidden_layers = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.hidden_layers(x)

    # 计算物理损失
    def physics_loss(self, x):
        # 训练过程
        d, w0 = 2, 20  # 定义某些物理参数
        mu, k = 2 * d, w0 ** 2  # 根据给定的物理参数计算mu和k
        # MSE_f
        x = x.requires_grad_(True)  # 确保x的梯度被计算
        u = self.forward(x)
        # 计算一阶导数的统一模式
        u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        #n_x = self.N_function(x).to(device)  # 使用传入的N(x)
        # MSE(f):对应文献的内容:F:u(x) + N(u) = 0
        #loss = torch.mean((u_xx - n_x) ** 2)
        loss = 1e-4 * torch.mean((u_xx + mu * u_x + k * u)**2)
        return loss


# 训练PINN模型
model = PINN(N_function=None).to(device)

# 生成训练数据
x_train = torch.linspace(0, 1, 100).view(-1, 1).to(device)


# 边界条件（u(0)=1, du(0)=0）
def boundary_condition(model):
    # 返回神经网络在x = 0
    x = torch.tensor([[0.0]], device=device, requires_grad=True)
    u_0 = model(x)
    return u_0

def d_boundary_condition(model):
    x = torch.tensor([[0.0]], device=device, requires_grad=True)
    output = model(x)
    u_1 = autograd.grad(outputs=output, inputs=x, create_graph=True)[0]
    return u_1

# 定义损失函数，包括物理损失和边界条件损失
def loss_fn(model, x):
    # 得到MSE_f
    physics_loss = model.physics_loss(x)
    # 返回神经网络在边界的输出
    u_0= boundary_condition(model)
    u_1 = d_boundary_condition(model)
    # MSE_u:(u(0) - 1)^2 + (du(1) - 0)^2
    boundary_loss = (u_0 - 1) ** 2 + 1e-1 * (u_1 - 0) ** 2
    return physics_loss + boundary_loss


# 使用 L-BFGS 优化器
"""
model.parameters():需要优化的模型
lr:学习率,更新的步长大小
max_iter:迭代次数
history_size:历史窗口大小
tolerance_grad:梯度容差,终止更新的梯度范数
tolerance_change:变化容差,终止更新的目标函数变化值
line_search_fn:线搜索函数，用于确定每步更新时的步长
"""
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=50000, history_size=50, tolerance_grad=1e-7,
                              tolerance_change=1e-9,
                              line_search_fn='strong_wolfe')

# counter
step_count = 0


# closure function of L-BFGS
def closure():
    global step_count
    # interval of each loss-printing
    print_interval = 1000
    optimizer.zero_grad()  # clear grad
    loss_value = loss_fn(model, x_train)
    loss_value.backward()  # calculate grad

    # print loss
    if step_count < print_interval:
        print(f'Iteration {step_count}: Loss = {loss_value.item()}')

    step_count += 1
    return loss_value


# 训练循环
optimizer.step(closure)  # 在调用 step 时传入 closure 函数

# 预测和可视化结果
x_test = torch.linspace(0, 1, 100).view(-1, 1).to(device)
u_prd = model(x_test).cpu().detach().numpy()

plt.plot(x_test.cpu(), u_prd, label='Predicted u(x)')
plt.plot(x_test.cpu(), exact_solution(2, 20, x_test.cpu()), label='Analytical u(x)')
plt.legend()
plt.show()