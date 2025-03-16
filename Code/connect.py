import paramiko

# 设置 SSH 连接信息
hostname = '192.168.193.129'  # 虚拟机的 IP 地址
port = 22  # 默认 SSH 端口
username = 'kismet22'  # 虚拟机的用户名
password = 'zhengshizhan0822'  # 虚拟机的密码

# 创建 SSH 客户端对象
client = paramiko.SSHClient()

# 自动添加主机密钥
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# 连接到虚拟机
client.connect(hostname, port=port, username=username, password=password)

# 执行启动 xvfb 的命令
stdin, stdout, stderr = client.exec_command('Xvfb :99 -screen 0 1024x768x24 &')

# 因为后台进程启动后不会马上有输出，可以等一段时间
stdin.close()  # 关闭 stdin

# 获取命令输出和错误输出
output = stdout.read().decode()
error_output = stderr.read().decode()

if output:
    print("输出信息:", output)
if error_output:
    print("错误信息:", error_output)

# 关闭 SSH 连接
client.close()
