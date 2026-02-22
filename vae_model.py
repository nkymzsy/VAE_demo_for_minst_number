import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """
    VAE编码器网络
    将输入图像编码为潜在空间的均值和方差
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAEEncoder, self).__init__()
        # 全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 输出层：分别计算均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)  # 对数方差
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像，形状 [batch_size, 784]
        Returns:
            mu: 均值向量，形状 [batch_size, latent_dim]
            logvar: 对数方差向量，形状 [batch_size, latent_dim]
        """
        # 展平输入图像
        x = x.view(x.size(0), -1)
        
        # 编码过程
        h = F.relu(self.fc1(x))      # 第一层激活
        h = F.relu(self.fc2(h))      # 第二层激活
        
        # 计算均值和对数方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """
    VAE解码器网络
    将潜在向量解码回图像空间
    """
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(VAEDecoder, self).__init__()
        # 全连接层
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        """
        前向传播
        Args:
            z: 潜在向量，形状 [batch_size, latent_dim]
        Returns:
            reconstructed: 重构图像，形状 [batch_size, 784]
        """
        # 解码过程
        h = F.relu(self.fc1(z))      # 第一层激活
        h = F.relu(self.fc2(h))      # 第二层激活
        reconstructed = torch.sigmoid(self.fc3(h))  # 输出层使用sigmoid激活
        
        return reconstructed

class VAE(nn.Module):
    """
    完整的VAE模型
    包含编码器和解码器
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 初始化编码器和解码器
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧
        从N(mu, var)分布中采样，通过epsilon ~ N(0,1)实现
        这样可以使得梯度能够反向传播
        
        Args:
            mu: 均值，形状 [batch_size, latent_dim]
            logvar: 对数方差，形状 [batch_size, latent_dim]
        Returns:
            z: 采样的潜在向量，形状 [batch_size, latent_dim]
        """
        # 计算标准差
        std = torch.exp(0.5 * logvar)
        
        # 从标准正态分布采样
        eps = torch.randn_like(std)
        
        # 重参数化：z = mu + std * epsilon
        return mu + eps * std
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像，形状 [batch_size, 1, 28, 28]
        Returns:
            recon_x: 重构图像，形状 [batch_size, 784]
            mu: 均值向量，形状 [batch_size, latent_dim]
            logvar: 对数方差向量，形状 [batch_size, latent_dim]
        """
        # 编码：获取均值和方差
        mu, logvar = self.encoder(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码：重构图像
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples, device):
        """
        从标准正态分布中采样并生成新图像
        Args:
            num_samples: 采样数量
            device: 设备类型 ('cuda' 或 'cpu')
        Returns:
            generated_images: 生成的图像，形状 [num_samples, 784]
        """
        # 从标准正态分布采样潜在向量
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # 解码生成图像
        generated_images = self.decoder(z)
        
        return generated_images

def vae_loss(recon_x, x, mu, logvar):
    """
    VAE损失函数
    包含两部分：重构损失 + KL散度损失
    
    Args:
        recon_x: 重构图像，形状 [batch_size, 784]
        x: 原始图像，形状 [batch_size, 784]
        mu: 均值向量，形状 [batch_size, latent_dim]
        logvar: 对数方差向量，形状 [batch_size, latent_dim]
        
    Returns:
        total_loss: 总损失值
    """
    # 1. 重构损失：使用二元交叉熵
    # 衡量重构图像与原始图像之间的差异
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL散度损失：衡量q(z|x)与p(z)之间的距离
    # 推导后的解析形式：KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 = 重构损失 + KL散度损失
    # 重构损失鼓励模型准确重建输入
    # KL散度损失鼓励潜在分布接近标准正态分布
    total_loss = recon_loss + kl_loss
    
    return total_loss

# 测试代码
if __name__ == '__main__':
    # 创建模型实例
    model = VAE(latent_dim=20)
    
    # 创建测试输入
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    
    # 测试前向传播
    recon_x, mu, logvar = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"重构输出形状: {recon_x.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"对数方差形状: {logvar.shape}")
    
    # 测试损失函数
    loss = vae_loss(recon_x, test_input, mu, logvar)
    print(f"损失值: {loss.item()}")
    
    # 测试采样功能
    samples = model.sample(16, 'cpu')
    print(f"采样输出形状: {samples.shape}")