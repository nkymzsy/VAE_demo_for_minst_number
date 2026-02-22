import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAEEncoder(nn.Module):
    """
    条件VAE编码器网络
    将输入图像和类别标签一起编码为潜在空间的均值和方差
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(CVAEEncoder, self).__init__()
        self.num_classes = num_classes
        
        # 输入层：图像展平 + 类别标签one-hot编码
        self.input_dim_with_label = input_dim + num_classes
        
        # 全连接层
        self.fc1 = nn.Linear(self.input_dim_with_label, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 输出层：分别计算均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)  # 对数方差
        
    def forward(self, x, labels):
        """
        前向传播
        Args:
            x: 输入图像，形状 [batch_size, 1, 28, 28]
            labels: 类别标签，形状 [batch_size] 或 [batch_size, num_classes]
        Returns:
            mu: 均值向量，形状 [batch_size, latent_dim]
            logvar: 对数方差向量，形状 [batch_size, latent_dim]
        """
        # 展平输入图像
        x_flat = x.view(x.size(0), -1)  # [batch_size, 784]
        
        # 处理标签：转换为one-hot编码
        if labels.dim() == 1:  # 如果是类别索引
            labels_onehot = F.one_hot(labels, self.num_classes).float()
        else:  # 如果已经是one-hot编码
            labels_onehot = labels
            
        # 拼接图像和标签信息
        x_with_label = torch.cat([x_flat, labels_onehot], dim=1)
        
        # 编码过程
        h = F.relu(self.fc1(x_with_label))  # 第一层激活
        h = F.relu(self.fc2(h))             # 第二层激活
        
        # 计算均值和对数方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class CVAEDecoder(nn.Module):
    """
    条件VAE解码器网络
    将潜在向量和类别标签解码回图像空间
    """
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784, num_classes=10):
        super(CVAEDecoder, self).__init__()
        self.num_classes = num_classes
        
        # 输入层：潜在向量 + 类别标签one-hot编码
        self.latent_dim_with_label = latent_dim + num_classes
        
        # 全连接层
        self.fc1 = nn.Linear(self.latent_dim_with_label, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, labels):
        """
        前向传播
        Args:
            z: 潜在向量，形状 [batch_size, latent_dim]
            labels: 类别标签，形状 [batch_size] 或 [batch_size, num_classes]
        Returns:
            reconstructed: 重构图像，形状 [batch_size, 784]
        """
        # 处理标签：转换为one-hot编码
        if labels.dim() == 1:  # 如果是类别索引
            labels_onehot = F.one_hot(labels, self.num_classes).float()
        else:  # 如果已经是one-hot编码
            labels_onehot = labels
            
        # 拼接潜在向量和标签信息
        z_with_label = torch.cat([z, labels_onehot], dim=1)
        
        # 解码过程
        h = F.relu(self.fc1(z_with_label))  # 第一层激活
        h = F.relu(self.fc2(h))             # 第二层激活
        reconstructed = torch.sigmoid(self.fc3(h))  # 输出层使用sigmoid激活
        
        return reconstructed

class CVAE(nn.Module):
    """
    完整的条件VAE模型
    支持根据指定类别生成对应的图像
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 初始化编码器和解码器
        self.encoder = CVAEEncoder(input_dim, hidden_dim, latent_dim, num_classes)
        self.decoder = CVAEDecoder(latent_dim, hidden_dim, input_dim, num_classes)
        
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
    
    def forward(self, x, labels):
        """
        前向传播
        Args:
            x: 输入图像，形状 [batch_size, 1, 28, 28]
            labels: 类别标签，形状 [batch_size]
        Returns:
            recon_x: 重构图像，形状 [batch_size, 784]
            mu: 均值向量，形状 [batch_size, latent_dim]
            logvar: 对数方差向量，形状 [batch_size, latent_dim]
        """
        # 编码：获取均值和方差
        mu, logvar = self.encoder(x, labels)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码：重构图像
        recon_x = self.decoder(z, labels)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples, target_labels, device):
        """
        根据指定标签生成新图像
        Args:
            num_samples: 采样数量
            target_labels: 目标类别标签，形状 [num_samples] 或具体数字
            device: 设备类型 ('cuda' 或 'cpu')
        Returns:
            generated_images: 生成的图像，形状 [num_samples, 784]
        """
        # 处理标签输入
        if isinstance(target_labels, int):  # 如果输入单个数字
            labels = torch.full((num_samples,), target_labels, dtype=torch.long, device=device)
        elif isinstance(target_labels, list):  # 如果输入列表
            labels = torch.tensor(target_labels, dtype=torch.long, device=device)
        else:  # 如果已经是tensor
            labels = target_labels.to(device)
        
        # 从标准正态分布采样潜在向量
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # 解码生成图像
        generated_images = self.decoder(z, labels)
        
        return generated_images

def cvae_loss(recon_x, x, mu, logvar):
    """
    CVAE损失函数
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
    
    # 2. KL散度损失：衡量q(z|x,y)与p(z|y)之间的距离
    # 推导后的解析形式：KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 = 重构损失 + KL散度损失
    total_loss = recon_loss + kl_loss
    
    return total_loss

# 测试代码
if __name__ == '__main__':
    # 创建模型实例
    model = CVAE(latent_dim=20, num_classes=10)
    
    # 创建测试输入
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    test_labels = torch.randint(0, 10, (batch_size,))  # 随机标签 0-9
    
    # 测试前向传播
    recon_x, mu, logvar = model(test_input, test_labels)
    
    print(f"输入形状: {test_input.shape}")
    print(f"标签形状: {test_labels.shape}")
    print(f"重构输出形状: {recon_x.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"对数方差形状: {logvar.shape}")
    
    # 测试损失函数
    loss = cvae_loss(recon_x, test_input, mu, logvar)
    print(f"损失值: {loss.item()}")
    
    # 测试指定数字生成
    generated_ones = model.sample(16, target_labels=1, device='cpu')
    print(f"生成数字1的输出形状: {generated_ones.shape}")
    
    # 测试批量生成不同数字
    target_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2  # 生成每个数字2个样本
    generated_mixed = model.sample(20, target_labels=target_digits, device='cpu')
    print(f"混合生成输出形状: {generated_mixed.shape}")