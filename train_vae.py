import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from vae_model import VAE, vae_loss

def train_vae():
    """
    训练VAE模型的主函数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数设置
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 50
    latent_dim = 20  # 潜在空间维度
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
    ])
    
    # 加载MNIST数据集
    print("正在加载MNIST数据集...")
    try:
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请检查网络连接或手动下载MNIST数据集")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 初始化模型
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 训练历史记录
    train_losses = []
    test_losses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # 计算损失
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            # 反向传播
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # 打印训练进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += vae_loss(recon_batch, data, mu, logvar).item()
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        
        print(f'====> Epoch: {epoch} 平均损失 Train: {avg_train_loss:.4f} Test: {avg_test_loss:.4f}')
        
        # 每5个epoch保存模型和生成样本
        if epoch % 5 == 0:
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, f'./checkpoints/vae_epoch_{epoch}.pth')
            
            # 生成样本
            generate_and_save_samples(model, device, epoch)
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
    }, './checkpoints/vae_final.pth')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, test_losses)
    
    print("训练完成！")

def generate_and_save_samples(model, device, epoch):
    """
    生成样本并保存
    """
    model.eval()
    with torch.no_grad():
        # 生成新的手写数字
        samples = model.sample(64, device)
        samples = samples.cpu().view(64, 1, 28, 28)
        
        # 绘制生成的样本
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'./results/samples_epoch_{epoch}.png')
        plt.close()

def plot_training_curves(train_losses, test_losses):
    """
    绘制训练曲线
    """
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/training_curves.png')
    plt.close()

def visualize_latent_space(model, test_loader, device, num_samples=1000):
    """
    可视化潜在空间（二维情况下）
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, logvar = model.encoder(data)
            # 使用均值作为潜在表示
            latent_vectors.append(mu.cpu().numpy())
            labels.append(target.numpy())
            
            if len(latent_vectors) * len(data) >= num_samples:
                break
    
    latent_vectors = np.concatenate(latent_vectors)[:num_samples]
    labels = np.concatenate(labels)[:num_samples]
    
    # 如果潜在维度大于2，使用PCA降维到2D
    if latent_vectors.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors[:, :2]
    
    # 绘制潜在空间
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.savefig('./results/latent_space.png')
    plt.close()

if __name__ == '__main__':
    train_vae()