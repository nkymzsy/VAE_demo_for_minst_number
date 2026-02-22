import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from cvae_model import CVAE, cvae_loss

def train_cvae():
    """
    训练条件VAE模型的主函数
    支持按指定数字类别进行训练和生成
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数设置
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 50
    latent_dim = 20  # 潜在空间维度
    num_classes = 10  # MNIST有10个类别(0-9)
    
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
    model = CVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 训练历史记录
    train_losses = []
    test_losses = []
    
    print("开始训练条件VAE...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)  # 获取类别标签
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            
            # 计算损失
            loss = cvae_loss(recon_batch, data, mu, logvar)
            
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
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                recon_batch, mu, logvar = model(data, labels)
                test_loss += cvae_loss(recon_batch, data, mu, logvar).item()
        
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
            }, f'./checkpoints/cvae_epoch_{epoch}.pth')
            
            # 生成各样本数字
            generate_digit_samples(model, device, epoch)
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
    }, './checkpoints/cvae_final.pth')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, test_losses)
    
    print("条件VAE训练完成！")

def generate_digit_samples(model, device, epoch):
    """
    为每个数字生成样本并保存
    """
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Generated Samples at Epoch {epoch}', fontsize=16)
    
    with torch.no_grad():
        for digit in range(10):
            # 为每个数字生成8个样本
            samples = model.sample(8, target_labels=digit, device=device)
            samples = samples.cpu().view(8, 28, 28)
            
            # 显示第一个样本
            row = digit // 5
            col = digit % 5
            axes[row, col].imshow(samples[0], cmap='gray')
            axes[row, col].set_title(f'Digit {digit}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./results/cvae_samples_epoch_{epoch}.png')
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
    plt.title('CVAE Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/cvae_training_curves.png')
    plt.close()

def generate_specific_digits(model, target_digits, num_per_digit=5, device='cpu'):
    """
    生成指定数字的样本
    
    Args:
        model: 训练好的CVAE模型
        target_digits: 目标数字列表，如 [0, 1, 2] 或单个数字
        num_per_digit: 每个数字生成的样本数
        device: 设备类型
    """
    model.eval()
    
    # 处理输入
    if isinstance(target_digits, int):
        target_digits = [target_digits]
    
    total_samples = len(target_digits) * num_per_digit
    all_labels = []
    
    # 构建标签列表
    for digit in target_digits:
        all_labels.extend([digit] * num_per_digit)
    
    with torch.no_grad():
        # 一次性生成所有样本
        generated_images = model.sample(total_samples, target_labels=all_labels, device=device)
        generated_images = generated_images.cpu().view(total_samples, 28, 28)
    
    # 可视化结果
    rows = len(target_digits)
    cols = num_per_digit
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle('Generated Specific Digits', fontsize=16)
    
    for i, digit in enumerate(target_digits):
        for j in range(num_per_digit):
            idx = i * num_per_digit + j
            if rows == 1:
                axes[j].imshow(generated_images[idx], cmap='gray')
                axes[j].set_title(f'Digit {digit}')
                axes[j].axis('off')
            else:
                axes[i, j].imshow(generated_images[idx], cmap='gray')
                axes[i, j].set_title(f'Digit {digit}')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/generated_specific_digits.png')
    plt.show()
    
    return generated_images

def main():
    """
    主函数：训练模型并演示生成功能
    """
    # 训练模型
    train_cvae()
    
    # 演示生成特定数字
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练好的模型
    model = CVAE(latent_dim=20, num_classes=10).to(device)
    checkpoint = torch.load('./checkpoints/cvae_final.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 生成指定数字的样本
    print("生成指定数字样本...")
    target_digits = [3, 7, 9]  # 生成数字3, 7, 9
    generate_specific_digits(model, target_digits, num_per_digit=6, device=device)

if __name__ == '__main__':
    main()