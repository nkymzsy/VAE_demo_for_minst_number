import torch
import matplotlib.pyplot as plt
import numpy as np
from vae_model import VAE

def load_trained_model(checkpoint_path, latent_dim=20, device='cpu'):
    """
    加载训练好的VAE模型
    
    Args:
        checkpoint_path: 模型检查点路径
        latent_dim: 潜在空间维度
        device: 设备类型
        
    Returns:
        model: 加载好的模型
    """
    # 创建模型实例
    model = VAE(latent_dim=latent_dim).to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    print(f"成功加载模型，训练轮次: {checkpoint.get('epoch', 'unknown')}")
    return model

def generate_digits(model, num_samples=16, device='cpu'):
    """
    生成手写数字图像
    
    Args:
        model: 训练好的VAE模型
        num_samples: 生成样本数量
        device: 设备类型
        
    Returns:
        generated_images: 生成的图像数组
    """
    with torch.no_grad():
        # 从标准正态分布采样潜在向量
        generated_images = model.sample(num_samples, device)
        generated_images = generated_images.cpu().view(num_samples, 28, 28)
    
    return generated_images

def interpolate_latent_space(model, digit1_idx=0, digit2_idx=1, steps=10, device='cpu'):
    """
    在两个数字的潜在空间之间进行插值
    
    Args:
        model: 训练好的VAE模型
        digit1_idx: 第一个数字的索引
        digit2_idx: 第二个数字的索引
        steps: 插值步数
        device: 设备类型
        
    Returns:
        interpolated_images: 插值生成的图像
    """
    # 生成两个随机潜在向量
    z1 = torch.randn(1, model.latent_dim).to(device)
    z2 = torch.randn(1, model.latent_dim).to(device)
    
    # 在两个点之间线性插值
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in alphas:
            # 线性插值: z = (1-alpha) * z1 + alpha * z2
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = model.decoder(z_interp)
            interpolated_images.append(img.cpu().view(28, 28))
    
    return torch.stack(interpolated_images)

def visualize_generated_digits(images, title="Generated Digits"):
    """
    可视化生成的手写数字
    
    Args:
        images: 图像张量，形状 [num_samples, 28, 28]
        title: 图标题
    """
    num_samples = len(images)
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle(title, fontsize=16)
    
    # 处理单行或多行的情况
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < num_samples:
            axes[row, col].imshow(images[i], cmap='gray')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_interpolation(images, title="Latent Space Interpolation"):
    """
    可视化潜在空间插值结果
    
    Args:
        images: 插值图像序列
        title: 图标题
    """
    steps = len(images)
    
    fig, axes = plt.subplots(1, steps, figsize=(steps*1.5, 2))
    fig.suptitle(title, fontsize=14)
    
    for i in range(steps):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Step {i+1}')
    
    plt.tight_layout()
    plt.savefig(f'./results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_latent_distribution(model, num_samples=1000, device='cpu'):
    """
    分析潜在空间的分布特性
    
    Args:
        model: 训练好的VAE模型
        num_samples: 样本数量
        device: 设备类型
    """
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        # 生成多个潜在向量
        for _ in range(num_samples):
            z = torch.randn(1, model.latent_dim).to(device)
            latent_vectors.append(z.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # 统计每个维度的均值和方差
    means = np.mean(latent_vectors, axis=0)
    stds = np.std(latent_vectors, axis=0)
    
    print("潜在空间统计信息:")
    print(f"维度数: {model.latent_dim}")
    print(f"样本数: {num_samples}")
    print(f"各维度均值范围: [{np.min(means):.4f}, {np.max(means):.4f}]")
    print(f"各维度标准差范围: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
    
    # 可视化分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(model.latent_dim), means)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Mean')
    plt.title('Mean of Each Latent Dimension')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(model.latent_dim), stds)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Std of Each Latent Dimension')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/latent_distribution_analysis.png')
    plt.show()

def main():
    """
    主函数：演示VAE的各种功能
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    latent_dim = 20
    checkpoint_path = './checkpoints/vae_final.pth'
    
    try:
        # 加载训练好的模型
        print("正在加载模型...")
        model = load_trained_model(checkpoint_path, latent_dim, device)
        
        # 生成随机手写数字
        print("生成手写数字样本...")
        generated_digits = generate_digits(model, num_samples=25, device=device)
        visualize_generated_digits(generated_digits, "Randomly Generated Digits")
        
        # 潜在空间插值
        print("执行潜在空间插值...")
        interpolated_digits = interpolate_latent_space(model, steps=8, device=device)
        visualize_interpolation(interpolated_digits, "Latent Space Interpolation")
        
        # 分析潜在空间分布
        print("分析潜在空间分布...")
        analyze_latent_distribution(model, num_samples=500, device=device)
        
        print("所有操作完成！结果已保存到 ./results 目录")
        
    except FileNotFoundError:
        print(f"找不到模型文件: {checkpoint_path}")
        print("请先运行 train_vae.py 训练模型")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()