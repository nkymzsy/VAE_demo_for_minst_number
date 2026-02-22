import torch
import matplotlib.pyplot as plt
import numpy as np
from cvae_model import CVAE

def load_trained_cvae(checkpoint_path, latent_dim=20, num_classes=10, device='cpu'):
    """
    加载训练好的条件VAE模型
    
    Args:
        checkpoint_path: 模型检查点路径
        latent_dim: 潜在空间维度
        num_classes: 类别数量
        device: 设备类型
        
    Returns:
        model: 加载好的模型
    """
    # 创建模型实例
    model = CVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    print(f"成功加载条件VAE模型，训练轮次: {checkpoint.get('epoch', 'unknown')}")
    return model

def generate_specific_numbers(model, target_numbers, samples_per_number=8, device='cpu'):
    """
    生成指定的数字
    
    Args:
        model: 训练好的CVAE模型
        target_numbers: 目标数字列表，如 [0, 1, 2, 3]
        samples_per_number: 每个数字生成的样本数
        device: 设备类型
        
    Returns:
        generated_images: 生成的图像数组
        labels: 对应的标签
    """
    model.eval()
    
    # 构建完整的标签列表
    all_labels = []
    for number in target_numbers:
        all_labels.extend([number] * samples_per_number)
    
    total_samples = len(all_labels)
    
    with torch.no_grad():
        # 生成图像
        generated_images = model.sample(total_samples, target_labels=all_labels, device=device)
        generated_images = generated_images.cpu().view(total_samples, 28, 28)
    
    return generated_images, torch.tensor(all_labels)

def visualize_specific_generation(images, labels, title="Specific Number Generation"):
    """
    可视化指定数字的生成结果
    
    Args:
        images: 生成的图像张量
        labels: 对应的标签
        title: 图标题
    """
    unique_labels = torch.unique(labels)
    samples_per_label = (labels == unique_labels[0]).sum().item()
    
    rows = len(unique_labels)
    cols = samples_per_label
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    fig.suptitle(title, fontsize=16)
    
    for i, digit in enumerate(unique_labels):
        digit_mask = (labels == digit)
        digit_images = images[digit_mask]
        
        for j in range(samples_per_label):
            if rows == 1:
                axes[j].imshow(digit_images[j], cmap='gray')
                axes[j].set_title(f'{digit.item()}', fontsize=12)
                axes[j].axis('off')
            else:
                axes[i, j].imshow(digit_images[j], cmap='gray')
                axes[i, j].set_title(f'{digit.item()}', fontsize=12)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_single_number_grid(model, target_number, grid_size=5, device='cpu'):
    """
    生成单个数字的网格图
    
    Args:
        model: 训练好的CVAE模型
        target_number: 目标数字 (0-9)
        grid_size: 网格大小 (grid_size x grid_size)
        device: 设备类型
    """
    total_samples = grid_size * grid_size
    
    with torch.no_grad():
        generated_images = model.sample(total_samples, target_labels=target_number, device=device)
        generated_images = generated_images.cpu().view(total_samples, 28, 28)
    
    # 绘制网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    fig.suptitle(f'Generated Number {target_number} ({grid_size}x{grid_size} Grid)', fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(generated_images[idx], cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./results/number_{target_number}_grid_{grid_size}x{grid_size}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def interpolate_between_numbers(model, num1, num2, steps=8, device='cpu'):
    """
    在两个不同数字之间进行潜在空间插值
    
    Args:
        model: 训练好的CVAE模型
        num1: 起始数字
        num2: 结束数字
        steps: 插值步数
        device: 设备类型
    """
    model.eval()
    
    # 为两个数字生成潜在向量
    with torch.no_grad():
        # 生成多个样本取平均作为代表
        z1_samples = []
        z2_samples = []
        
        for _ in range(10):  # 生成10个样本取平均
            z1_temp = torch.randn(1, model.latent_dim).to(device)
            z2_temp = torch.randn(1, model.latent_dim).to(device)
            z1_samples.append(z1_temp)
            z2_samples.append(z2_temp)
        
        z1 = torch.mean(torch.stack(z1_samples), dim=0)
        z2 = torch.mean(torch.stack(z2_samples), dim=0)
        
        # 插值生成
        alphas = torch.linspace(0, 1, steps).to(device)
        interpolated_images = []
        
        for alpha in alphas:
            # 线性插值: z = (1-alpha) * z1 + alpha * z2
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # 逐渐改变标签权重（简化处理：固定使用起始数字的标签）
            # 实际应用中可以更复杂地插值标签
            interp_label = torch.tensor([num1], device=device)
            img = model.decoder(z_interp, interp_label)
            interpolated_images.append(img.cpu().view(28, 28))
    
    return torch.stack(interpolated_images)

def visualize_interpolation(images, num1, num2, title=None):
    """
    可视化数字间插值结果
    """
    if title is None:
        title = f'Interpolation from {num1} to {num2}'
    
    steps = len(images)
    fig, axes = plt.subplots(1, steps, figsize=(steps*1.5, 2))
    fig.suptitle(title, fontsize=14)
    
    for i in range(steps):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title(f'{num1}')
        elif i == steps-1:
            axes[i].set_title(f'{num2}')
        else:
            axes[i].set_title(f'Step {i+1}')
    
    plt.tight_layout()
    plt.savefig(f'./results/interpolation_{num1}_to_{num2}.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_all_digits(model, samples_per_digit=3, device='cpu'):
    """
    生成并比较所有数字（0-9）
    """
    all_digits = list(range(10))
    images, labels = generate_specific_numbers(
        model, all_digits, samples_per_digit, device
    )
    
    visualize_specific_generation(images, labels, "All Digits Comparison")

def main():
    """
    主函数：演示条件VAE的各种生成功能
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    latent_dim = 20
    num_classes = 10
    checkpoint_path = './checkpoints/cvae_final.pth'
    
    try:
        # 加载训练好的模型
        print("正在加载条件VAE模型...")
        model = load_trained_cvae(checkpoint_path, latent_dim, num_classes, device)
        
        # 1. 生成指定的几个数字
        print("生成指定数字样本...")
        target_numbers = [1, 4, 7, 9]
        images, labels = generate_specific_numbers(model, target_numbers, 6, device)
        visualize_specific_generation(images, labels, "Selected Numbers Generation")
        
        # 2. 生成单个数字的网格
        print("生成数字3的网格...")
        generate_single_number_grid(model, target_number=3, grid_size=6, device=device)
        
        # 3. 数字间插值
        print("执行数字插值...")
        interp_images = interpolate_between_numbers(model, 1, 8, steps=10, device=device)
        visualize_interpolation(interp_images, 1, 8)
        
        # 4. 比较所有数字
        print("生成所有数字对比...")
        compare_all_digits(model, samples_per_digit=4, device=device)
        
        print("所有生成任务完成！结果已保存到 ./results 目录")
        
    except FileNotFoundError:
        print(f"找不到模型文件: {checkpoint_path}")
        print("请先运行 train_cvae.py 训练模型")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()