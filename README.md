# VAE手写数字生成网络

这是一个基于PyTorch实现的变分自编码器(VAE)和条件变分自编码器(CVAE)项目，用于生成MNIST手写数字。

## 项目结构

```
VAE/
├── vae_model.py        # 标准VAE模型实现
├── cvae_model.py       # 条件VAE模型实现（支持指定数字生成）
├── train_vae.py        # 标准VAE训练脚本
├── train_cvae.py       # 条件VAE训练脚本
├── inference_vae.py    # 标准VAE推理脚本
├── inference_cvae.py   # 条件VAE推理脚本
├── requirements.txt    # 项目依赖
└── README.md          # 说明文档
```

## 功能特点

### 标准VAE (vae_model.py)
- 自动生成手写数字
- 学习数据的潜在表示
- 支持潜在空间可视化

### 条件VAE (cvae_model.py) 
- **指定数字生成**：可以明确指定要生成哪个数字(0-9)
- **条件控制**：通过类别标签控制生成内容
- **数字间插值**：在不同数字间进行平滑过渡
- **批量生成**：一次生成多个指定数字

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

#### 训练标准VAE
```bash
python train_vae.py
```

#### 训练条件VAE（推荐）
```bash
python train_cvae.py
```

### 2. 生成数字

#### 使用标准VAE生成（随机数字）
```bash
python inference_vae.py
```

#### 使用条件VAE生成指定数字
```bash
python inference_cvae.py
```

### 3. 编程方式调用

```python
import torch
from cvae_model import CVAE

# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CVAE(latent_dim=20, num_classes=10).to(device)
# ... 加载检查点 ...

# 生成指定数字
generated_images = model.sample(
    num_samples=16,           # 生成16个样本
    target_labels=3,          # 生成数字3
    device=device
)

# 生成多个不同数字
target_digits = [1, 4, 7, 9]  # 生成数字1,4,7,9
images = model.sample(
    num_samples=len(target_digits) * 4,  # 每个数字4个样本
    target_labels=target_digits * 4,     # 重复标签
    device=device
)
```

## 模型参数说明

- `latent_dim`: 潜在空间维度（默认20）
- `hidden_dim`: 隐藏层维度（默认400）
- `num_classes`: 类别数量（MNIST为10）
- `input_dim`: 输入图像维度（MNIST为784）

## 输出文件

训练和推理结果保存在：
- `./checkpoints/`: 模型检查点
- `./results/`: 生成的图像和可视化结果

## 性能提示

- 使用GPU训练可以显著加速
- 建议训练50-100个epoch获得较好效果
- 潜在维度可根据需要调整（10-50效果较好）

## 学习资源

这个项目适合学习：
- 变分自编码器(VAE)原理
- 条件生成模型
- PyTorch深度学习实践
- 生成式模型应用
