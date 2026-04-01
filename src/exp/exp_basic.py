"""
基础实验类

提供模型训练和验证的基础设施。
"""

import torch
import os


class ExpBasic:
    """
    基础实验类

    Attributes:
        model: PyTorch 模型实例
        device: 计算设备 ('cuda' 或 'cpu')
    """

    def __init__(self, model: torch.nn.Module, config):
        self.model = model
        self.config = config

        # 设置设备
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config.gpu}")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def _prepare_inputs(self, *inputs):
        """将输入张量移动到计算设备"""
        return [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]

    def save_checkpoint(self, epoch: int, checkpoint_dir: str = "./checkpoints/"):
        """保存模型检查点"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)
