"""
全局超参数配置模块
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """时序数据无监督样本挖掘 Pipeline 全局配置"""

    # ==================== 数据参数 ====================
    dataset: str = 'dummy'            # 数据集名称
    seq_len: int = 336                # 输入序列 X 的窗口长度
    pre_len: int = 96                # 预测目标 Y 的窗口长度
    feature_dim: int = 21             # 时序数据特征维度（多变量时序）

    # ==================== 模型参数 ====================
    y_hidden_dim: int = 256           # Y-PreEncoder 输出特征维度
    x_hidden_dim: int = 256           # X-Encoder 输出特征维度
    lstm_layers: int = 4             # X-Encoder LSTM 层数
    lstm_hidden_size: int = 256      # LSTM 隐状态维度
    
    # 以下为 Autoformer/Informer/Transformer 模型参数
    label_len: int = 336              # 解码器起始长度（通常等于 seq_len）
    enc_in: int = 21                 # 编码器输入维度（特征数）
    dec_in: int = 21                 # 解码器输入维度
    c_out: int = 128                  # 输出维度
    d_model: int = 512              # 模型维度
    n_heads: int = 8                # 注意力头数
    e_layers: int = 2               # 编码器层数
    d_layers: int = 1               # 解码器层数
    d_ff: int = 2048               # 前馈网络维度
    factor: int = 1                # 注意力因子
    moving_avg: int = 25            # 移动平均窗口（series_decomp）
    dropout: float = 0.05           # Dropout 率
    embed: str = 'timeF'           # 时间嵌入方式: timeF, fixed, learned
    activation: str = 'gelu'        # 激活函数
    freq: str = 'h'                # 时间频率: h, t, d 等
    output_attention: bool = False  # 是否输出注意力权重
    distil: bool = True             # Informer 蒸留（是否使用 ConvLayer）
    # Reformer
    bucket_size: int = 4
    n_hashes: int = 4

    # ==================== 聚类参数 ====================
    K: int = 5                       # K-Center 聚类数（伪标签类别数）
    kcenter_method: str = 'greedy'   # K-Center 方法: 'greedy' 或 'robust'
    outliers_fraction: float = 0.05  # Robust K-Center 异常值比例
    n_clusters: int = 4              # 谱聚类簇数
    n_prototypes: int = 4            # 最终提取的典型样本数 N
    spectral_sigma: float = 1.0      # 谱聚类 RBF 核带宽参数

    # ==================== 训练参数 ====================
    epochs: int = 2                  # 阶段3 训练轮数
    batch_size: int = 32             # 批大小
    lr_encoder: float = 1e-3         # X-Encoder 学习率 (Adam)
    lr_centers: float = 0.5          # Center Loss 中心点学习率 (SGD)
    center_loss_weight: float = 1.0  # Center Loss 权重 λ (联合 Loss: CE + λ * CenterLoss)

    # ==================== Y-Encoder 预训练参数 ====================
    pretrain_epochs: int = 5         # Y-Encoder 预训练轮数
    pretrain_lr: float = 1e-3        # Y-Encoder 预训练学习率
    pretrain_batch_size: int = 64    # Y-Encoder 预训练批大小

    @property
    def pred_len(self) -> int:
        """pred_len 别名 — 兼容 Autoformer/Informer/Transformer 模型接口"""
        return self.pre_len
