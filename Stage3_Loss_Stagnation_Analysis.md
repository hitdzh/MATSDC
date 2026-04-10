# Stage 3 Loss 停滞分析报告与修复方案

## 一、日志现象总结

| 阶段 | Loss 变化 | 正常性 |
|------|-----------|--------|
| Stage 2 (100 epochs) | 0.332 → 0.031，持续下降 | 正常 |
| **Stage 3 (100 epochs)** | **1.191 → 0.699 → ~0.699，Epoch 20 后完全停滞** | **异常** |

**核心问题**：Stage 3 在 Epoch 20 后 loss 几乎完全不变（从 0.6990 到 0.6989），持续 80 个 epoch 无任何下降。

---

## 二、原因分析

### 根本原因：多因素叠加导致梯度失效

#### 1. 学习率设置不当（最主要原因）
- **lr_encoder = 1e-3**（Adam）：对 Transformer encoder 来说过大，导致振荡而非收敛
- **lr_centers = 0.5**（SGD）：极大，使 centers 在第一个 epoch 就几乎完全"吞掉"了特征，锁死了特征空间
- **无学习率调度器**：100 个 epoch 全程固定 lr，Epoch 20 后优化器在局部最优点附近振荡

#### 2. Backbone 输出 L2 归一化（架构问题）
- `PatchTSTFeatureExtractor.forward()` 最后一步：`features = F.normalize(features, p=2, dim=1)`（[PatchTSTEncoder.py:338](src/layers/PatchTSTEncoder.py#L338)）
- 所有特征被投影到单位超球面，**所有样本共享同一个 manifold**
- Center Loss 在单位球面上优化欧氏距离 = 优化 `||f||² + ||c||² - 2·f·c = 1 + ||c||² - 2·cos(f,c)`，等价于最大化 cosine similarity
- 但 **projection 层 + classifier 层** 之间的 L2 归一化使得特征方向的"探索空间"极为有限
- 加上 lr_centers=0.5 的 centers 几乎瞬间就抓住了特征，导致后续特征无法移动

#### 3. 标签分布极度不平衡（数据问题）
```
Label distribution: [35733, 116, 77, 82, 79, 102, 85, 83, 46, 53]
```
- Class 0 占总数的 **98.0%**（35733/36456），其余 9 个类合计不到 2%
- `TensorDataset + DataLoader(shuffle=True)` 无法保证每个 batch 有足够的类别多样性
- SupCon Loss 的正样本对几乎全部来自 Class 0，少数类别的样本在随机 batch 中碰面的概率极低
- 对少数类来说，SupCon Loss 接近 0（`n_positives ≈ 0`），无法提供有效的学习信号

#### 4. 无梯度裁剪 + 联合损失耦合振荡
- 两个优化器同时更新：Adam 更新 encoder，SGD 更新 centers
- 当 centers 快速漂移时，Center Loss 的梯度方向不断变化，encoder 的梯度被"拉扯"
- 无梯度裁剪（`max_norm`），大梯度可能进一步破坏特征空间的稳定性

#### 5. Loss 0.699 的意义
- `0.699 ≈ ln(2)` ≈ 真实标签熵（Class 0 占比 98% 时，预测"Class 0"的交叉熵 ≈ -0.98·ln(0.98) ≈ 0.02）
- 这个 loss 值说明模型已经收敛到一个"预测 Class 0"的平凡解
- Center Loss ≈ 1.0 说明各样本特征已被拉到各自的 center 附近（但这些 center 是随机初始化的）
- 后续所有 loss 的变化都在第三位小数，说明 **特征空间已被 centers 固定死**

---

## 三、修复方案

---

### 优先级 1：修复学习率问题（必做）

#### 1.1 添加 CosineAnnealing 学习率调度器

**解决的问题**：
- 当前 lr 固定为 1e-3，optimizer 在前 20 个 epoch 快速下降 loss 后，到达一个平坦区域（plateau），此时大学习率导致在极浅的局部最优点附近来回振荡，无法进一步收敛
- 固定 lr 意味着无法在训练后期使用更精细的步长进行"微调"

**为什么使用 CosineAnnealing**：
- 学习率按余弦曲线从初始值逐渐衰减到最小值，初期快速探索，后期精细收敛
- 相比 StepLR（阶梯式下降）更平滑，相比 ReduceLROnPlateau（基于指标）不需要验证集
- 100 个 epoch 的训练非常适合 T_max=100 的余弦调度

**如何实现**：
修改 [src/trainers/metric_trainer.py](src/trainers/metric_trainer.py)：

在 `__init__` 方法中（optimizer 定义之后）添加：
```python
self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer_encoder, T_max=epochs, eta_min=1e-5
)
```

在 `fit()` 方法的每个 epoch 结束后添加调度器更新：
```python
self.scheduler.step()
```

**预计结果**：
- 学习率从 1e-3 逐渐衰减到 1e-5，Epoch 20 之后的学习率约为 3e-4，能在 plateau 区域进行精细搜索
- Loss 在 Epoch 20 之后应继续下降，而非完全停滞

---

#### 1.2 降低初始学习率并重新平衡 centers 学习率

**解决的问题**：
- `lr_encoder=1e-3` 对 4 层 Transformer encoder 来说过大，尤其是当 center loss 主导梯度方向时，容易造成特征空间的剧烈振荡
- `lr_centers=0.5` 极端过大——center 向量在第一个 batch 的更新幅度可能超过特征向量本身的大小，导致 centers 在特征空间中"跑在前面"，锁死了特征向量的优化空间

**为什么这样设置**：
- `lr_encoder=1e-4`：Transformer 微调的标准学习率范围（1e-4 到 5e-5），在已有部分收敛的特征空间中进行细粒度优化
- `lr_centers=0.01`：参考 Center Loss 原始论文（Wen et al., ECCV 2016），center 更新应该"跟在特征后面"，即特征移动后 centers 再缓慢跟近，而非反客为主
- 原始论文中 center lr=0.5 是在 CIFAR-100 等平衡数据集上验证的，对于 98% vs 2% 的极端不平衡数据，centers 过快会导致大类迅速占据大部分特征空间

**如何实现**：
修改 [configs/default.py](configs/default.py)：
```python
lr_encoder: float = 1e-4         # 1e-3 → 1e-4
lr_centers: float = 0.01          # 0.5 → 0.01
```

**预计结果**：
- Centers 不再主导特征空间，特征向量和 centers 协同优化
- Loss 下降更平滑，无剧烈振荡

---

### 优先级 2：修复标签不平衡问题（必做）

#### 2.1 实现类别均衡采样（WeightedRandomSampler）

**解决的问题**：
- 当前使用 `shuffle=True` 的随机采样，Class 0 占比 98%，导致绝大多数 batch 以 Class 0 为主
- 在 batch_size=32 的情况下，少数类（每个类约 50-116 个样本）平均每个 epoch 只能被采样到 1-2 次
- SupCon Loss 需要同标签的正样本对才能计算有意义梯度，少数类样本在大多数 batch 中无法形成正样本对

**为什么使用 WeightedRandomSampler**：
- 每个样本的采样概率与其所属类的样本数成反比——Class 0 的样本被采样概率低，少数类样本被采样概率高
- 保证每个 batch 的类别分布接近均衡，从而 SupCon Loss 和 Center Loss 对所有类都能提供有效梯度
- 不需要修改数据集本身，只需改变采样策略

**如何实现**：
修改 [scripts/stage3_metric_learning.py](scripts/stage3_metric_learning.py#L118) 和 [scripts/run_full_pipeline.py](scripts/run_full_pipeline.py#L262)，将 DataLoader 的创建改为：

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

class_counts = np.bincount(valid_labels.astype(int))  # 每类的样本数
weights = 1.0 / class_counts[valid_labels.astype(int)]  # 每类样本的权重（少数类权重更大）
sample_weights = weights[valid_labels.astype(int)]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler)
```

注意：使用 `sampler` 后不能再传 `shuffle=True`。

**预计结果**：
- 每个 batch 中各类别样本数量趋于均衡（~3-4 个/类）
- SupCon Loss 对所有类都能计算正样本对梯度
- Center Loss 对少数类的约束同样有效

---

#### 2.2 添加类别权重到 CrossEntropy Loss

**解决的问题**：
- 即使修复了采样问题，原始 CE Loss 仍会给大类（Class 0）更多的梯度贡献
- 在 98% vs 2% 的极端不平衡下，少数类的分类边界几乎由大类主导，少数类样本被错误分类的惩罚太小

**为什么使用逆频率加权**：
- 类别权重 = 1 / 该类样本数，使得每类的总权重贡献相等
- Class 0 的权重 = 1/35733 ≈ 2.8e-5，Class 9 的权重 = 1/46 ≈ 0.022——差距约 800 倍
- 少数类样本被正确分类时获得更大的奖励，被错误分类时受到更大的惩罚

**如何实现**：
修改 [src/trainers/metric_trainer.py](src/trainers/metric_trainer.py) 的 `__init__`：

```python
def __init__(self, ..., class_weights: torch.Tensor = None, ...):
    ...
    if class_weights is not None:
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        self.ce_criterion = nn.CrossEntropyLoss()
```

在调用处（[stage3_metric_learning.py](scripts/stage3_metric_learning.py) 和 [run_full_pipeline.py](scripts/run_full_pipeline.py)）计算并传入：

```python
class_counts = np.bincount(valid_labels.astype(int))
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum() * len(class_counts)  # 归一化
trainer = MetricTrainer(..., class_weights=class_weights, ...)
```

**预计结果**：
- 少数类（Class 1-9）的分类准确率显著提升
- 各类的 Center Loss 贡献趋于平衡
- 模型不再"放弃"少数类

---

### 优先级 3：修复 L2 归一化问题（推荐）

#### 3.1 将 L2 归一化从 backbone 移至 XEncoder.encode()，且设为可选

**解决的问题**：
- 当前 `PatchTSTFeatureExtractor.forward()` 末尾强制对特征做 L2 归一化（[PatchTSTEncoder.py:338](src/layers/PatchTSTEncoder.py#L338)）
- L2 归一化将所有特征映射到单位超球面，使得欧氏距离等于 `sqrt(2 - 2·cos(θ))`——本质上优化的是 cosine similarity
- 但 Center Loss 的设计意图是在**欧氏空间**中缩小类内距离，强制归一化后，Center Loss 和 SupCon Loss 都在优化同一件事（cosine similarity），冗余且限制了特征的多样性
- 更严重的是：归一化后特征 scale 被固定为 1，center loss 的梯度 `f - c` 的 scale 也被限制，加上 centers 过大学习率，特征几乎无法移动

**为什么这样做**：
- Stage 2 的 Y-Encoder 也用同一个 `PatchTSTFeatureExtractor`，如果直接删除 L2 归一化会影响 Stage 2 的输出分布
- 将 L2 归一化设为可选参数（`l2_normalize=True/False`），Stage 2 保持默认 True（K-Center 需要归一化特征），Stage 3 X-Encoder 设为 False（允许特征有 scale 变化）
- 或者在 `XEncoder.encode()` 层面控制：backbone 不归一化，在 encode() 末尾根据需求归一化

**如何实现**：
方案 A（修改 PatchTSTFeatureExtractor）：
修改 [src/layers/PatchTSTEncoder.py](src/layers/PatchTSTEncoder.py#L231)：

```python
def __init__(..., l2_normalize: bool = True):
    ...
    self.l2_normalize = l2_normalize

def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
    if self.l2_normalize:
        features = F.normalize(features, p=2, dim=1)
    return features
```

同时修改 [src/models/x_encoder.py](src/models/x_encoder.py)：
```python
self.backbone = PatchTSTFeatureExtractor(
    ...
    l2_normalize=False,  # Stage 3 X-Encoder 不在 backbone 层面归一化
)
```

方案 B（在 XEncoder.encode() 中控制，更简洁）：
不修改 PatchTSTFeatureExtractor，直接在 XEncoder 中决定是否归一化：

```python
def encode(self, X: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    backbone_features = self.backbone(X)
    features = self.projection(backbone_features)
    if normalize:
        features = F.normalize(features, p=2, dim=1)
    return features
```

调用处 `x_encoder.encode(all_X)` 不传 `normalize=True`，让特征保持原始 scale。

**预计结果**：
- 特征不再被困在单位超球面上，可以有 scale 变化
- Center Loss 在欧氏空间中的优化更有效
- Loss 能够继续下降到 0.5 以下

---

### 优先级 4：训练稳定性增强（推荐）

#### 4.1 添加梯度裁剪（Gradient Clipping）

**解决的问题**：
- Stage 3 使用两个优化器（Adam + SGD）同时更新，centers 的梯度可能很大，尤其在训练初期
- 无梯度裁剪时，大梯度可能导致参数更新过大，破坏已收敛的特征空间
- 当前 loss 停滞在 0.699 很可能就是因为一次大梯度更新将特征空间打乱后，优化器陷入了一个"不坏但也学不动"的平衡态

**为什么使用 norm-based clipping（`clip_grad_norm_`）**：
- 相比 clip_grad_value（逐元素裁剪），norm-based 保留梯度方向，只控制幅度
- max_norm=1.0 是 Transformer 训练的标准默认值（BERT、ViT 等都使用）
- 不影响梯度方向，只防止梯度爆炸

**如何实现**：
修改 [src/trainers/metric_trainer.py](src/trainers/metric_trainer.py) 的 `train_epoch()` 方法中 `backward()` 之后：

```python
loss_total.backward()

# 梯度裁剪：防止梯度爆炸
torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(self.center_loss.parameters(), max_norm=1.0)

self.optimizer_encoder.step()
self.optimizer_centers.step()
```

**预计结果**：
- 训练过程更稳定，无损失骤升或 NaN
- 与 CosineAnnealing 结合使用效果更好（衰减的学习率 + 有界的梯度）

---

#### 4.2 分阶段训练（Warmup + Backbone Freeze）

**解决的问题**：
- 当前从第一个 epoch 开始，backbone、projection、classifier 一起训练，但 backbone 参数最多（4 层 Transformer），随机初始化后对梯度贡献最大
- 这会"稀释" projection 和 classifier 的学习信号，导致早期优化效率低
- 同时，Center Loss 的梯度方向会过早地被 backbone 主导，projection 层的学习空间受限

**为什么先冻结 backbone**：
- 只训练 classifier 和 projection 层（参数量小），可以快速让网络收敛到一个较好的分类面
- 5-10 个 epoch 后再解冻 backbone，进行端到端的微调
- 这和 BERT 的预训练+微调策略、以及 LoRA 等 Adapter 方法的思路一致

**如何实现**：
修改 [src/trainers/metric_trainer.py](src/trainers/metric_trainer.py) 的 `fit()` 方法：

```python
def fit(self, dataloader, pseudo_labels, epochs, warmup_epochs=5):
    loss_history = []
    warmup_finished = False

    for epoch in range(epochs):
        # Warmup 结束后解冻 backbone
        if epoch == warmup_epochs and not warmup_finished:
            for param in self.encoder.backbone.parameters():
                param.requires_grad = True
            warmup_finished = True
            print(f"  [Epoch {epoch+1}] Backbone unfrozen")

        avg_loss = self.train_epoch(dataloader, pseudo_labels)
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.4f}")

        if self.scheduler is not None:
            self.scheduler.step()

    return loss_history
```

同时在 `__init__` 中默认冻结 backbone：
```python
# 默认冻结 backbone，只训练 projection + classifier
for param in self.encoder.backbone.parameters():
    param.requires_grad = False
```

**预计结果**：
- Warmup 阶段 loss 快速下降到 ~0.5（只优化简单分类面）
- 解冻后 loss 继续下降（端到端优化特征空间）
- 最终特征质量更高，对 Stage 4 的聚类更有帮助

---

#### 4.3 降低 center_loss_weight

**解决的问题**：
- 当前 `center_loss_weight=1.0` 与 CE Loss 平权，但 Center Loss 只优化类内紧致性，不直接优化分类边界
- 在特征空间被 centers 固定后（lr_centers=0.5 导致），增大 center_loss_weight 反而强化了对已固定空间的约束，进一步锁死优化
- Center Loss 过大会压缩每个类的特征分布，减少类间可分性

**为什么使用 0.1**：
- Center Loss 是辅助损失，权重应该小于主损失（CE）
- 参考 Face Recognition 领域的 Center Loss 使用方式（Wen et al.），center loss weight 通常在 0.001-0.1 之间
- 设为 0.1 使得 CE Loss 主导分类边界优化，Center Loss 提供辅助的类内紧致约束

**如何实现**：
修改 [configs/default.py](configs/default.py)：
```python
center_loss_weight: float = 0.1   # 1.0 → 0.1
```

**预计结果**：
- CE Loss 有更大的优化空间来调整分类边界
- Center Loss 不会主导优化方向
- 各类别的特征分布更加合理（紧凑但有区分度）

---

## 四、修改文件清单与执行顺序

建议按以下顺序修改和测试：

| 步骤 | 文件 | 修改内容 | 优先级 |
|------|------|----------|--------|
| 1 | [configs/default.py](configs/default.py) | lr_encoder=1e-4, lr_centers=0.01, center_loss_weight=0.1 | P1 |
| 2 | [src/trainers/metric_trainer.py](src/trainers/metric_trainer.py) | 添加 CosineAnnealingLR 调度器 + 梯度裁剪 + backbone freeze | P1+P4 |
| 3 | [scripts/stage3_metric_learning.py](scripts/stage3_metric_learning.py) | WeightedRandomSampler + class_weights 传入 trainer | P2 |
| 4 | [scripts/run_full_pipeline.py](scripts/run_full_pipeline.py) | 同上（Stage 3 调用的另一处） | P2 |
| 5 | [src/models/x_encoder.py](src/models/x_encoder.py) | backbone 传入 l2_normalize=False | P3 |

---

## 五、验证方案

1. **短期验证**：修改后重新运行 Stage 3，观察 loss 是否持续下降（至少 Epoch 20 后仍有明显变化，例如从 0.699 降到 0.6 以下）
2. **分量验证**：在 `train_epoch()` 中返回 `loss_ce`、`loss_center`、`loss_supcon` 的平均值，而非仅返回总 loss，分别监控三者变化
3. **分类验证**：每个 epoch 结束后计算各类的召回率（top-1 accuracy per class），确保少数类也被正确学习（召回率 > 60%）
4. **聚类验证**：Stage 3 完成后查看 Stage 4 的聚类结果，与修改前的 silhouette score 对比
5. **对比实验**：先只做 P1 修改观察效果，再逐步加入 P2/P3/P4

---

## 六、快速验证命令

```bash
# 修改后的 Stage 3 单独运行命令
python scripts/stage3_metric_learning.py \
    --input outputs/stage2/pseudo_labels.pt \
    --epochs 100 \
    --lr_encoder 1e-4 \
    --lr_centers 0.01 \
    --center_loss_weight 0.1

# 或在配置文件中直接修改后运行完整流程
python scripts/run_full_pipeline.py --dataset weather
```
