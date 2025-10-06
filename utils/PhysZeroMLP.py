# -*- coding: utf-8 -*-
"""
第三阶段: 物理与业务规则约束 (模型核心)

文件名: utils/PhysZeroMLP.py

功能:
1. 定义一个标准的MLP网络结构。
2. 在训练步骤中，实现一个自定义的“约束损失函数”。
3. 该损失函数不仅惩罚预测误差(MSE)，还强力惩罚违反物理和业务逻辑的预测。
4. 模型可以直接从'train_enhanced.py'脚本中调用和训练。
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class PhysZeroMLP(pl.LightningModule):
    """
    PhysZeroMLP: 一个集成了物理和业务逻辑约束的神经网络模型。
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 depth: int,
                 lr: float,
                 weight_decay: float = 0.0,
                 lambda_constraint: float = 1.0):
        """
        模型初始化。

        Args:
            input_size (int): 输入特征的总数量。
            hidden_size (int): 隐藏层的神经元数量。
            depth (int): 隐藏层的深度。
            lr (float): 学习率。
            weight_decay (float): L2正则化权重。
            lambda_constraint (float): 物理约束损失的权重超参数。
        """
        super(PhysZeroMLP, self).__init__()
        # 保存超参数，方便后续加载模型和日志记录
        self.save_hyperparameters()

        # --- 1. 定义网络结构 ---
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        return self.network(x)

    def _calculate_constrained_loss(self, x, y_hat, y):
        """
        核心函数: 计算包含物理和业务逻辑约束的总损失。
        """
        # --- 损失1: 主要预测损失 (MSE) ---
        # 衡量模型预测的下一个温度点与真实值之间的差距
        prediction_loss = nn.MSELoss()(y_hat, y)

        # --- 损失2: 物理/业务逻辑约束损失 ---
        # 我们需要从输入的x张量中，根据列的顺序找到对应的状态特征
        # 这个顺序由 `train_enhanced.py` 中的 `feature_columns` 列表决定
        # 假设顺序是: ['Current_State', ..., 'Is_Cooling', 'Is_Defrosting', ...]
        # 注意: 这里的索引需要根据您在 train_enhanced.py 中 feature_columns 的最终顺序来确定！
        # 这是一个示例索引，您需要根据实际情况进行调整。
        # 让我们假设:
        # col 0: Current_State (归一化后)
        # col 4: Is_Cooling (0或1)
        # col 5: Is_Defrosting (0或1)
        
        current_state_norm = x[:, 0]
        is_cooling = x[:, 4]
        is_defrosting = x[:, 5]

        # 计算预测的温度变化量 (在归一化空间中)
        delta_t_predicted = y_hat.squeeze() - current_state_norm
        
        # 约束1: 制冷时，温度必须下降 (delta_t_predicted 应该为负)
        # 我们只选择正在制冷(is_cooling > 0.5)的样本，并惩罚所有预测温度上升(delta_t_predicted > 0)的情况
        cooling_violation = torch.relu(delta_t_predicted[is_cooling > 0.5]).mean()

        # 约束2: 化霜时，温度必须上升 (delta_t_predicted 应该为正)
        # 我们只选择正在化霜(is_defrosting > 0.5)的样本，并惩罚所有预测温度下降(delta_t_predicted < 0)的情况
        defrost_violation = torch.relu(-delta_t_predicted[is_defrosting > 0.5]).mean()
        
        # 约束3: 关机时(自然状态)，温度应该缓慢上升或不变，但绝不能下降
        # 我们选择既不制冷也不化霜的样本，并惩罚预测温度下降的情况
        natural_violation_mask = (is_cooling < 0.5) & (is_defrosting < 0.5)
        natural_violation = torch.relu(-delta_t_predicted[natural_violation_mask]).mean()
        
        # 将所有违规损失加起来
        constraint_loss = cooling_violation + defrost_violation + natural_violation
        
        # --- 总损失 ---
        total_loss = prediction_loss + self.hparams.lambda_constraint * constraint_loss

        # 返回各个损失分量，方便日志记录
        return total_loss, prediction_loss, constraint_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        total_loss, prediction_loss, constraint_loss = self._calculate_constrained_loss(x, y_hat, y)

        # 使用self.log记录所有损失，可以在TensorBoard中查看
        self.log('train/total_loss', total_loss, on_step=False, on_epoch=True)
        self.log('train/prediction_loss', prediction_loss, on_step=False, on_epoch=True)
        self.log('train/constraint_loss', constraint_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        total_loss, prediction_loss, constraint_loss = self._calculate_constrained_loss(x, y_hat, y)
        
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/prediction_loss', prediction_loss)
        self.log('val/constraint_loss', constraint_loss)

        return total_loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer