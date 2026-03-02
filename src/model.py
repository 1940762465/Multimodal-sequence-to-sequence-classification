"""
修改多张图片融合的方式，不仅仅只是简单的拉伸变换，应该以softmax的形式进行相对应的融合
"""
import numpy as np
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import math
import os
from safetensors.torch import load_file  # Safetensors加载权重的函数


# 位置编码
class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码：不引入额外可训练参数
    d_model：token维度
    max_len:处理序列的最大维度
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 不参与训练

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)  # (1, T, d_model)


class PatchEmbed(nn.Module):
    """
    对2D图像作Patch Embedding操作
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=256, norm_layer=None):
        """
        :param img_size: 输入图像的大小
        :param patch_size: 一个patch的大小
        :param in_c: 输入图像的通道数
        :param embed_dim: 输出的每个token的维度
        :param norm_layer: 指定归一化方式，默认为None
        """
        super().__init__()
        img_size = (img_size, img_size)  # 224 -> (224, 224)
        patch_size = (patch_size, patch_size)  # 16 -> (16, 16)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 计算原始图像被划分为(14, 14)个小块
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 计算patch的个数为14*14=196个
        # 定义卷积层，考虑多通道输入
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        # 定义归一化方式
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        :param x: 原始图像
        :return: 处理后的图像
        """
        B, C, H, W = x.shape
        # 检查图像高宽和预先设定是否一致，不一致则报错
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 对图像依次作卷积、展平和调换处理: [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 归一化处理
        x = self.norm(x)
        return x


# ViT编码器--windows+外网直接下载
class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=256,vit_type="vit_tiny_patch16_224",image_method='log_curve'):
        super(ViTEncoder, self).__init__()
        # 使用预训练的ViT模型
        self.vit_model = timm.create_model(vit_type, pretrained=pretrained)
        self.vit_model.head = nn.Identity()  # 去除分类头
        self.image_method =image_method  # 图片转换方法
        self.relu = nn.ReLU()

        # 输出层，将图像特征维度压缩到output_dim
        if vit_type == "vit_tiny_patch16_224":
            self.fc = nn.Linear(192, output_dim)
        if vit_type == "vit_base_patch16_224":
            self.fc = nn.Linear(768, output_dim)

        self.image_ln = nn.LayerNorm(output_dim)  # 对提取的图像特征进行归一化

        # 冻结所有ViT的参数
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        输入：x (batch_size, num_images, channels, height, width)
        输出：x (batch_size, num_images, output_dim)
        """
        if self.image_method in ["log_curve","wave_trans","gaf_trans"]:
            batch_size, num_images, C, H, W = x.shape
            x_reshaped = x.view(-1, C, H, W)  # (batch_size*num_images, C, H, W)
            all_features = self.vit_model(x_reshaped)  # (batch_size*num_images, vit_feat_dim)
            all_features = all_features.contiguous().view(batch_size, num_images, -1)  # (batch_size, num_images, vit_feat_dim)
            # 投影到目标维度
            features = self.fc(all_features)
            features = self.image_ln(features)
            features = self.relu(features)  # (batch_size, num_images, output_dim)
            features = features.transpose(1, 2)
        else:
            x = self.vit_model(x)  # 输出是768维
            x = self.fc(x)
            features = self.image_ln(x)
            features = self.relu(features)
            features = features.unsqueeze(2)
        return features


# ViT本地权重加载
class ViTEncoder2(nn.Module):
    def __init__(self, pretrained=False, output_dim=256,vit_type="vit_tiny_patch16_224",image_method='log_curve',weight_path=None):
        super(ViTEncoder2, self).__init__()
        # Load the pre-trained ViT model
        self.vit_model = timm.create_model(vit_type, pretrained=pretrained)
        self.vit_model.head = nn.Identity()  # 去除分类头
        self.image_method = image_method  # 图片转换方法
        self.relu = nn.ReLU()

        # 输出层，将图像特征维度压缩到output_dim
        if vit_type == "vit_tiny_patch16_224":
            self.fc = nn.Linear(192, output_dim)
        if vit_type == "vit_base_patch16_224":
            self.fc = nn.Linear(768, output_dim)

        self.image_ln = nn.LayerNorm(output_dim)  # 对提取的图像特征进行归一化

        if weight_path is not None and os.path.exists(weight_path):
            self._load_weights(weight_path)

        # 冻结所有ViT的参数
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def _load_weights(self, weight_path):
        """Load weights from a .saftensors file"""
        try:
            # 使用safetensors库加载权重
            print(f"Loading model weights from: {weight_path}")
            checkpoint = load_file(weight_path)  # Safetensors加载
            self.vit_model.load_state_dict(checkpoint, strict=False)
            print(f"Successfully loaded weights from {weight_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def forward(self, x):
        """
        输入：x (batch_size, num_images, channels, height, width)
        输出：x (batch_size, num_images, output_dim)
        """
        if self.image_method in ["log_curve", "wave_trans","gaf_trans"]:
            batch_size, num_images, C, H, W = x.shape
            x_reshaped = x.view(-1, C, H, W)  # (batch_size*num_images, C, H, W)
            all_features = self.vit_model(x_reshaped)  # (batch_size*num_images, vit_feat_dim)
            all_features = all_features.contiguous().view(batch_size, num_images, -1)  # (batch_size, num_images, vit_feat_dim)
            # 投影到目标维度
            features = self.fc(all_features)
            features = self.image_ln(features)
            features = self.relu(features) # (batch_size, num_images, output_dim)
            features = features.transpose(1, 2)
        else:
            x = self.vit_model(x)  # 输出是768维
            x = self.fc(x)
            features = self.image_ln(x)
            features = self.relu(features)
            features = features.unsqueeze(2)
        return features  # 输出维度768


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=False, output_dim=256, resnet_type="resnet18", image_method='log_curve',
                 weight_path=None):
        super(ResNetEncoder, self).__init__()

        # 使用tim库加载预训练的ResNet模型
        self.resnet_model = timm.create_model(resnet_type, pretrained=pretrained)

        # 去除最后的全连接层
        self.resnet_model.fc = nn.Identity()

        self.image_method = image_method  # 图片转换方法
        self.relu = nn.ReLU()

        # 输出层，将ResNet提取的特征压缩到output_dim
        if resnet_type in ["resnet18", "resnet34"]:
            self.fc = nn.Linear(512, output_dim)
        elif resnet_type in ["resnet50", "resnet101", "resnet152"]:
            self.fc = nn.Linear(2048, output_dim)

        self.image_ln = nn.LayerNorm(output_dim)  # 对提取的图像特征进行归一化

        if weight_path is not None and os.path.exists(weight_path):
            self._load_weights(weight_path)

        # 冻结ResNet的所有参数
        for param in self.resnet_model.parameters():
            param.requires_grad = False

    def _load_weights(self, weight_path):
        """Load weights from a .saftensors file"""
        try:
            # 使用safetensors库加载权重
            print(f"Loading model weights from: {weight_path}")
            checkpoint = load_file(weight_path)  # Safetensors加载
            self.resnet_model.load_state_dict(checkpoint, strict=False)
            print(f"Successfully loaded weights from {weight_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def forward(self, x):
        """
        输入：x (batch_size, num_images, channels, height, width)
        输出：x (batch_size, num_images, output_dim)
        """
        if self.image_method in ["log_curve", "wave_trans", "gaf_trans"]:
            batch_size, num_images, C, H, W = x.shape
            x_reshaped = x.view(-1, C, H, W)  # (batch_size*num_images, C, H, W)
            all_features = self.resnet_model(x_reshaped)  # (batch_size*num_images, vit_feat_dim)
            all_features = all_features.contiguous().view(batch_size, num_images, -1)  # (batch_size, num_images, vit_feat_dim)
            # 投影到目标维度
            features = self.fc(all_features)
            features = self.image_ln(features)
            features = self.relu(features)  # (batch_size, num_images, output_dim)
            features = features.transpose(1, 2)
        else:
            x = self.resnet_model(x)  # 输出是2048维
            x = self.fc(x)
            features = self.image_ln(x)
            features = self.relu(features)
            features = features.unsqueeze(2)
        return features  # 输出维度output_dim


# 数值模态分类器-CNN
class SimpleCNNSeq2Seq(nn.Module):
    def __init__(self, input_size, num_classes, window_size, hidden_dim=32,dropout=0.1):
        super(SimpleCNNSeq2Seq, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 编码器 - 1D CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),

            nn.Conv1d(16, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # 分类器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # x形状: (batch_size, window_size, input_size)
        # 转置为 (batch_size, input_size, window_size) 用于CNN
        x = x.transpose(1, 2)

        # 编码器
        encoded = self.encoder(x)  # (batch_size, hidden_dim, 1)
        output = self.decoder(encoded.transpose(1, 2))

        return output


# 数值模态分类器-BiLSTM
class SimpleLSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_dim=64,
        num_layers=2,
        bidirectional=False,  # 决定是否是双向LSTM
        dropout=0.1
    ):
        super(SimpleLSTMSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # ========= Encoder: LSTM =========
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,      # 输入 (B, T, F)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # ========= Decoder: MLP（与CNN版保持一致） =========
        self.decoder = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        """
        x: (batch_size, window_size, input_size)
        """

        # LSTM 编码
        encoded, _ = self.encoder(x)
        # encoded: (B, T, hidden_dim * num_directions)

        # 逐时刻分类
        output = self.decoder(encoded)
        # output: (B, T, num_classes)
        return output


# 数值模态编码器-Transformer编码器
class SimpleTransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size

        # 1) 输入投影：把数值特征 F -> d_model（Transformer 的 token embedding）
        self.dim_token = nn.Linear(input_size, d_model)

        # 2) 位置编码（必须有，否则 Transformer 不知道顺序）
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(5000, window_size + 1))

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            # dropout=dropout,
            activation="gelu",
            batch_first=True,   # 让输入保持 (B, T, d_model)
            # norm_first=True     # 更稳定（Pre-LN 风格）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) 逐时间步分类头（与 CNN/LSTM 保持一致思路）
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)  —— 逐点标签任务
        return: (B, T, num_classes)
        """
        # 可选：保证固定窗长一致（论文复现更稳）
        if x.size(1) != self.window_size:
            raise ValueError(f"window_size mismatch: expected {self.window_size}, got {x.size(1)}")

        x = self.dim_token(x)  # (B, T, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 编码（不加mask：默认窗口内全注意力）
        encoded = self.encoder(x)  # (B, T, d_model)

        # 逐点分类
        out = self.decoder(encoded)  # (B, T, num_classes)
        return out


# 图像模态编码器-ViT
class ViTSeq2Seq(nn.Module):
    def __init__(self,input_size, num_classes, window_size, ViT_feature_dim=256,vit_type="vit_tiny_patch16_224",image_method='matrix_visual',system="windows",weight_path=None):
        super(ViTSeq2Seq, self).__init__()
        self.window_size = window_size
        self.method = image_method
        if system=="windows":
            self.image_feature_extractor = ViTEncoder(pretrained=True, vit_type=vit_type, image_method=self.method,
                                                  output_dim=ViT_feature_dim)
        else:
            self.image_feature_extractor = ViTEncoder2(pretrained=False, vit_type=vit_type, image_method=self.method,
                                                      output_dim=ViT_feature_dim,weight_path=weight_path)
        if self.method in ["log_curve", "wave_trans","gaf_trans"]:
            self.image_expand_windows = nn.Linear(input_size, window_size)
        else:
            self.image_expand_windows = nn.Linear(1, window_size)
        self.decoder = nn.Sequential(
            nn.Linear(ViT_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, image_features):
        image_expand_features = self.image_expand_windows(image_features)
        x = image_expand_features.transpose(1, 2)  # (batch_size, hidden_dim, window_size)
        output = self.decoder(x)  # (batch_size, window_size, num_classes)
        return output


class RestructViTSeq2Seq(nn.Module):
    def __init__(self, num_classes, window_size, drop_ratio=0.1, hidden_dim=64, num_heads=4, n_layer=1, input_channels=8,img_size=224,patch_size=16,image_method='matrix_visual'):
        """
        重建的ViT适用于灰度通道的叠加，也就是和原来的BxHxWxRGBxC不同，没有RGB，把C看为RGB
        """
        super(RestructViTSeq2Seq, self).__init__()
        self.window_size = window_size
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=input_channels,
                                      embed_dim=hidden_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码--可学习位置编码，和传统transformer位置编码不一样
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # 额外加1是为了添加[CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # Transformer Encoder部分
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)

        self.norm = nn.LayerNorm(hidden_dim)

        # 扩展升维
        self.image_expand_windows = nn.Linear(1, window_size)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # x的形状是 (batch_size, input_channels, H, W)
        # Step 1: 使用 Patch Embedding 对图像进行处理
        x = self.patch_embed(x)  # (batch_size, num_patches, hidden_dim)

        # Step 2: 添加 [CLS] token 并应用位置编码
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (batch_size, 1, hidden_dim)
        x = torch.cat([cls_token, x], dim=1)  # 在最前面添加 [CLS] token (batch_size, num_patches+1, hidden_dim)
        x = self.pos_drop(x + self.pos_embed)  # (batch_size, num_patches+1, hidden_dim)

        # Step 3: 使用 Transformer 进行处理
        x = self.transformer_encoder(x)  # (batch_size, num_patches+1, hidden_dim)
        x = self.norm(x)
        # Step 4: 输出 [CLS] token 的特征
        cls_token_out = x[:, 0]  # (batch_size, hidden_dim) 取出 [CLS] token 对应的特征

        # 使用自己构建的ViT进行分类
        cls_token_out = cls_token_out.unsqueeze(2)
        image_expand_features = self.image_expand_windows(cls_token_out)  # (batch_size, hidden_dim, window_size)
        x = image_expand_features.transpose(1,2)  # (batch_size, window_size, hidden_dim)
        output = self.decoder(x)
        return output


class ResNetSeq2Seq(nn.Module):
    def __init__(self,input_size, num_classes, window_size, resnet_feature_dim=256,resnet_type="resnet50",image_method='matrix_visual',system="windows",weight_path=None):
        super(ResNetSeq2Seq, self).__init__()
        self.window_size = window_size
        self.method = image_method
        if system=="windows":
            self.image_feature_extractor = ResNetEncoder(pretrained=True, resnet_type=resnet_type,image_method=self.method,
                                                         output_dim=resnet_feature_dim)
        else:
            self.image_feature_extractor = ResNetEncoder(pretrained=False, resnet_type=resnet_type,image_method=self.method,
                                                         output_dim=resnet_feature_dim,weight_path=weight_path)
        if self.method in ["log_curve", "wave_trans","gaf_trans"]:
            self.image_expand_windows = nn.Linear(input_size, window_size)
        else:
            self.image_expand_windows = nn.Linear(1, window_size)
        self.decoder = nn.Sequential(
            nn.Linear(resnet_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, image_features):
        image_expand_features = self.image_expand_windows(image_features)
        x = image_expand_features.transpose(1, 2)  # (batch_size, hidden_dim, window_size)
        output = self.decoder(x)  # (batch_size, window_size, num_classes)
        return output


# Encoder+Decoder编码器-concat
class MultiModeSeq2SeqModel1(nn.Module):
    def __init__(self, input_size, num_classes, window_size,d_model=128,n_head=8,dim_feedforward=128,n_layer=1,dropout=0.1,vit_type="vit_tiny_patch16_224",image_method='log_curve',system="windows",weight_path=None):
        super(MultiModeSeq2SeqModel1, self).__init__()
        self.window_size = window_size
        self.method = image_method

        # 数值模态编码器 - 保持时间维度
        # 1) 输入投影：把数值特征 F -> d_model（Transformer 的 token embedding）
        self.dim_token = nn.Linear(input_size, d_model)

        # 2) 位置编码（必须有，否则 Transformer 不知道顺序）
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(5000, window_size + 1))

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 让输入保持 (B, T, d_model)
            # norm_first=True     # 更稳定（Pre-LN 风格）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 图像模态的ViT特征提取器
        if system=="windows":
            self.image_feature_extractor = ViTEncoder(pretrained=True, vit_type=vit_type, image_method=self.method,
                                                  output_dim=d_model)
        else:
            self.image_feature_extractor = ViTEncoder2(pretrained=False, vit_type=vit_type, image_method=self.method,
                                                      output_dim=d_model,weight_path=weight_path)

        if self.method in ["log_curve","wave_trans","gaf_trans"]:
            self.image_expand_windows = nn.Linear(input_size, window_size)
        else:
            self.image_expand_windows = nn.Linear(1, window_size)

        # 解码器 - 使用1x1卷积进行时间步预测
        self.decoder = nn.Sequential(
            nn.Linear(2*d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, image_features):
        # x形状: (batch_size, window_size, input_size)
        x = self.dim_token(x)  # (B, T, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 编码（不加mask：默认窗口内全注意力）
        encoded = self.encoder(x)  # (B, T, d_model)

        # 解码 - 直接预测每个时间步
        image_expand_features = self.image_expand_windows(image_features)  # (batch_size, hidden_dim, window_size)

        # 模态融合
        fused_features = torch.cat([encoded, image_expand_features.transpose(1, 2)], dim=-1)  # (batch_size, window_size, hidden_dim)

        output = self.decoder(fused_features)  # (batch_size, window_size, num_classes)

        # 转置回序列格式: (batch_size, window_size, num_classes)
        return output


# Encoder+Decoder编码器-注意力机制，要求两个encoder输出维度相同
class MultiModeSeq2SeqModel2(nn.Module):
    def __init__(self, input_size, num_classes, window_size, d_model=128,n_head=8,dim_feedforward=128,n_layer=1,dropout=0.1,vit_type="vit_tiny_patch16_224",image_method='log_curve',system="windows",weight_path=None):
        super(MultiModeSeq2SeqModel2, self).__init__()
        self.window_size = window_size
        self.method = image_method
        self.attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=-2)
        )

        # 数值模态编码器 - 保持时间维度
        # 1) 输入投影：把数值特征 F -> d_model（Transformer 的 token embedding）
        self.dim_token = nn.Linear(input_size, d_model)

        # 2) 位置编码（必须有，否则 Transformer 不知道顺序）
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(5000, window_size + 1))

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 让输入保持 (B, T, d_model)
            # norm_first=True     # 更稳定（Pre-LN 风格）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 图像模态的ViT特征提取器
        if system == "windows":
            self.image_feature_extractor = ViTEncoder(pretrained=True, vit_type=vit_type, image_method=self.method,
                                                      output_dim=d_model)
        else:
            self.image_feature_extractor = ViTEncoder2(pretrained=False, vit_type=vit_type, image_method=self.method,
                                                       output_dim=d_model, weight_path=weight_path)

        if self.method in ["log_curve","wave_trans","gaf_trans"]:
            self.image_expand_windows = nn.Linear(input_size, window_size)
        else:
            self.image_expand_windows = nn.Linear(1, window_size)

        # 解码器 - 使用1x1卷积进行时间步预测
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, image_features):
        # x形状: (batch_size, window_size, input_size)
        x = self.dim_token(x)  # (B, T, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 编码（不加mask：默认窗口内全注意力）
        encoded = self.encoder(x)  # (B, T, d_model)

        # 解码 - 直接预测每个时间步
        image_expand_features = self.image_expand_windows(image_features)  # (batch_size, hidden_dim, window_size)

        # 模态融合
        fused = torch.stack([encoded, image_expand_features.transpose(1, 2)], dim=-2)  # (batch_size, window_size, 2, hidden_dim)
        attention_weights = self.attention(fused)  # (batch_size, window_size, 2 , 1)

        fused_features = (fused * attention_weights).sum(dim=-2)

        output = self.decoder(fused_features)  # (batch_size, window_size, num_classes)

        # 转置回序列格式: (batch_size, window_size, num_classes)
        return output


# Encoder+Decoder编码器-加权融合，要求两个encoder输出维度相同
class MultiModeSeq2SeqModel3(nn.Module):
    def __init__(self, input_size, num_classes, window_size, d_model=64,n_head=4,dim_feedforward=128,num_layers=1,dropout=0.1,weight_num=0.8,weight_img=0.2,vit_type="vit_tiny_patch16_224",image_method='log_curve',system="windows",weight_path=None):
        super(MultiModeSeq2SeqModel3, self).__init__()
        self.window_size = window_size
        self.method = image_method
        self.weight_num = weight_num
        self.weight_img = weight_img

        # 数值模态编码器 - 保持时间维度
        # 1) 输入投影：把数值特征 F -> d_model（Transformer 的 token embedding）
        self.dim_token = nn.Linear(input_size, d_model)

        # 2) 位置编码（必须有，否则 Transformer 不知道顺序）
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(5000, window_size + 1))

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 让输入保持 (B, T, d_model)
            # norm_first=True     # 更稳定（Pre-LN 风格）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 图像模态的ViT特征提取器
        if system == "windows":
            self.image_feature_extractor = ViTEncoder(pretrained=True, vit_type=vit_type, image_method=self.method,
                                                      output_dim=d_model)
        else:
            self.image_feature_extractor = ViTEncoder2(pretrained=False, vit_type=vit_type, image_method=self.method,
                                                       output_dim=d_model, weight_path=weight_path)

        if self.method in ["log_curve","wave_trans","gaf_trans"]:
            self.image_expand_windows = nn.Linear(input_size, window_size)
        else:
            self.image_expand_windows = nn.Linear(1, window_size)

        # 解码器 - 使用1x1卷积进行时间步预测
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, image_features):
        # x形状: (batch_size, window_size, input_size)
        x = self.dim_token(x)  # (B, T, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 编码（不加mask：默认窗口内全注意力）
        encoded = self.encoder(x)  # (B, T, d_model)

        # 解码 - 直接预测每个时间步
        image_expand_features = self.image_expand_windows(image_features)  # (batch_size, hidden_dim, window_size)

        # 模态融合
        fused_features = self.weight_num*encoded + self.weight_img*image_expand_features.transpose(1, 2)  # (batch_size, window_size, 2, hidden_dim)

        output = self.decoder(fused_features)  # (batch_size, window_size, num_classes)

        # 转置回序列格式: (batch_size, window_size, num_classes)
        return output


# Encoder+Decoder编码器-交叉自注意力融合，要求两个encoder输出维度相同
class MultiModeSeq2SeqModel4(nn.Module):
    def __init__(self, input_size, num_classes, window_size, d_model=64,n_head=4,dim_feedforward=128,num_layers=1,dropout=0.1,vit_type="vit_tiny_patch16_224",image_method='log_curve',fuse_heads=10,system="windows",weight_path=None):
        super(MultiModeSeq2SeqModel4, self).__init__()
        self.window_size = window_size
        self.method = image_method
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=fuse_heads)

        # 数值模态编码器 - 保持时间维度
        # 1) 输入投影：把数值特征 F -> d_model（Transformer 的 token embedding）
        self.dim_token = nn.Linear(input_size, d_model)

        # 2) 位置编码（必须有，否则 Transformer 不知道顺序）
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(5000, window_size + 1))

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 让输入保持 (B, T, d_model)
            # norm_first=True     # 更稳定（Pre-LN 风格）
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 图像模态的ViT特征提取器
        if system == "windows":
            self.image_feature_extractor = ViTEncoder(pretrained=True, vit_type=vit_type, image_method=self.method,
                                                      output_dim=d_model)
        else:
            self.image_feature_extractor = ViTEncoder2(pretrained=False, vit_type=vit_type, image_method=self.method,
                                                       output_dim=d_model, weight_path=weight_path)

        if self.method in ["log_curve","wave_trans","gaf_trans"]:
            self.image_expand_windows = nn.Linear(input_size, window_size)
        else:
            self.image_expand_windows = nn.Linear(1, window_size)

        # 解码器 - 使用1x1卷积进行时间步预测
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, image_features):
        # x形状: (batch_size, window_size, input_size)
        x = self.dim_token(x)  # (B, T, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 编码（不加mask：默认窗口内全注意力）
        encoded = self.encoder(x)  # (B, T, d_model)

        # 解码 - 直接预测每个时间步
        image_expand_features = self.image_expand_windows(image_features).transpose(1, 2)  # (batch_size, window_size, hidden_dim)

        # 模态融合
        attn_output, attn_output_weights = self.attention(encoded.transpose(0, 1), image_expand_features.transpose(0, 1), image_expand_features.transpose(0, 1))  # (seq_len, batch_size, hidden_dim)
        fused_features = attn_output.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)

        output = self.decoder(fused_features)  # (batch_size, window_size, num_classes)

        # 转置回序列格式: (batch_size, window_size, num_classes)
        return output


# 类别非均衡损失
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, epsilon=1e-7):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.epsilon = epsilon

        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = torch.ones(num_classes) * alpha

    def forward(self, logits, targets):
        self.alpha = self.alpha.to(logits.device)

        # 使用log_softmax提高数值稳定性
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # 使用gather方法高效获取目标类别概率
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        probs_t = torch.clamp((probs * targets_one_hot).sum(dim=1), min=self.epsilon, max=1.0 - self.epsilon)
        log_probs_t = torch.log(probs_t)

        alpha_t = (self.alpha * targets_one_hot).sum(dim=1)

        # Focal Loss计算
        focal_weight = torch.pow(1 - probs_t, self.gamma)
        loss = -alpha_t * focal_weight * log_probs_t
        return loss.mean()