"""
参数配置文件
对于调参，现在一个随机种子上调，调参后再在多个随机种子上跑平均实验
目前实验下来，加权融合以及transformer做编码器效果最好
加入另外几个图像识别编码器，如resnet50等等
"""
import torch


model_status = "all"  # "train" or "test"(pre) or all
model_type = 'numeric_Transformer'  # multi_cross_att-多模态交叉自注意力机制,multi_weight-多模态加权求和,multi_concat-多模态concat融合, multi_att-多模态注意力机制融合, numeric_CNN-数值模态CNN, numeric_BiLSTM-数值模态双向LSTM, numeric_Transformer-数值模态transformer, img_ViT-图像模态ViT, img_resnet-图像编码器ViT
system_env = 'windows'  # linux/windows
blind_well = True

# 单井输入
file_path = "../data/lr_data_2025_11_20.xlsx"

# 多井输入
# file_path = "../data/load_las/xlsx"
well_num_select = 1  # 多井输入数量

method = 'matrix_visual'  # "log_curve","wave_trans",'matrix_visual','gaf_trans'-不同转换数值模态为图像模态的方法
img_encoder_type = "vit_tiny_patch16_224"  # "vit_tiny_patch16_224","vit_base_patch16_224",resnet18,resnet34

load_weight_path = f"../img_encoder_weight/{img_encoder_type}/model.safetensors"  # ViT模型权重加载

img_trans_status = True  # True代表模型进行图像转换并保存
save_img_path = f"../data/{method}/image"  # 保存转换后的图像路径,image单井数据图像，multi_well_image多井数据图像

save_pt_path = ""  # 保存加载的数据集图像torch张量,f"../data/{method}"

save_model_path = "../result/model_param_save/best_model.pth"  # 模型参数保存路径--跳过训练步骤包含 名称.pth，需要指定保存路径 ../result/model_param_save/model.pth


random_seed = 42  # 随机数种子
num_layers = 1  # 网络编码器层数-适用于LSTM与Transformer
window_size = 10  # 窗长
stride = 10  # 滑动步长
weight_num = 0.8  # 加权融合数值模态权重
weight_img = 0.2  # 加权融合图像模态权重
wave_type = "morl"  # 连续小波变换类型
scale = None  # 小波变换尺度-'morl'、'cmor1.5-1.0'、'mexh'、'gaus4'
batch_size = 16  # 分块大小
learning_rate = 0.0008  # 学习率
epochs = 90  # 训练轮次
img_feature_dim = 256  # Vit特征输出变换维度
hidden_dim = 64  # 数值编码器输出维度
dropout = 0.1  # 训练时，神经元失活比例

# Transformer参数
d_model = 64  # 伪token化的embedding维度，能够整除n_head
n_head = 4  # transformer自注意力头数
fused_head = 2  # 交叉自注意力机制融合注意力头数
dim_feedforward = 128  # 前馈网络（FFN / Mlp）中间隐藏层维度

patience = 8  # 学习率调度器参数
factor = 0.5  # 学习率调度器参数
max_norm = 0.5  # 梯度裁剪参数
alpha = [0.3, 0.3, 0.6, 0.4, 0.35, 0.6, 0.6, 0.6]  # 测井数据2025-[0.3, 0.3, 0.5, 0.4, 0.35, 0.5, 0.5, 0.5]，focal loss调节非均衡每个类别关注度，list长度取决类别数量
gamma = 2
weight_decay = 1e-4  # L2正则化参数

valid_size = 0.1  # 验证集比例
test_size = 0.2  # 测试集比例
max_workers = 50  # 保存图像的最大并发数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练位置


