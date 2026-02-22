"""
工具函数包
"""
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import shutil


class ModalDataset(Dataset):
    def __init__(self, images, numeric, labels):
        """
        多模态数据集构建
        images: 图像模态数据（torch)
        numeric: 数值模态数据
        labels: 标签
        """
        self.numeric = torch.FloatTensor(numeric)
        self.labels = torch.LongTensor(labels)
        self.images = images

    def __len__(self):
        return len(self.numeric)

    def __getitem__(self, idx):
        # 获取数值模态数据
        window_data = self.numeric[idx]
        label_data = self.labels[idx]
        image_data = self.images[idx]  # 从内存中加载图像

        return window_data, label_data, image_data


def plot_loss_curves(train_losses, test_losses, save_path):
    """
    绘制当前随机数种子的训练和验证损失
    train_loss:训练损失
    test_loss:测试损失
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', linewidth=2)
    plt.plot(test_losses, label='Test loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../result/optimation_img/{save_path}")  # 保存图像
    plt.close()


def plot_metrics_comparison(metrics,save_path):
    """
    绘制单个随机种子训练集和验证集（测试集）的评价指标状况，验证集上进行超参数调参
    metrics：预测的指标
    save_path:保存的图片地址
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    train_values = [metrics['train']['accuracy'], metrics['train']['precision'],
                    metrics['train']['recall'], metrics['train']['f1']]
    test_values = [metrics['test']['accuracy'], metrics['test']['precision'],
                   metrics['test']['recall'], metrics['test']['f1']]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, train_values, width, label='Train set', color='skyblue', alpha=0.8)
    bars2 = plt.bar(x + width / 2, test_values, width, label='Valid set', color='lightcoral', alpha=0.8)

    plt.xlabel('Evaluation indicators')
    plt.ylabel('Score')
    plt.title('Comparison of evaluation metrics between the training set and the validation set')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.ylim(0, 1)

    for bar, v in zip(bars1, train_values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    for bar, v in zip(bars2, test_values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../result/optimation_img/{save_path}")  # 保存图像
    plt.close()


def plot_metrics_with_error_bars(df,save_path):
    """
    绘制包含误差线的柱状图，展示每个指标的均值和标准差。适用于测试集
    df: DataFrame
        col 代表不同的评价指标
        row 代表不同的随机种子跑出的实验结果
    """
    # 设置绘图风格
    sns.set_context("paper")  # 适合论文的绘图风格
    sns.set_style("whitegrid")  # 网格背景
    plt.figure(figsize=(10, 6))

    # 转换数据为长格式
    df_melted = df.melt(var_name='Metric', value_name='Score')

    # 创建柱状图
    ax = sns.barplot(data=df_melted, x='Metric', y='Score',
                     errorbar='sd',  # 使用标准差作为误差线
                     capsize=0.1,  # 误差线顶端的横线长度
                     err_kws={'color': 'black', 'linewidth': 1.5}  # 更新为err_kws参数
                     )

    # 设置图表标题和标签
    ax.set_title('Evaluation Metrics with Error Bars (Mean ± SD)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    # 美化图表
    plt.ylim(0.5, 1.0)  # 根据实际数据调整y轴范围
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_labels(df):
    # 计算标签占比
    label_counts = df.iloc[:, -1].value_counts()
    label_percentages = label_counts / label_counts.sum() * 100

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    ax = label_percentages.plot(kind='bar', color='skyblue', edgecolor='black')

    # 设置标题和标签
    plt.title('Label Distribution (Percentage)', fontsize=16)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)

    # 在柱子上方显示百分比
    for i, v in enumerate(label_percentages):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

    # 设置x轴标签旋转
    plt.xticks(rotation=45, ha='right')

    # 设置y轴刻度间隔
    plt.yticks(range(0, 51, 10))

    # 展示图像
    plt.tight_layout()
    plt.show()


def clear_files_in_folder(folder_path):
    """
    清空指定文件夹中的所有文件（不删除子文件夹）

    参数:
    folder_path: 要清空文件的文件夹路径
    """
    if not os.path.exists(folder_path):
        return "文件夹不存在"

    if not os.path.isdir(folder_path):
        return "路径不是文件夹"

        # 遍历文件夹中的所有内容
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # 如果是文件或链接，直接删除
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        # 如果是文件夹，删除整个子文件夹
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"文件夹 '{folder_path}' 中的所有内容已成功删除")


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")
    else:
        print(f"文件夹 '{folder_path}' 已存在")


if __name__ == '__main__':
    df = pd.read_excel("../data/lr_data_2025_11_20.xlsx")
    plot_labels(df)