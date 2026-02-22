import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
import random
import time
import copy


def read_and_preprocess(file_path,normal=True,class_label=True):
    """
    数据读取与规范化--固定窗长
    file_path:文件路径，最后一列为标签
    normal：是否使用归一化
    """
    df = pd.read_excel(file_path)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    if class_label:
        # 对标签进行编码，防止后续模型训练损失计算错误
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
    else:
        num_classes = None

    if normal:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        well_data = np.column_stack([features_scaled, labels])
    else:
        well_data = np.column_stack([features, labels])

    return well_data,num_classes


def split_random_segments(num_rows, test_ratio, min_segment_length, random_state):
    # 设置随机种子
    if random_state is not None:
        random.seed(random_state)

    max_N = num_rows // min_segment_length

    # 取不大于 max_N 的最大5的倍数作为N
    N = (max_N // 5) * 5
    if N == 0:
        raise ValueError("给定的最小片段长度太大，无法满足条件。")

    segment_len = num_rows // N
    remainder = num_rows % N

    # 划分为N个连续等长片段，保存每个片段的索引
    segments = []
    for i in range(N):
        start = i * segment_len
        end = start + segment_len
        segment_indices = list(range(start, end))
        segments.append(segment_indices)

    # 随机打乱片段
    random.shuffle(segments)

    # 80% 片段用于训练，20% 用于测试
    train_segments = segments[:int(N * (1 - test_ratio))]
    test_segments = segments[int(N * (1 - test_ratio)):]

    # 收集训练和测试集索引
    train_indices = [idx for seg in train_segments for idx in seg]
    test_indices = [idx for seg in test_segments for idx in seg]

    # 将剩余行（remainder）加入训练集中
    if remainder > 0:
        remaining_indices = list(range(num_rows - remainder, num_rows))
        train_indices.extend(remaining_indices)

    return train_indices, test_indices


def split_continuous_segments(indices):
    """
    把一组索引列表划分成多个连续的段。
    例如: [1, 2, 3, 7, 8] -> [[1,2,3], [7,8]]
    """
    indices = np.sort(np.array(indices))
    segments = []
    segment = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            segment.append(indices[i])
        else:
            segments.append(segment)
            segment = [indices[i]]
    segments.append(segment)
    return segments


def create_sliding_dataset(X, Y, segment_idx, window_size=5, stride=1):
    X = np.asarray(X[segment_idx])
    Y = np.asarray(Y[segment_idx])

    A = []
    B = []
    ID = []

    for i in range(0, len(X) - window_size + 1, stride):
        a_i = X[i:i + window_size]
        b_i = Y[i:i + window_size]
        ID_i = segment_idx[i:i + window_size]
        A.append(a_i)
        B.append(b_i)
        ID.append(ID_i)

    A = np.stack(A)
    B = np.stack(B)
    ID = np.stack(ID)

    return A, B, ID


def load_data(X, y, test_size=0.2, window_size=5, stride=1,min_segment_length=50 ,random_state=42):
    train_idx, test_idx = split_random_segments(num_rows=len(X),
                                                test_ratio=test_size,
                                                min_segment_length=min_segment_length,
                                                random_state=random_state)

    X_train1, y_train1, ID_train1 = [], [], []
    split_train_idx = split_continuous_segments(train_idx)
    for segment_idx in split_train_idx:
        if len(segment_idx) >= window_size:
            X_segment, y_segment, ID_segment = create_sliding_dataset(X, y, segment_idx, window_size=window_size)
            X_train1.append(X_segment)
            y_train1.append(y_segment)
            ID_train1.append(ID_segment)
        else:
            for single_idx in segment_idx:
                X_segment = np.tile(X[single_idx], (window_size, 1))
                X_segment = X_segment[np.newaxis, :, :]
                y_segment = np.full(window_size, y[single_idx]).reshape(1, -1)
                X_train1.append(X_segment)
                y_train1.append(y_segment)
                ID_train1.append(single_idx.reshape(1, -1))
                print(1)
    X_train = np.concatenate(X_train1, axis=0)
    y_train = np.concatenate(y_train1, axis=0)
    ID_train = np.concatenate(ID_train1, axis=0)

    X_test1, y_test1, ID_test1 = [], [], []
    split_test_idx = split_continuous_segments(test_idx)
    for segment_idx in split_test_idx:
        if len(segment_idx) >= window_size:
            X_segment, y_segment, ID_segment = create_sliding_dataset(X, y, segment_idx, window_size=window_size,stride=stride)
            X_test1.append(X_segment)
            y_test1.append(y_segment)
            ID_test1.append(ID_segment)
        else:
            for single_idx in segment_idx:
                X_segment = np.tile(X[single_idx], (window_size, 1))
                X_segment = X_segment[np.newaxis, :, :]
                y_segment = np.full(window_size, y[single_idx]).reshape(1, -1)
                X_test1.append(X_segment)
                y_test1.append(y_segment)
                ID_test1.append(single_idx.reshape(1, -1))
                print(1)
    X_test = np.concatenate(X_test1, axis=0)
    y_test = np.concatenate(y_test1, axis=0)
    ID_test = np.concatenate(ID_test1, axis=0)

    # 转为 Tensor
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.long)
    # y_test = torch.tensor(y_test, dtype=torch.long)
    # ID_train = torch.tensor(ID_train, dtype=torch.long)
    # ID_test = torch.tensor(ID_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, ID_train, ID_test


def normalize_2d_matrix(matrix):
    flattened = matrix.flatten()
    min_val = flattened.min()
    max_val = flattened.max()

    if max_val - min_val <= 1e-5:
        normalized = np.full(matrix.shape, 128, dtype=np.uint8)
    else:
        normalized = ((matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized


def creat_dataloaders(numeric,y,idx, batch_size):
    dataset = TensorDataset(torch.from_numpy(numeric).float(), torch.from_numpy(y).long(),torch.from_numpy(idx).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    return loader


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
        encoded = self.encoder(x)  # (batch_size, hidden_dim, windows_size)
        output = self.decoder(encoded.transpose(1, 2))

        return output


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


def label_reduct(pred,label,idx):
    pred, label, idx = np.array(pred), np.array(label), np.array(idx)
    pred_ori = []
    label_ori = []
    for tem_id in np.unique(idx):
        tem_preds_Single = stats.mode(pred[idx == tem_id], keepdims=True).mode[0]
        tem_y_train_Single = stats.mode(label[idx == tem_id], keepdims=True).mode[0]
        pred_ori.append(tem_preds_Single)
        label_ori.append(tem_y_train_Single)
    return pred_ori, label_ori


def evaluate_indicator(pred,label):
    accuracy = accuracy_score(label, pred)
    precision = precision_score(label, pred, average='weighted', zero_division=0)
    recall = recall_score(label, pred, average='weighted', zero_division=0)
    f1 = f1_score(label, pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def train_model(model, train_loader, valid_loader, num_classes,device, patience=8, factor=0.5,
                max_norm=0.5,alpha=None,gamma=2,weight_decay=1e-4,num_epochs=60, learning_rate=0.001,
                save_model_name="./best_model.pth"):
    if alpha is None:
        alpha=[0.3, 0.3, 0.6, 0.4, 0.35, 0.6, 0.6, 0.6]
    criterion = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes).to(
        device)  # 多少个类别，给多少个alpha，且为0-1之间，避免非均衡

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)

    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y,  batch_idx in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.long().to(device)
            # batch_idx = batch_idx.to(device)
            outputs = model(batch_X)

            # 展平序列维度计算损失
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_y.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            train_loss += loss.item()

        # 测试阶段
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y,  batch_idx in valid_loader:
                batch_X = batch_X.float().to(device)
                batch_y = batch_y.long().to(device)
                # batch_idx = batch_idx.to(device)
                outputs = model(batch_X)

                loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_y.reshape(-1))
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)

        avg_train_loss = train_loss / len(train_loader)

        # 学习率调度
        scheduler.step(avg_valid_loss)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        # 保存最佳模型
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Best Valid loss: {best_valid_loss:.4f}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型参数")
        if save_model_name:
            torch.save(best_model_state, f"{save_model_name}")
            print(f"最佳模型已保存至 {save_model_name}")

    return train_losses, valid_losses,best_model_state


def evaluate_seq2seq_model(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_idx = []
    with torch.no_grad():
        for batch_X, batch_y, batch_idx in data_loader:
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.long().to(device)
            batch_idx = batch_idx.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 2)
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            all_idx.extend(batch_idx.cpu().numpy().flatten())

    preds, labels = label_reduct(all_preds,all_labels,all_idx)
    accuracy, precision, recall, f1 = evaluate_indicator(preds,labels)
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return metrics, preds, labels


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
    plt.savefig(f"./{save_path}")  # 保存图像
    plt.close()


def clear_files_in_folder(folder_path):
    """
    清空指定文件夹中的所有文件（不删除子文件夹）

    参数:
    folder_path: 要清空文件的文件夹路径
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            return "文件夹不存在"

        if not os.path.isdir(folder_path):
            return "路径不是文件夹"

        file_count = 0

        # 遍历文件夹中的内容
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # 只删除文件，保留文件夹
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                file_count += 1

        return f"删除完成，共删除 {file_count} 个文件"

    except Exception as e:
        return f"删除文件时出错：{e}"


def main(
        file_path="../data/lr_data_2025_11_20.xlsx",
        test_ratio=0.2,
        random_seed=24,
        window_size=3, stride=3,
        batch_size=16,
        dropout=0.1,
        hidden_dim=64,
        num_epochs=80,
        lr=0.0008,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
    well_data,num_classes = read_and_preprocess(file_path,normal=False)
    # 训练用井进行固定窗口划分，再进行滑动窗口划分
    x_train, x_val, y_train, y_val, ids_train, ids_val = load_data(X=well_data[:,:-1], y=well_data[:,-1],
              test_size=test_ratio,
              window_size=window_size,
              min_segment_length=18,
              stride=stride,
              random_state=random_seed
    )

    input_size = x_train.shape[2]  # 测井曲线数量

    # 打包数据加载器
    train_loader = creat_dataloaders(x_train,y_train,ids_train,batch_size)
    valid_loader = creat_dataloaders(x_val, y_val, ids_val, batch_size)
    # test_loader = creat_dataloaders(x_test, img_test, y_test, ids_test, batch_size)

    model = SimpleCNNSeq2Seq(input_size=input_size,
                                        num_classes=num_classes,
                                        window_size=window_size,
                                        hidden_dim=hidden_dim,
                                        dropout=dropout)

    model = model.to(device)

    train_losses, valid_losses, best_model_state = train_model(model, train_loader, valid_loader, num_classes, device, patience=8, factor=0.5,
                max_norm=0.5, alpha=None, gamma=2, weight_decay=1e-4, num_epochs=num_epochs, learning_rate=lr,
                save_model_name="")
    # 损失绘制
    plot_loss_curves(train_losses, valid_losses,
                     f"numeric_cnn_loss")

    metrics_train, preds_train, labels_train = evaluate_seq2seq_model(model, device, train_loader)
    start = time.time()
    metrics_valid, preds_valid, labels_valid = evaluate_seq2seq_model(model, device, valid_loader)
    end = time.time()
    delta_time = end - start
    metrics_valid['time'] = round(delta_time, 4)
    metrics_valid['random_seed'] = random_seed
    print(f"训练集评价指标:{metrics_train}")
    print(f"验证集集评价指标:{metrics_valid}")
    # print(f"测试集集评价指标:{metrics_test}")
    clear_files_in_folder("./img/train")
    clear_files_in_folder("./img/valid")
    # clear_files_in_folder("./img/test")
    return metrics_valid


if __name__ == '__main__':
    metric_list = []
    window_size = 3
    stride = 3
    # 92
    for random_seed in [random.randint(0, 1000) for _ in range(20)]:
        metric = main(random_seed=random_seed,window_size=window_size,stride=stride)
        print('*'*100)
        metric_list.append(metric)
    metrics_df = pd.DataFrame(metric_list)
    metrics_df.to_csv(f'./20_tril_result/numeric_CNN_{window_size}_metrics.csv',
                      index=False)

