"""
后续需要加入对不同的井进行抽取的预处理过程
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import lasio
import os
from concurrent.futures import ThreadPoolExecutor,as_completed


#===============================单口井的数据预处理方法===============================#
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
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"编码 {i} -> 原始标签 '{class_name}'")
    else:
        num_classes = None

    if normal:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        well_data = np.column_stack([features_scaled, labels])
    else:
        well_data = np.column_stack([features, labels])

    return well_data,num_classes


def creat_windows(data, window_size, stride):
    """
        创建滑动窗口数据集，windows=stride则代表固定窗长划分
        data:原始数据
        windows_size:窗长
        stride：滑动步长
    """
    features, labels = [], []

    # 生成滑动窗口样本
    for i in range(0, len(data) - window_size, stride):
        # 窗口内的特征（前8列）
        window_features = data[i:i + window_size, :-1]

        # 窗口内所有时间点的标签
        window_labels = data[i:i + window_size, -1]

        # # 计算每个类别的数量-label取众数
        # unique, counts = np.unique(window_labels, return_counts=True)
        # # 选择数量最多的类别作为标签
        # window_label = unique[np.argmax(counts)]

        features.append(window_features)
        labels.append(window_labels)
    return np.array(features),np.array(labels)


#===============================当前地区所有井的合并===============================#
def las_processing(folder_path,las_file):
    """
    对每个las文件样本进行过滤，排除测井曲线数值为空的样本点深度
    """
    las_df = lasio.read(f"{folder_path}/las/{las_file}").df()

    # 查看列名
    # print("DataFrame列名:")
    # print(las_df.columns.tolist())

    col_name = ["SP","GR","CALI","NPHI","RHOB","DTC","RDEP","RSHA","RMED","FORCE_2020_LITHOFACIES_LITHOLOGY"]
    las_df = las_df[col_name].dropna(axis=0, how='any')

    las_df.to_excel(f"{folder_path}/xlsx/{os.path.splitext(las_file)[0]}.xlsx", index=False)


def las2xls(folder_path):
    """
    对las数据转换为xlsx数据，选取所有测井曲线都有的深度样本点，分别转换为xlsx
    file_dir：文件所在目录地址
    """
    las_files = [f for f in os.listdir(f"{folder_path}/las") if f.endswith('.las')]
    # 使用多线程并行处理每个.las文件
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 为每个.las文件调用las_processing
        executor.map(lambda las_file: las_processing(folder_path, las_file), las_files)


# 获取目录下所有.xlsx文件
def get_xlsx_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.xlsx')]


def process_single_file(file_path, well_id, window_size, stride):
    well_data,_ = read_and_preprocess(file_path,class_label=False)
    features, labels = creat_windows(well_data, window_size, stride)

    # 添加well_id，表示当前样本来自哪个井
    well_ids = np.full(len(features), well_id)

    return features, labels, well_ids


# 主函数：按线程执行文件的读取和窗口创建
def process_files(directory, number_select=None, window_size=4, stride=4, num_threads=4):
    # 获取所有.xlsx文件
    xlsx_files = get_xlsx_files(directory)

    # 如果number_select为None，则选择所有文件，否则选择前number_select个文件
    if number_select is not None:
        xlsx_files = xlsx_files[:number_select]

    # 存储所有的数据
    all_features = []
    all_labels = []
    all_well_ids = []

    # 用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, file in enumerate(xlsx_files):
            file_path = f"{directory}/{file}"
            futures.append(executor.submit(process_single_file, file_path, i + 1, window_size, stride))

        # 等待所有线程完成
        for future in as_completed(futures):
            well_features, well_labels, well_ids = future.result()
            all_features.append(well_features)
            all_labels.append(well_labels)
            all_well_ids.extend(well_ids)

    # 将所有窗口数据拼接成一个整体
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 对拼接后的labels进行展平后编码
    all_labels_flat = all_labels.flatten()

    # 标签编码
    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels_flat)

    # 还原为原来形状
    all_labels = all_labels_encoded.reshape(all_labels.shape)
    num_classes = len(label_encoder.classes_)

    # 对应关系
    # for original, encoded in zip(all_labels_encoded, range(len(all_labels_encoded))):
    #     print(f"Original: {original}, Encoded: {encoded}")

    return all_features, all_labels, all_well_ids, num_classes


#===============================滑动窗口的数据预处理方法===============================#


# 数据集划分
def train_dataset_split(feature, label, well_ids=None, random_seed=42, test_ratio=0.2, valid_ratio=None):
    """
    适用于大规模数据集的训练集，验证集，测试集的划分。对于小规模数据集进行k折交叉验证。按井的深度点个数分层抽取训练集、验证集，以及测试集
    feature:特征
    label:标签
    ori_idx:原始id索引
    well_ids: 井对应的索引
    test_ratio:测试集占比
    valid_ratio:验证集占比
    """
    ori_idx = np.arange(feature.shape[0])
    if well_ids is not None:
        # 按不同井样本比例进行分层划分
        well_ids = np.array(well_ids)
        X_train, X_test, y_train, y_test, idx_train, idx_test,well_ids_train,well_ids_test = train_test_split(
            feature, label, ori_idx, well_ids, stratify=well_ids, test_size=test_ratio, random_state=random_seed, shuffle=True)
        if valid_ratio:
            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
                X_train, y_train, idx_train, stratify=well_ids_train, test_size=valid_ratio, random_state=random_seed, shuffle=True)
            return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test
        else:
            return X_train, X_test, y_train, y_test, idx_train, idx_test
    else:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            feature, label, ori_idx, test_size=test_ratio, random_state=random_seed, shuffle=True)
        if valid_ratio:
            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
                X_train, y_train, idx_train, test_size=valid_ratio, random_state=random_seed, shuffle=True)
            return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test
        else:
            return X_train, X_test, y_train, y_test, idx_train, idx_test


if __name__ == '__main__':
    number_select = 5
    window_size = 4
    stride = 4

    ## 多井数据操作
    # las2xls("../data/load_las")  # 转换为.xlsx文件
    # directory = '../data/load_las/xlsx'
    # all_features, all_labels, all_well_ids,number_class = process_files(directory, number_select, window_size, stride, num_threads=number_select)
    # X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = train_dataset_split(all_features, all_labels, all_well_ids)

    ## 单井数据操作
    file_path = "../data/lr_data_2025_11_20.xlsx"
    well_data,num_class = read_and_preprocess(file_path)

    # X, y= creat_windows(well_data, window_size, stride)
    # X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = train_dataset_split(X, y)
    # print(X_train.shape, X_val.shape, X_test.shape)