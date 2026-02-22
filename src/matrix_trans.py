import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib as mpl
import cv2
import pywt
from PIL import Image
from data_processing import *
import torch
from torchvision import transforms
from untils import *
import warnings
from torch.utils.data import DataLoader, TensorDataset
from pyts.image import GramianAngularField


class ImageTrans(object):
    def __init__(self, numeric,method,save_path,max_workers=50,wavelet='morl',scale=None,gaf_method="summation"):
        """
        不同方法，将数值矩阵转化为图像
        numeric:输入的数值矩阵
        method:转化为图像的方法
        save_path:图像保存地址
        max_worker:图像保存最大并发次数
        wavelet:小波变换类型选择
        scale:变换尺度
        gaf_method:GAF二维变换，默认角和场
        """
        self.numeric = numeric
        self.method = method
        self.n_features = self.numeric.shape[2]
        self.save_path = save_path
        self.wave_type = wavelet
        self.max_workers = max_workers
        self.scale = scale
        # GAF transformer（第三方库，简单清晰）
        self.gaf = GramianAngularField(method=gaf_method)

        allowed_methods = ['log_curve','matrix_visual','wave_trans',"gaf_trans"]
        if allowed_methods == "wave_trans":
            print(f"小波类型为{self.wave_type},尺度大小为{self.scale},不同小波类型应使用不同的尺度")
        # 设置报错
        if method not in allowed_methods:
            # 抛出自定义异常，而不是通用的 ValueError
            raise ValueError(f"方法参数命名'{method}' 无效。请使用以下方法之一：{allowed_methods}")

    def create_curve_image(self,window_data, feature_idx, img_size=(224, 224), line_width=2):
        """
        method1:从窗口数据创建单个特征的无坐标曲线图像
        Args:
            window_data: 窗口数据 (window_size, n_features)
            feature_idx: 特征索引
            img_size: 图像大小
            line_width: 曲线宽度
        """
        # 获取单个特征的时间序列
        feature_values = window_data[:, feature_idx]
        window_size = len(feature_values)

        # 创建空白的白色背景图像
        img = np.ones((img_size[0], img_size[1]), dtype=np.uint8) * 255

        # 归一化特征值到图像高度范围内
        y_min, y_max = np.min(feature_values), np.max(feature_values)

        # 处理常数值情况
        if y_max - y_min < 1e-10:
            y_range = 1.0
        else:
            y_range = y_max - y_min

        # 将特征值映射到图像坐标
        x_coords = np.linspace(10, img_size[1] - 10, window_size).astype(int)
        y_norm = (feature_values - y_min) / y_range
        y_coords = (img_size[0] - 10 - y_norm * (img_size[0] - 20)).astype(int)

        # 绘制曲线
        for i in range(window_size - 1):
            cv2.line(img, (x_coords[i], y_coords[i]),
                     (x_coords[i + 1], y_coords[i + 1]),
                     color=0,  # 黑色
                     thickness=line_width)

        return img

    def normalize_2d_matrix(self,matrix):
        """
        method2:将二维矩阵整体归一化到0-255范围
        matrix:单个窗口样本的矩阵
        """
        flattened = matrix.flatten()
        min_val = flattened.min()
        max_val = flattened.max()

        if max_val - min_val <= 1e-5:
            normalized = np.full(matrix.shape, 128, dtype=np.uint8)
        else:
            normalized = ((matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return normalized

    def wavelet_transform(self, window_data, feature_idx):
        """
        method3:将3D时序数组转换为多通道CNN输入张量

        参数:
        window_data: 2D numpy数组 [序列长度, 特征维度]
        wavelet: 使用的连续小波类型
             'morl'
            'cmor1.5-1.0'
            'mexh'
            'gaus4'
        mode: 小波变换模式
        scales: 小波变换的尺度数

        返回:
        cnn_tensor: 4D张量 [样本量, 高度, 宽度, 通道数]
        """
        # 初始化结果张量
        signal_data = window_data[:, feature_idx]
        if self.scale:
            scales = self.scale
        else:
            scales = len(signal_data)  # 选择变换尺度<len(signal_data)

        # 连续小波变换
        coefficients, frequencies = pywt.cwt(signal_data, scales=np.arange(1, scales + 1),
                                             wavelet=self.wave_type, sampling_period=1)

        # 归一化并转换为图像格式
        cwt_matrix = np.abs(coefficients)

        # 归一化到0-255范围
        cwt_normalized = ((cwt_matrix - cwt_matrix.min()) /
                          (cwt_matrix.max() - cwt_matrix.min()) * 255).astype(np.uint8)

        return cwt_normalized

    def gaf_transform(self, window_data, feature_idx):
        """将单特征序列 -> GAF 图像（uint8）"""
        x = window_data[:, feature_idx].reshape(1, -1)     # (1, window_size)
        gaf_img = self.gaf.fit_transform(x)[0]             # (window_size, window_size)

        # pyts输出一般为float（[-1,1]左右），映射到0-255
        gaf_uint8 = ((gaf_img + 1) * 0.5 * 255).astype(np.uint8)
        return gaf_uint8

    def save_curve_images(self, id, window_data, n_features):
        """
        保存所有窗口的所有特征曲线图像
        Args:
            id: 窗口编号
            window_data: 数据
            n_features: 特征数量
        """
        if self.method == "log_curve":
            # 为每个特征创建并保存曲线图像
            for feature_idx in range(n_features):
                img = self.create_curve_image(window_data, feature_idx)
                # 保存图像
                filename = f"feature_{feature_idx}_{id}.png"
                filepath = os.path.join(self.save_path, filename)
                cv2.imwrite(filepath, img)

        if self.method == "matrix_visual":
            image_array = self.normalize_2d_matrix(window_data)
            # 展示图片矩阵数值
            filename = f"{id}.png"
            filepath = os.path.join(self.save_path, filename)
            # 重塑图像，加快图像保存速度
            image_pil = Image.fromarray(image_array)  # 转换为PIL图像（像素值0-255）
            image_resized = image_pil.resize((224, 224))  # 修改图像尺寸为 200x200
            # 将调整后的图像转换回 NumPy 数组
            image_resized_array = np.array(image_resized)
            # 保存调整后的图像
            plt.imsave(filepath, image_resized_array, cmap='jet', dpi=300)

        if self.method == "wave_trans":
            for feature_idx in range(n_features):
                img = self.wavelet_transform(window_data, feature_idx)
                filename = f"feature_{feature_idx}_{id}.png"
                filepath = os.path.join(self.save_path, filename)
                image_pil = Image.fromarray(img)  # 转换为PIL图像（像素值0-255）
                image_resized = image_pil.resize((224, 224))  # 修改图像尺寸为 200x200
                # 将调整后的图像转换回 NumPy 数组
                image_resized_array = np.array(image_resized)
                # 保存调整后的图像
                plt.imsave(filepath, image_resized_array, cmap='jet', dpi=300)

        if self.method == "gaf_trans":
            for feature_idx in range(n_features):
                img = self.gaf_transform(window_data, feature_idx)
                filename = f"feature_{feature_idx}_{id}.png"
                filepath = os.path.join(self.save_path, filename)

                image_pil = Image.fromarray(img)  # 0-255
                image_resized = image_pil.resize((224, 224))
                image_resized_array = np.array(image_resized)

                plt.imsave(filepath, image_resized_array, cmap='jet', dpi=300)

    def trans_fig(self):
        mpl.rcParams['figure.max_open_warning'] = self.max_workers+10
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 保存图像
            for i, window_data in enumerate(self.numeric):
                executor.submit(self.save_curve_images, i, window_data, self.n_features)
        print(f"{self.method}方法图像模态转换已完成")

    def dataset_image2pt(self,save_pt_dir,split_idx,image_transform,dataset_name):
        """
            根据划分数据集的索引选取对应的图像数据,保存为torch的.pt格式
            save_pt_dir:.pt缓存文件保存地址
            split_idx：划分数据集对应的原始索引
            random_seed:随机种子，用于.pt缓存命名
            image_transform:图像数据预处理
            dataset_name:保存pt的命名
        """
        # os.makedirs(save_pt_dir, exist_ok=True)
        all_images = []
        if self.method in ["log_curve","wave_trans","gaf_trans"]:
            for sample_idx in split_idx:
                sample_images = []
                for feature_idx in range(self.n_features):  # 特征数量
                    image_path = os.path.join(self.save_path, f"feature_{feature_idx}_{sample_idx}.png")
                    image = Image.open(image_path).convert("RGB")
                    # 图像预处理
                    if image_transform:
                        image = image_transform(image)
                    sample_images.append(image)
                all_images.append(torch.stack(sample_images, dim=0))

        if self.method == "matrix_visual":
            image_paths = [os.path.join(self.save_path, f"{i}.png") for i in split_idx]
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                image = image_transform(image)  # 应用图像预处理
                all_images.append(image)

        # 将图像转换为Tensor并保存
        all_images_tensor = torch.stack(all_images, dim=0)
        # if os.path.exists(os.path.join(save_pt_dir, f"{dataset_name}.pt")):
        #     warnings.warn("当前命名的pt已存在，不会再次执行保存", UserWarning)
        # else:
        #     torch.save(all_images_tensor, os.path.join(save_pt_dir, f"{dataset_name}.pt"))
        return all_images_tensor

    def load_image_pt(self,save_pt_dir,dataset_name):
        images_torch = torch.load(os.path.join(save_pt_dir, f"{dataset_name}.pt"), weights_only=True)
        return images_torch


def creat_dataloaders(img_trans_class,features,label,img_process,idx,save_pt_path,batch_size,dataset_name,save=True):
    """
    创建相应的dataloaders
    img_trans_class:之前保存的图片转换的类
    img_process:图像预处理流程
    idx:对应数据的索引
    save_pt_path:数据集图片保存加载地址
    save:是否保存为pt文件
    """
    # if save:
    #     img_trans_class.dataset_image2pt(save_pt_dir=save_pt_path, split_idx=idx, image_transform=img_process,
    #                                  dataset_name=dataset_name)
    # image_torch = img_trans_class.load_image_pt(save_pt_dir=save_pt_path, dataset_name=dataset_name)
    image_torch = img_trans_class.dataset_image2pt(save_pt_dir=save_pt_path, split_idx=idx, image_transform=img_process,
                                                   dataset_name=dataset_name)
    # 自定义数据集加载
    # dataset = ModalDataset(image_torch, features, label)
    # loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)  # 多进程处理__getitem__中的转换操作
    dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(label).long(), image_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False) # drop_last=True
    return loader


if __name__ == '__main__':
    window_size = 4
    stride = 4
    batch_size = 16
    random_seed = 42

    file_path = "../data/lr_data_2025_11_20.xlsx"
    well_data,num_class = read_and_preprocess(file_path)
    X, y= creat_windows(well_data, window_size, stride)
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = train_dataset_split(X, y,
                                                                                                       random_seed=random_seed)

    # 图片数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalization using ImageNet stats
    ])

    image_trans = ImageTrans(X, 'wave_trans', "../data/wave_trans/image", 10)
    # 对数据集所有窗口进行图片转换
    image_trans.trans_fig()
    # 保存对应数据集的图片数据--目前只对训练集进行测试
    # train_loaders = creat_dataloaders(img_trans_class=image_trans,features=X_train,label=y_train,
    #                                   img_process=transform,idx=idx_train,save_pt_path="../result",batch_size=batch_size,
    #                                   dataset_name=f"image_train{random_seed}",save=True)
    # valid_loaders = creat_dataloaders(img_trans_class=image_trans,features=X_val,label=y_val,
    #                                   img_process=transform,idx=idx_val,save_pt_path="../result",batch_size=batch_size,
    #                                   dataset_name=f"image_valid{random_seed}",save=True)
    # test_loaders = creat_dataloaders(img_trans_class=image_trans, features=X_test, label=y_test,
    #                                   img_process=transform, idx=idx_test, save_pt_path="../result",
    #                                   batch_size=batch_size,dataset_name=f"image_test{random_seed}", save=True)






