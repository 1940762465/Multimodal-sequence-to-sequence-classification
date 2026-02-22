import time
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_seq2seq_model(model, device, train_loader=None, test_loader=None,model_type="multi"):
    """
        评估序列到序列模型
        model:训练获取到的模型
        device:模型预测的装置
        train_loader:模型训练的数据集
        test_loader:模型测试的数据集
    """
    train_accuracy, train_precision, train_recall, train_f1 = None, None, None,None
    test_accuracy, test_precision, test_recall, test_f1 = None, None, None,None
    train_preds, test_preds, train_labels, test_labels = None,None,None,None
    model.eval()

    def get_predictions(loader,model_type,device):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y, batch_images in loader:
                batch_X = batch_X.float().to(device)
                batch_y = batch_y.long().to(device)
                if model_type in ["multi_concat","multi_att","multi_weight","multi_cross_att"]:
                    batch_images = batch_images.float().to(device)
                    # 提取图像特征
                    image_features = model.image_feature_extractor(batch_images).to(device)
                    outputs = model(batch_X,image_features)
                elif model_type in ["numeric_CNN", "numeric", "numeric_BiLSTM", "numeric_Transformer"]:
                    # 数值模态分类器
                    outputs = model(batch_X)
                else:
                    # 图像模态分类器
                    batch_images = batch_images.float().to(device)
                    image_features = model.image_feature_extractor(batch_images).to(device)
                    outputs = model(image_features)
                _, predicted = torch.max(outputs.data, 2)
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())
        return np.array(all_preds), np.array(all_labels)

    def calculate_metrics(preds, labels):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    if train_loader:
        train_preds, train_labels = get_predictions(train_loader,model_type,device)
        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_preds, train_labels)
    if test_loader:
        test_preds, test_labels = get_predictions(test_loader,model_type,device)
        test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_preds, test_labels)

    metrics = {
        'train': {'accuracy': train_accuracy, 'precision': train_precision, 'recall': train_recall, 'f1': train_f1},
        'test': {'accuracy': test_accuracy, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1}
    }

    return metrics, train_preds, train_labels, test_preds, test_labels