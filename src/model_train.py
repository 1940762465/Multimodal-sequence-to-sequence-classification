from model import *
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy


def train_model(model, train_loader, valid_loader, num_classes,device, patience=8, factor=0.5,
                max_norm=0.5,alpha=None,gamma=2,weight_decay=1e-4,num_epochs=60, learning_rate=0.001,
                model_type="multi", save_model_name="../result/model_param_save/best_model.pth"):
    """
    Train a model with a specified number of epochs.
    :param model: model to be trained--训练的网络模型结构
    :param train_loader: train data loader--数据加载
    :param valid_loader: valid data loader--数据加载
    :param num_classes: number of classes--数据分类
    :param device:模型用于训练的装置--GPU or CPU
    :param patience: 学习率调度器进行学习率变化的epoch轮数
    :param factor: 学习率衰减比例
    :param max_norm: max norm of gradients--梯度裁剪
    :param alpha: focal loss参数,控制每个类别关注的非均衡占比，为list数组
    :param gamma: focal loss参数
    :param weight_decay: weight decay-L2正则
    :param num_epochs: number of epochs--训练epoch
    :param learning_rate: learning rate--学习率
    :param save_best_model: 是否需要保存模型的最佳参数
    :return: trained model,train_loss,valid_loss
    """
    if alpha is None:
        alpha=[0.3, 0.3, 0.5, 0.4, 0.35, 0.5, 0.5, 0.5]
    criterion = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes).to(
        device)  # 多少个类别，给多少个alpha，且为0-1之间，避免非均衡
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)

    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    best_valid_f1 = float('-inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y, batch_images in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.long().to(device)
            if model_type in ["multi_concat","multi_att","multi_weight","multi_cross_att"]:
                batch_images = batch_images.float().to(device)
                # 提取图像特征
                image_features = model.image_feature_extractor(batch_images).to(device)
                outputs = model(batch_X, image_features)
            elif model_type in ["numeric_CNN", "numeric", "numeric_BiLSTM", "numeric_Transformer"]:
                # 数值模态分类器
                outputs = model(batch_X)
            else:
                # 图像模态分类器
                batch_images = batch_images.float().to(device)
                image_features = model.image_feature_extractor(batch_images).to(device)
                outputs = model(image_features)

            # 展平序列维度计算损失
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_y.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            train_loss += loss.item()

        # 测试阶段
        val_preds = []
        val_labels = []
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y, batch_images in valid_loader:
                batch_X = batch_X.float().to(device)
                batch_y = batch_y.long().to(device)
                if model_type in ["multi_concat","multi_att","multi_weight","multi_cross_att"]:
                    # 多模态分类器
                    batch_images = batch_images.float().to(device)
                    image_features = model.image_feature_extractor(batch_images).to(device)
                    outputs = model(batch_X, image_features)
                elif model_type in ["numeric_CNN","numeric","numeric_BiLSTM","numeric_Transformer"]:
                    # 数值模态分类器
                    outputs = model(batch_X)
                else:
                    # 图像模态分类器
                    batch_images = batch_images.float().to(device)
                    image_features = model.image_feature_extractor(batch_images).to(device)
                    outputs = model(image_features)
                _, predicted = torch.max(outputs.data, 2)
                val_preds.extend(predicted.cpu().numpy().flatten())
                val_labels.extend(batch_y.cpu().numpy().flatten())
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_y.reshape(-1))
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_train_loss = train_loss / len(train_loader)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        # 学习率调度
        scheduler.step(avg_valid_loss)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        # 保存最佳模型-按损失保存最佳模型以及按评价指标保存最佳模型
        # if avg_valid_loss < best_valid_loss:
        #     best_valid_loss = avg_valid_loss
        #     best_model_state = model.state_dict().copy()
        if val_f1 > best_valid_f1:
            best_valid_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid F1: {val_f1:.4f}, best Valid F1: {best_valid_f1:.4f}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型参数")
        if save_model_name:
            torch.save(best_model_state, f"{save_model_name}")
            print(f"最佳模型已保存至 {save_model_name}")

    return train_losses, valid_losses, best_model_state