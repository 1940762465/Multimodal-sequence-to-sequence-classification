"""
添加shape可视化解释、结果保存,多随机数种子选取
"""
from data_processing import *
from matrix_trans import *
from model import *
from model_train import *
from model_test import *
from untils import *
import config
import time
import random
import scipy.stats as stats
import shap


def multi_random():
    """
    多个随机种实验结果
    """
    metric_list = []
    for random_seed in [random.randint(0, 1000) for _ in range(20)]:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match ViT's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Normalization using ImageNet stats
        ])
        if config.well_num_select != 1:
            X, y, all_well_ids,num_class = process_files(config.file_path, config.well_num_select, config.window_size, config.stride,
                                                                   num_threads=config.well_num_select)
            X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = train_dataset_split(X,
                                                                                                               y,
                                                                                                               all_well_ids)
        else:
            well_data,num_class = read_and_preprocess(config.file_path)
            print("\n步骤1: 窗口划分...")
            X, y = creat_windows(well_data, config.window_size, config.stride)
            X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = (
                train_dataset_split(X, y,
                                    random_seed=random_seed,
                                    test_ratio=config.test_size,
                                    valid_ratio=config.valid_size))
        print("\n步骤2: 图像模态转换...")
        image_trans = ImageTrans(X, config.method, config.save_img_path, config.max_workers,config.wave_type,config.scale)
        if config.img_trans_status:
            image_trans.trans_fig()

        print("\n步骤3: 创建训练、验证、测试数据集...")
        train_loader = creat_dataloaders(img_trans_class=image_trans, features=X_train, label=y_train,
                                          img_process=transform, idx=idx_train, save_pt_path=config.save_pt_path,
                                          batch_size=config.batch_size,
                                          dataset_name=f"train_{random_seed}", save=True)

        valid_loader = creat_dataloaders(img_trans_class=image_trans, features=X_val, label=y_val,
                                          img_process=transform, idx=idx_val, save_pt_path=config.save_pt_path,
                                          batch_size=config.batch_size,
                                          dataset_name=f"valid_{random_seed}", save=True)

        test_loader = creat_dataloaders(img_trans_class=image_trans, features=X_test, label=y_test,
                                         img_process=transform, idx=idx_test, save_pt_path=config.save_pt_path,
                                         batch_size=config.batch_size, dataset_name=f"test_{random_seed}", save=True)
        input_size = X.shape[2]
        print(f"\n步骤4: {config.model_type}模型创建...")
        if config.model_type == 'multi_concat':
            # 多模态模型
            model = MultiModeSeq2SeqModel1(input_size, num_class,
                                           config.window_size, vit_type=config.img_encoder_type,
                                           image_method=config.method, dropout=config.dropout,
                                           system=config.system_env, weight_path=config.load_weight_path,
                                           d_model=config.d_model,
                                           n_head=config.n_head,
                                           n_layer=config.num_layers,
                                           dim_feedforward=config.dim_feedforward
                                           )
        elif config.model_type == "multi_att":
            model = MultiModeSeq2SeqModel2(input_size, num_class,
                                           config.window_size,
                                           vit_type=config.img_encoder_type,
                                           image_method=config.method, dropout=config.dropout,
                                           system=config.system_env, weight_path=config.load_weight_path,
                                           d_model=config.d_model,
                                           n_head=config.n_head,
                                           n_layer=config.num_layers,
                                           dim_feedforward=config.dim_feedforward
                                           )
        elif config.model_type == "multi_weight":
            model = MultiModeSeq2SeqModel3(input_size, num_class,
                                           config.window_size, weight_num=config.weight_num,
                                           weight_img=config.weight_img,
                                           vit_type=config.img_encoder_type,
                                           image_method=config.method, dropout=config.dropout,
                                           system=config.system_env, weight_path=config.load_weight_path,
                                           d_model=config.d_model,
                                           n_head=config.n_head,
                                           num_layers=config.num_layers,
                                           dim_feedforward=config.dim_feedforward,
                                           )
        elif config.model_type == "multi_cross_att":
            model = MultiModeSeq2SeqModel4(input_size, num_class,
                                           config.window_size, fuse_heads=config.fused_head,
                                           vit_type=config.img_encoder_type,
                                           image_method=config.method, dropout=config.dropout,
                                           system=config.system_env, weight_path=config.load_weight_path,
                                           d_model=config.d_model,
                                           n_head=config.n_head,
                                           num_layers=config.num_layers,
                                           dim_feedforward=config.dim_feedforward
                                           )
        elif config.model_type == 'numeric_CNN':
            # 数值模态模型
            model = SimpleCNNSeq2Seq(
                input_size=input_size,
                num_classes=num_class,
                window_size=config.window_size,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout
            )

        elif config.model_type == 'numeric_BiLSTM':
            model = SimpleLSTMSeq2Seq(input_size,
                                      num_class,
                                      hidden_dim=config.hidden_dim,
                                      num_layers=config.num_layers,
                                      bidirectional=True,  # 决定是否是双向LSTM
                                      dropout=config.dropout)
        elif config.model_type == 'numeric_Transformer':
            model = SimpleTransformerSeq2Seq(input_size=input_size,
                                             num_classes=num_class,
                                             window_size=config.window_size,
                                             d_model=config.d_model,
                                             nhead=config.n_head,
                                             num_layers=config.num_layers,
                                             dim_feedforward=config.dim_feedforward,
                                             dropout=config.dropout)
        elif config.model_type == 'img_ViT':
            model = ViTSeq2Seq(input_size, num_class, config.window_size,
                               ViT_feature_dim=config.img_feature_dim,
                               vit_type=config.img_encoder_type, image_method=config.method,
                               system=config.system_env, weight_path=config.load_weight_path)
        else:
            model = ResNetSeq2Seq(input_size, num_class, config.window_size,
                                  resnet_feature_dim=config.img_feature_dim,
                                  resnet_type=config.img_encoder_type, image_method=config.method,
                                  system=config.system_env, weight_path=config.load_weight_path)

        model = model.to(config.device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数数量: {param_count:,}")

        print("\n步骤5: 模型训练...")
        train_losses, valid_losses = train_model(model, train_loader, valid_loader, num_class, config.device, patience=config.patience, factor=config.factor,
                    max_norm=config.max_norm, alpha=config.alpha, gamma=config.gamma, weight_decay=config.weight_decay,
                    num_epochs=config.epochs, learning_rate=config.learning_rate,model_type=config.model_type,
                    save_model_name=config.save_model_path)

        print("\n步骤6: 模型预测及评测...")
        # metrics, train_preds, train_labels, valid_preds, valid_labels = evaluate_seq2seq_model(
        #     model, config.device, train_loader=train_loader, test_loader=valid_loader,model_type=config.model_type
        # )
        # plot_metrics_comparison(metrics, f"{config.model_type}_{config.vit_type}_{config.method}_valid_evalue")

        print("\n步骤5: 测试集预测及评测...")
        if os.path.exists(config.save_model_path):
            model.load_state_dict(torch.load(config.save_model_path, map_location=config.device))
            print(f"Model loaded from {config.save_model_path}")
        else:
            raise FileNotFoundError(f"模型参数文件未发现: {config.save_model_path}")
        start = time.time()
        metrics, _, _, test_preds, test_labels = evaluate_seq2seq_model(
            model, config.device, train_loader=None, test_loader=test_loader,model_type=config.model_type)
        end = time.time()
        print(f"模型预测所花时间: {end - start},模型预测指标: {metrics['test']}")
        delta_time = end - start
        metrics['test']['time'] = round(delta_time, 4)
        metrics['test']['random_seed'] = random_seed
        metric_list.append(metrics['test'])
        clear_files_in_folder(config.save_img_path)
    metrics_df = pd.DataFrame(metric_list)
    metrics_df.to_csv(f'../result/evaluate/{config.model_type}_{config.img_encoder_type}_{config.method}_metrics.csv', index=False)


# 计算每个指标的置信区间--用于不同窗长实验结果的绘制
def calc_mean_and_ci(values, confidence=0.95):
    """
    实验结果置信区间计算
    """
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    n = len(values)
    t_value = stats.t.ppf((1 + confidence) / 2., n - 1)  # t-distribution value for 95% CI

    # Calculate margin of error
    margin_of_error = t_value * (std_val / np.sqrt(n))

    # Calculate confidence interval
    lower_bound = mean_val - margin_of_error
    upper_bound = mean_val + margin_of_error

    return mean_val, (lower_bound, upper_bound)


def model_shap(model,train_X,train_img,test_X,test_img):
    """
    模型shape解释说明
    后续需要使用sheep解释，需要将模型（网络结构定义以及前向传播）的输入输出分别先重置为二维张量（batch_size*seq_len,num_class）的形式，才能进行相对应的sheep解释
    当前的seq2seq并不适合直接进行sheep解释，暂时先舍弃，后续通过直接对比不同模态的评价指标表明多模态的优点。
    """
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_X = train_X.to(config.device)
    train_img = train_img.to(config.device)
    train_image_features = model.image_feature_extractor(train_img)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_X = test_X.to(config.device)
    test_img = test_img.to(config.device)
    test_image_features = model.image_feature_extractor(test_img)

    # 多模态
    if "multi" in config.model_type:
        explainer = shap.DeepExplainer(model, [train_X, train_image_features])
        shap_values_image = explainer.shap_values([test_X, test_image_features])[0]
        shap_values_numeric = explainer.shap_values([test_X, test_image_features])[1]

        # 可视化图像模态的 SHAP 值
        shap.summary_plot(shap_values_image, test_image_features)

        # 可视化数值模态的 SHAP 值
        shap.summary_plot(shap_values_numeric, test_X)
    # 数值模态
    elif "numeric" in config.model_type:
        explainer = shap.DeepExplainer(model, train_X)
        # 计算 SHAP 值
        shap_values = explainer.shap_values(test_X)

        # 可视化图像的 SHAP 值
        shap.image_plot(shap_values, test_X)
    # 图像模态
    else:
        explainer = shap.DeepExplainer(model, train_image_features)
        # 计算 SHAP 值
        shap_values = explainer.shap_values(test_image_features)

        # 可视化图像的 SHAP 值
        shap.image_plot(shap_values, test_image_features)


if __name__ == '__main__':
    multi_random()