"""
添加另外的图像编码器
实验内容：窗长大小影响、数值模态最优、图像模态最优-不同转换方式的vit模块分类评价指标、多模态融合最优方法
不同的盲井上选择窗口大小
改进：
1.加入m每个windows有一个mlp对接分类（moe）
2.多尺度，padding引入
3.模态融合方法改进
4.ViT的微调
"""
from data_processing import *
from matrix_trans import *
from model import *
from model_train import *
from model_test import *
from untils import *
from result_output import model_shap
import config
import time


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalization using ImageNet stats
    ])
    if config.blind_well:
        well_data, num_class = read_and_preprocess(config.file_path)
        print("\n步骤1: 窗口划分及盲井段选取...")
        # 盲井选取
        start, end = 1513, 1680  # 1480, 1680
        mask = np.ones(well_data.shape[0], dtype=bool)
        mask[start:end] = False
        train_well, test_well = well_data[mask],well_data[~mask]
        # 不划分验证集
        # X_train, y_train = creat_windows(train_well, config.window_size, config.stride)
        # idx_train = np.arange(X_train.shape[0])
        # X_test,y_test = creat_windows(test_well, config.window_size, config.stride)
        # idx_test = np.arange(X_test.shape[0])
        # create_folder(f"{config.save_img_path}/train")
        # create_folder(f"{config.save_img_path}/blind")
        #
        # print("\n步骤2: 图像模态转换...")
        # image_trans_train = ImageTrans(X_train, config.method, f"{config.save_img_path}/train", config.max_workers,
        #                                config.wave_type,
        #                                config.scale)
        # image_trans_blind = ImageTrans(X_test, config.method, f"{config.save_img_path}/blind", config.max_workers,
        #                                config.wave_type,
        #                                config.scale)
        # if config.img_trans_status:
        #     image_trans_train.trans_fig()
        #     image_trans_blind.trans_fig()
        #
        # print("\n步骤3: 创建训练、验证、测试数据集...")
        # train_loader = creat_dataloaders(img_trans_class=image_trans_train, features=X_train, label=y_train,
        #                                  img_process=transform, idx=idx_train, save_pt_path=config.save_pt_path,
        #                                  batch_size=config.batch_size,
        #                                  dataset_name=f"train_{config.random_seed}", save=True)
        #
        # test_loader = creat_dataloaders(img_trans_class=image_trans_blind, features=X_test, label=y_test,
        #                                 img_process=transform, idx=idx_test, save_pt_path=config.save_pt_path,
        #                                 batch_size=config.batch_size, dataset_name=f"test_{config.random_seed}",
        #                                 save=True)
        # valid_loader = test_loader
        # 划分验证集
        train_X, train_y = creat_windows(train_well, config.window_size, config.stride)
        X_test,y_test = creat_windows(test_well, config.window_size, config.stride)
        idx_test = np.arange(X_test.shape[0])
        X_train, X_val, y_train, y_val, idx_train, idx_val = (
            train_dataset_split(train_X, train_y,
                                random_seed=config.random_seed,
                                test_ratio=config.valid_size,  # 盲井段验证集尽可能少
                                ))
        create_folder(f"{config.save_img_path}/train")
        create_folder(f"{config.save_img_path}/blind")

        print("\n步骤2: 图像模态转换...")
        image_trans_train = ImageTrans(train_X, config.method, f"{config.save_img_path}/train", config.max_workers, config.wave_type,
                                       config.scale)
        image_trans_blind = ImageTrans(X_test, config.method, f"{config.save_img_path}/blind", config.max_workers,
                                       config.wave_type,
                                       config.scale)
        if config.img_trans_status:
            image_trans_train.trans_fig()
            image_trans_blind.trans_fig()

        print("\n步骤3: 创建训练、验证、测试数据集...")
        train_loader = creat_dataloaders(img_trans_class=image_trans_train, features=X_train, label=y_train,
                                         img_process=transform, idx=idx_train, save_pt_path=config.save_pt_path,
                                         batch_size=config.batch_size,
                                         dataset_name=f"train_{config.random_seed}", save=True)

        valid_loader = creat_dataloaders(img_trans_class=image_trans_train, features=X_val, label=y_val,
                                         img_process=transform, idx=idx_val, save_pt_path=config.save_pt_path,
                                         batch_size=config.batch_size,
                                         dataset_name=f"valid_{config.random_seed}", save=True)

        test_loader = creat_dataloaders(img_trans_class=image_trans_blind, features=X_test, label=y_test,
                                        img_process=transform, idx=idx_test, save_pt_path=config.save_pt_path,
                                        batch_size=config.batch_size, dataset_name=f"test_{config.random_seed}",
                                        save=True)

    else:
        if config.well_num_select != 1:
            X, y, all_well_ids,num_class = process_files(config.file_path, config.well_num_select, config.window_size, config.stride,
                                                                   num_threads=config.well_num_select)
            X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = train_dataset_split(X,
                                                                                                               y,
                                                                                                               all_well_ids)
        else:
            well_data, num_class = read_and_preprocess(config.file_path)
            # 自动划分train、val、test
            X, y = creat_windows(well_data, config.window_size, config.stride)
            X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = (
                train_dataset_split(X, y,
                                    random_seed=config.random_seed,
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
                                          dataset_name=f"train_{config.random_seed}", save=True)

        valid_loader = creat_dataloaders(img_trans_class=image_trans, features=X_val, label=y_val,
                                          img_process=transform, idx=idx_val, save_pt_path=config.save_pt_path,
                                          batch_size=config.batch_size,
                                          dataset_name=f"valid_{config.random_seed}", save=True)

        test_loader = creat_dataloaders(img_trans_class=image_trans, features=X_test, label=y_test,
                                         img_process=transform, idx=idx_test, save_pt_path=config.save_pt_path,
                                         batch_size=config.batch_size, dataset_name=f"test_{config.random_seed}", save=True)
    input_size = X_train.shape[2]
    print(f"\n步骤4: {config.model_type}模型创建...")
    if config.model_type =='multi_concat':
        # 多模态模型
        model = MultiModeSeq2SeqModel1(input_size, num_class,
                                       config.window_size,vit_type=config.img_encoder_type,
                                       image_method=config.method,dropout=config.dropout,
                                       system=config.system_env,weight_path=config.load_weight_path,
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
                                       system=config.system_env,weight_path=config.load_weight_path,
                                       d_model=config.d_model,
                                       n_head=config.n_head,
                                       n_layer=config.num_layers,
                                       dim_feedforward=config.dim_feedforward
                                        )
    elif config.model_type == "multi_weight":
        model = MultiModeSeq2SeqModel3(input_size, num_class,
                                       config.window_size,weight_num=config.weight_num,weight_img=config.weight_img,
                                       vit_type=config.img_encoder_type,
                                       image_method=config.method, dropout=config.dropout,
                                       system=config.system_env,weight_path=config.load_weight_path,
                                       d_model=config.d_model,
                                       n_head=config.n_head,
                                       num_layers=config.num_layers,
                                       dim_feedforward=config.dim_feedforward,
                                       )
    elif config.model_type == "multi_cross_att":
        model = MultiModeSeq2SeqModel4(input_size, num_class,
                                       config.window_size,fuse_heads=config.fused_head,
                                       vit_type=config.img_encoder_type,
                                       image_method=config.method, dropout=config.dropout,
                                       system=config.system_env,weight_path=config.load_weight_path,
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
                           vit_type=config.img_encoder_type,image_method=config.method,
                           system=config.system_env,weight_path=config.load_weight_path)
    else:
        model = ResNetSeq2Seq(input_size, num_class, config.window_size,
                              resnet_feature_dim=config.img_feature_dim,
                              resnet_type=config.img_encoder_type,image_method=config.method,
                              system=config.system_env,weight_path=config.load_weight_path)

    model = model.to(config.device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {param_count:,}")

    if config.model_status in ["train","all"]:
        print("\n步骤5: 模型训练...")
        train_losses, valid_losses, best_model_state = train_model(model, train_loader, valid_loader, num_class, config.device, patience=config.patience, factor=config.factor,
                    max_norm=config.max_norm, alpha=config.alpha, gamma=config.gamma, weight_decay=config.weight_decay,
                    num_epochs=config.epochs, learning_rate=config.learning_rate,model_type=config.model_type,
                    save_model_name=config.save_model_path)
        # 训练和验证损失绘制
        plot_loss_curves(train_losses, valid_losses, f"{config.model_type}_{config.img_encoder_type}_{config.method}_valid_loss")

        # # 训练模型验证集上的shap解释--存在问题，对于多模态需要设定为特征进行阐述
        # train_img = image_trans.load_image_pt(save_pt_dir=config.save_pt_path, dataset_name=f"train_{config.random_seed}")
        # val_img = image_trans.load_image_pt(save_pt_dir=config.save_pt_path,
        #                                       dataset_name=f"valid_{config.random_seed}")
        # model_shap(model,X_train,train_img,X_val,val_img)

        print("\n步骤6: 模型预测及评测...")
        model.load_state_dict(best_model_state)
        metrics, train_preds, train_labels, valid_preds, valid_labels = evaluate_seq2seq_model(
            model, config.device, train_loader=train_loader, test_loader=valid_loader,model_type=config.model_type
        )
        print(f"模型验证集指标: {metrics['test']}")
        plot_metrics_comparison(metrics, f"{config.model_type}_{config.img_encoder_type}_{config.method}_valid_evalue")
        # return metrics, valid_preds, valid_labels

    if config.model_status in ["test","all"]:
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
        print(f"模型预测所花时间: {end - start:.4f}秒,模型预测指标: {metrics['test']}")
        df_pre = pd.DataFrame({"label":test_labels,"pre":test_preds})
        df_pre.to_csv(f"../result/evaluate/{config.model_type}_{config.img_encoder_type}_{config.method}_result.csv",index=False)
        # return metrics['test'], test_preds, test_labels
    if not config.blind_well and config.well_num_select == 1 and config.model_status == "all":
        restored_pre = np.full_like(y, np.nan)
        # 预测标签还原
        restored_pre[idx_train] = train_preds.reshape(-1, config.window_size)
        restored_pre[idx_val] = valid_preds.reshape(-1, config.window_size)
        restored_pre[idx_test] = test_preds.reshape(-1, config.window_size)
        # 预测数据集划分情况
        restored_split = np.full_like(y, np.nan)
        restored_split[idx_train] = np.full(config.window_size, 0)
        restored_split[idx_val] = np.full(config.window_size, 1)
        restored_split[idx_test] = np.full(config.window_size, 2)
        df_split = pd.DataFrame({
            'original': y.flatten(),
            'predicted_labels': restored_pre.flatten(),
            'dataset_split': restored_split.flatten()
        })

        # 保存为CSV文件
        df_split.to_csv('../result/evaluate/all_pre.csv', index=False)
    clear_files_in_folder(config.save_img_path)


if __name__ == '__main__':
    main()