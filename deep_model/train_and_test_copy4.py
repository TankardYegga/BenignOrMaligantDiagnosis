from pip import main
from train29 import *
from test_final import *


def test_experim_7():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_aug1'

    feats_csv_file = global_var.base_data_aug_prefix  + '/filtered_features_10.csv'
    print(feats_csv_file)
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # train_img_paths = ['/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_augmentation/roi/B1RMLO.jpg']
    # train_img_labels = [0]

    # train_img_paths = ['/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_augmentation/roi/B1RCC.jpg']
    # train_img_labels = [0]
    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug1/densenet121_net4_aug1_epoch_98_tensor(1.0000, device=\'cuda:3\')_best.pkl'
    # test_settings = (model, train_img_paths, train_img_labels, use_gpu, save_path, criterion, 'test')

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug1/densenet121_net4_aug1_epoch_30_tensor(1.0000, device=\'cuda:3\')_best.pkl'
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name, save_path)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, train_img_paths, train_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model2(*test_settings)


def test_experim_ljy():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_trains256_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug_ljy/densenet121_net4_aug_ljy_epoch_71.pkl'
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug_ljy/densenet121_net4_aug_ljy_epoch_29.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


if __name__ == "__main__":
    
    # test_experim_7()
    test_experim_ljy()
 


    