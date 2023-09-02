import sys
from pip import main
from train29 import *
from test_final import *


def experim_7():

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
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug1/densenet121_net4_aug1_epoch_18_tensor(0.6893, device=\'cuda:3\')_best.pkl'
  
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug1/densenet121_net4_aug1_epoch_39.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model(*test_settings)


def experim_7_1():

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

    model_name = 'densenet121_net4_aug1_1'

    feats_csv_file = global_var.base_data_aug_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

  
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  '')
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    # test_model(*test_settings)


def experim_8():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_aug2'

    feats_csv_file = global_var.base_data_aug_2_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_2_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug2/densenet121_net4_aug2_epoch_10.pkl'
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)


    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug2/densenet121_net4_aug2_epoch_37_tensor(1.0000, device=\'cuda:3\')_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, train_img_paths[-10:], train_img_labels[-10:], use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)



def experim_ljy():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

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
    # test_dir=global_var.base_test_data_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_aug_ljy/densenet121_net4_aug_ljy_epoch_43_tensor(1., device=\'cuda:3\')_best.pkl'
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = pkl_file
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_ljy2():

    model = models.densenet121(pretrained=True)
    model = Net11(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

    num_epochs = 50

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net11_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_trains256_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_ljy3():

    model = models.densenet121(pretrained=True)
    model = Net13(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

    num_epochs = 50

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net13_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_trains256_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)
    # print(test_img_paths)
    # print(test_img_labels)

    # pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net13_aug_ljy/densenet121_net13_aug_ljy_epoch_49_tensor(0.9798, device=\'cuda:3\')_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, train_img_paths[:10], train_img_labels[:10], use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_ljy4():

    model = models.densenet121(pretrained=True)
    model = Net14(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

    num_epochs = 50

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net14_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_trains256_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_ljy5():

    model = models.resnet50(pretrained=True)
    model = Net15(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    assert use_gpu == True

    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net15_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = global_var.base_data_trains256_prefix + '/roi'
    # test_dir =  global_var.base_test_tests256_prefix + '/roi'
    test_dir = global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net15_aug_ljy/densenet121_net15_aug_ljy_epoch_34_tensor(0.9979., device=\'cuda:0\')_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_ljy6():

    model = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32],
                                 window_size=7, drop_path_rate=0.5, num_classes=1024)
    model = Net16(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net16_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_trains256_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_ljy7():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    # print(model)
    # sys.exit(0)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_aug_ljy'

    feats_csv_file = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_trains256_prefix + '/roi'
    test_dir=global_var.base_test_tests256_prefix + '/roi'
    test_dir = global_var.base_test_data_prefix + '/roi'


    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_aug_ljy/densenet121_net17_aug_ljy_epoch_31_tensor(0.9996., device=\'cuda:0\')_best.pkl'
    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_aug_ljy/densenet121_net17_aug_ljy_epoch_22_tensor(0.9912., device=\'cuda:0\')_best.pkl'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


if __name__ == "__main__":
    print("yes")

    # experim_7()
    # experim_8()
    # experim_7_1()
    # experim_ljy()
    # experim_ljy3()
    # experim_ljy4()
    # experim_ljy5()

    # experim_ljy6()

    experim_ljy7()