from pip import main
from train29 import *
from test_final import *


def experim_1():

    model = models.densenet121(pretrained=True)
    model = Net11(model, num_extra_feats=10)

    num_epochs = 150

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.3, 0.7])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    feats_csv_file = global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net11_0'

    train_dir=global_var.base_data_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, model_name, img_feats_dict)
    save_path = train_model(*train_settings)
    
    test_settings = (model, test_img_paths, test_img_labels, use_gpu,  save_path, criterion, 'test')
    test_model(*test_settings)


def experim_2():

    model = models.densenet121(pretrained=True)
    model = Net11(model, num_extra_feats=10)

    num_epochs = 150

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.65, 0.35])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    feats_csv_file = global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net11_7'

    train_dir = global_var.base_data_prefix + '/roi'
    test_dir = global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, model_name, img_feats_dict)
    # save_path = train_model(*train_settings)
    
    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net11_7/densenet121_net11_7_epoch_119_tensor(1.0000, device=\'cuda:4\')_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu,  save_path, criterion, 'test')
    test_model(*test_settings)


def experim_3():

    model = models.densenet121(pretrained=True)
    model = Net11(model, num_extra_feats=10)

    num_epochs = 150

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.7, 0.3])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    feats_csv_file = global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net11_8'

    train_dir=global_var.base_data_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, model_name, img_feats_dict)
    # save_path = train_model(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net11_8/densenet121_net11_8_epoch_130_tensor(1.0000, device=\'cuda:6\')_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu,  save_path, criterion, 'test')
    test_model(*test_settings)


def experim_4():

    model = models.densenet121(pretrained=True)
    model = Net12(model, num_extra_feats=10)

    num_epochs = 200

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net12_data_aug4'

    feats_csv_file = global_var.base_data_aug_4_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_4_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  '')
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net12_data_aug4/densenet121_net12_data_aug4_epoch_153_0.7350_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu,  save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


    # model.load_state_dict(torch.load(save_path))
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  save_path)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

   
def experim_5():

    model = models.densenet121(pretrained=True)
    model = Net12(model, num_extra_feats=10)

    num_epochs = 20

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net12_data_aug'

    feats_csv_file = global_var.base_data_aug_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  '')
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_6():

    model = models.densenet121(pretrained=True)
    model = Net12(model, num_extra_feats=10)

    num_epochs = 20

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net12_data_aug2'

    feats_csv_file = global_var.base_data_aug_2_prefix  + '/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir=global_var.base_data_aug_2_prefix + '/roi'
    test_dir=global_var.base_test_data_prefix + '/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  '')
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


if __name__ == "__main__":
    pass



    