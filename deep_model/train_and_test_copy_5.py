import sys
from pip import main
from train29 import *
from test_final import *


def experim_train_data_part():

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

    model_name = 'densenet121_net4_train_part_1_no_weight_decay'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_train_part_1_no_weight_decay/densenet121_net4_train_part_1_no_weight_decay_epoch_54.pkl'
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_train_data_part_w_weight_decay():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.05)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


def experim_train_data_part_w_weight_decay2():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_train_part_1_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_train_part_1_2/densenet121_net4_train_part_1_2_epoch_94_tensor(0.7214, device=\'cuda:0\')_best.pkl'
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net4_train_part_1_2/densenet121_net4_train_part_1_2_epoch_94_tensor(0.7214, device=\'cuda:0\')_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)


if __name__ == "__main__":
    # experim_train_data_part_w_weight_decay()

    # experim_train_data_part_w_weight_decay2()

    experim_train_data_part()

    


    