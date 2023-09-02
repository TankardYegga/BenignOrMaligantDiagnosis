import sys
from numpy import save
from pip import main
from ResNet_ImageNet import ACmix_ResNet_Small, ACmix_ResNet_Small_Small
from train29 import *
from test_final import *

def experim_train_data_part_w_weight_decay_0():

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_train_part_1_3'

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

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay():

    model = models.densenet121(pretrained=True)
    model = Net15(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net15_train_part_1'

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
    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay2():

    model = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32],
                                window_size=7, drop_path_rate=0.5, num_classes=1024)
    model = Net16(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net16_train_part_1'

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
    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay3():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_1/densenet121_net17_train_part_1_epoch_30_tensor(0.9589, device=\'cuda:1\').pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, train_img_paths[32:45], test_img_labels[32:45], use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay3_2():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_2/densenet121_net17_train_part_2_epoch_15_0.8909090757369995.pkl'
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_2_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/densenet121_net17_2_train_part_2_epoch_40_0.8987012987012987.pkl'
    # save_path = os.path.dirname(save_dir) + '/densenet121_net17_2_train_part_2_epoch_42_0.9246753246753247.pkl'
    # save_path = os.path.dirname(save_dir) + '/densenet121_net17_2_train_part_2_epoch_43_0.9194805194805195.pkl'
    # save_path = os.path.dirname(save_dir) + '/densenet121_net17_2_train_part_2_epoch_32_0.9064935064935065.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay_wrapper():

    model = models.resnet50(pretrained=True)
    model = NetWrapper(model, 2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'resnet_wrapper_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    
    # test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet_wrapper_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/resnet_wrapper_train_part_2_epoch_18_0.9636363636363636.pkl'
    save_path = os.path.dirname(save_dir) + '/resnet_wrapper_train_part_2_epoch_43_1.0.pkl'
    save_path = os.path.dirname(save_dir) + '/resnet_wrapper_train_part_2_epoch_10_0.935064935064935.pkl'
    save_path = os.path.dirname(save_dir) + '/resnet_wrapper_train_part_2_epoch_12_0.9532467532467532.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay_wrapper2():

    model = models.densenet121(pretrained=True)
    model = NetWrapper(model, 2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densnet_wrapper_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    
    # test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densnet_wrapper_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/densnet_wrapper_train_part_2_epoch_59_1.0.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay_net17simple():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)
    
    model = Net17Simple(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17simple_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    
    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net17simple_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_train_part_2_epoch_12_0.9662337662337662_best.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_train_part_2_epoch_17_0.8909090909090909.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_train_part_2_epoch_18_0.922077922077922.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_train_part_2_epoch_20_0.9376623376623376.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_train_part_2_epoch_16_0.9454545454545454.pkl'
    save_path = os.path.dirname(save_path) + '/net17simple_train_part_2_epoch_14_0.9428571428571428.pkl' # good
    save_path = os.path.dirname(save_path) + '/net17simple_train_part_2_epoch_22_0.9272727272727272.pkl'


    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay_net17simple_pretrained():

    res_model = models.resnet50(pretrained=True)
    res_model = NetWrapper(res_model)
    res_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet_wrapper_train_part_2/resnet_wrapper_train_part_2_epoch_50_1.0.pkl'
    res_model.load_state_dict(torch.load(res_model_ckpt_path))
    res_model = res_model.cuda(device)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)
    
    model = Net17SimplePretrained(res_model, dense_model, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17simple_pretrained_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    
    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net17simple_pretrained_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_pretrained_train_part_2_epoch_2_1.0.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple_pretrained_train_part_2_epoch_1_0.9922077922077922.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_net17simple_pretrained_w_deep_latent():

    res_model = models.resnet50(pretrained=True)
    res_model = NetWrapper(res_model)
    res_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet_wrapper_train_part_2/resnet_wrapper_train_part_2_epoch_50_1.0.pkl'
    res_model.load_state_dict(torch.load(res_model_ckpt_path))
    res_model = res_model.cuda(device)


    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)
    

    model = Net17SimplePretrained_W_DeepLatent(res_model, dense_model, num_mid_feats=1000, num_classes=2, latent_from_dim=10, latent_to_dim=100)
    # model = Net17SimplePretrained_W_DeepLatent(res_model, dense_model, num_mid_feats=1000, num_classes=2, latent_from_dim=10, latent_to_dim=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'Net17SimplePretrained_W_DeepLatent_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)


    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/Net17SimplePretrained_W_DeepLatent_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent_train_part_2_epoch_4_0.9792207792207792.pkl'
    save_path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent_train_part_2_epoch_3_0.987012987012987.pkl'
    save_path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent_train_part_2_epoch_6_0.9974025974025974.pkl'
    save_Path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent_train_part_2_epoch_7_0.9844155844155844.pkl'


    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]


    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)


    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)

    

def experim_net17simple_pretrained_w_deep_latent2():

    res_model = models.resnet50(pretrained=True)
    res_model = NetWrapper(res_model)
    res_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet_wrapper_train_part_2/resnet_wrapper_train_part_2_epoch_50_1.0.pkl'
    res_model.load_state_dict(torch.load(res_model_ckpt_path))
    res_model = res_model.cuda(device)


    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)
    

    model = Net17SimplePretrained_W_DeepLatent(res_model, dense_model, num_mid_feats=1000, num_classes=2, latent_from_dim=10, latent_to_dim=50)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'Net17SimplePretrained_W_DeepLatent2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)


    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/Net17SimplePretrained_W_DeepLatent2_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent2_train_part_2_epoch_4_1.0.pkl'
    save_path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent2_train_part_2_epoch_1_0.9688311688311688.pkl'
    save_path = os.path.dirname(save_dir) + '/Net17SimplePretrained_W_DeepLatent2_train_part_2_epoch_2_0.987012987012987.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)
    

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay_net17simple2():

    res_model = models.resnet50(pretrained=True)
    res_model = Net32(res_model)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net17Simple2(res_model, dense_model, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17simple2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    
    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net17simple2_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple2_train_part_2_epoch_12_0.9636363636363636.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple2_train_part_2_epoch_9_0.948051948051948.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple2_train_part_2_epoch_11_0.9532467532467532.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple2_train_part_2_epoch_8_0.9194805194805195.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay_net17simple3():

    res_model = models.resnet50(pretrained=True)
    res_model = Net32(res_model)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net17Simple3(res_model, dense_model, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17simple3_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    
    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net17simple3_train_part_2/xx.pkl'
    save_path = os.path.dirname(save_dir) + '/net17simple3_train_part_2_epoch_17_0.9558441558441558.pkl'

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay4():

    model = models.resnet50(pretrained=True)
    model = Net18(model, num_extra_feats=10)

    num_epochs = 60

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net18_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    epoch_pos = -2
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name, pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay5():

    model1 = models.resnet50(pretrained=True)
    model2 = models.resnet50(pretrained=True)

    model = Net19(model1, model2,num_extra_feats=10)

    num_epochs = 60

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net19_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    epoch_pos = -2
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline_2(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay6():

    model = models.densenet121(pretrained=True)

    model = Net4(model, num_extra_feats=10, num_heads=1)

    num_epochs = 60

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net4_2_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    epoch_pos = -2
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay7():

    model = models.resnet50(pretrained=True)
    model = Net20(model,num_extra_feats=10)

    num_epochs = 60

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net20_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    epoch_pos = -2
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline_2(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online_2(*test_settings)
    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay8():

    model = models.resnet50(pretrained=True)
    model = Net20(model,num_extra_feats=10)

    num_epochs = 60

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net20_2_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    pkl_file = ''
    epoch_pos = -2
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline_2(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online_2(*test_settings)
    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay9():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 70

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net21_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_1/densenet121_net17_train_part_1_epoch_30_tensor(0.9589, device=\'cuda:1\').pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, train_img_paths[32:45], test_img_labels[32:45], use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay10():

    model = models.resnet50(pretrained=True)
    model = Net22(model, num_extra_feats=10)

    num_epochs = 70

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net22_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay11():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_train_part_1_changed_roi_165'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/changed_roi165'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_1_changed_roi_165/densenet121_net17_train_part_1_changed_roi_165_epoch_15_0.9723926186561584_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online_3(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay12():

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_train_part_1_stacked_data'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline_3(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_1_stacked_data/densenet121_net17_train_part_1_stacked_data_epoch_20_0.9794721603393555_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online_4(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay13():

    model = models.vgg19(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 110

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_vgg_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline0(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay14():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_train_part_1_added_data_3'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline_4(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_1_added_data/densenet121_net17_train_part_1_added_data_epoch_15_0.9354838728904724_best.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online_5(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay15():

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d( 3 * 64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net23(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net23_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay16():

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d( 3 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net23(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.45, 0.55])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net23_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)




def experim_train_data_part_w_weight_decay17():

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d( 3 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net24(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net24_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay18():

    model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d( 3 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net25(model, num_extra_feats=10)
    # model.conv1 = nn.Conv2d( 5 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net25(model, num_extra_feats=10)
    # model.conv1 = nn.Conv2d( 3 * 8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net25(model, num_extra_feats=10)
    model.conv1 = nn.Conv2d( 5 * 32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net25(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net25_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)


    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net25_train_part_1/densenet121_net25_train_part_1_epoch_10_0.8651026487350464.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay19():

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d( 4 * 64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net26(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net26_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net26_train_part_1/densenet121_net26_train_part_1_epoch_42_0.9970674514770508.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay20():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_train_whole'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 3, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_whole/'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay21():

    model = models.resnet50(pretrained=True)
    model = Net17(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net17_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    # test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_train_part_2/densenet121_net17_train_part_2_epoch_13_0.8961038589477539.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_net17_train_part_2_epoch_14_0.8519480228424072.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


# 训练topo特征用于分类
def experim_train_data_part_w_weight_decay22():

    topo_model = models.resnet50(pretrained=True)
    topo_model.fc = nn.Sequential(*[topo_model.fc,nn.Linear(1000,256,True),nn.Linear(256,2,True)])

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        topo_model = topo_model.cuda(device)

    optimizer = torch.optim.SGD(topo_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'topo_model_train_part_1_'

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (topo_model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, model_name,  pkl_file, epoch_pos)
    # save_path = train_topo_model(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/topo_model_train_part_1_/topo_model_train_part_1__epoch_28_1.0_best.pkl'
    test_settings = (topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_topo_model(*test_settings)  

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay22_plus():

    topo_model = models.resnet50(pretrained=True)
    topo_model.fc = nn.Sequential(*[topo_model.fc,nn.Linear(1000,256,True),nn.Linear(256,2,True)])

    model = models.resnet50(pretrained=True)
    model = Net17Plus(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)
        topo_model = topo_model.cuda(device)

    ckpt_pth = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/topo_model_train_part_1_/topo_model_train_part_1__epoch_22_0.997402548789978_best.pkl'
    topo_model.load_state_dict(torch.load(ckpt_pth))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17_plus_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, topo_model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_with_topo_model(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net17_plus_train_part_1/net17_plus_train_part_1_epoch_0_0.9648093581199646_best.pkl'
    test_settings = (model,topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_with_topo_model(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay23():

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d( 3 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net27(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net27_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos, True)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



# 训练topo特征用于分类
def experim_train_data_part_w_weight_decay24():

    topo_model = models.resnet50(pretrained=True)
    topo_model.fc = nn.Sequential(*[topo_model.fc,nn.Linear(1000,2,True)])

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        topo_model = topo_model.cuda(device)

    optimizer = torch.optim.SGD(topo_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'topo_model_2_train_part_1_'

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (topo_model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, model_name,  pkl_file, epoch_pos)
    save_path = train_topo_model(*train_settings)

    test_settings = (topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_topo_model(*test_settings)  
    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay24_plus():

    topo_model = models.resnet50(pretrained=True)
    topo_model.fc = nn.Sequential(*[topo_model.fc,nn.Linear(1000,2,True)])

    model = models.resnet50(pretrained=True)
    model = Net17PlusPlus(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)
        topo_model = topo_model.cuda(device)

    ckpt_pth = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/topo_model_2_train_part_1_/topo_model_2_train_part_1__epoch_17_0.9688311219215393.pkl'
    topo_model.load_state_dict(torch.load(ckpt_pth))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17_plus_plus_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, topo_model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_with_topo_model2(*train_settings)

    test_settings = (model,topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_with_topo_model(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay24_plus2():

    topo_model = models.resnet50(pretrained=True)
    topo_model.fc = nn.Sequential(*[topo_model.fc,nn.Linear(1000,2,True)])

    model = models.resnet50(pretrained=True)
    model = Net17PlusPlusPlus(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)
        topo_model = topo_model.cuda(device)

    ckpt_pth = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/topo_model_2_train_part_1_/topo_model_2_train_part_1__epoch_8_0.9677419066429138_best.pkl'
    topo_model.load_state_dict(torch.load(ckpt_pth))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net17_plus_plus_plus_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, topo_model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_with_topo_model2(*train_settings)

    test_settings = (model,topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_with_topo_model(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay25():

    print('28 starting...')

    model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model = Net28(model, num_extra_feats=10)
    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net28_train_part_1'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos, True)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay26():

    # renset50
    model = ACmix_ResNet(layers=[3,4,6,3], num_classes=1000)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net29_resnet50_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay27():

    # renset38
    model = ACmix_ResNet(layers=[3,3,3,3], num_classes=1000)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net29_resnet38_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay28():

    # renset18
    model = ACmix_ResNet2(layers=[2,2,2,2], num_classes=1000)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net29_resnet18_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay29():

    # renset14
    model = ACmix_ResNet(layers=[1,1,1,1], num_classes=1000)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net29_resnet14_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay30():

    model = models.resnet50(pretrained=True)
    model = Net30(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net30_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay31():

    model = models.resnet50(pretrained=True)
    model = Net31(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net31_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay32():

    # renset38
    model = ACmix_ResNet(layers=[3,3,3,3], num_classes=1000, k_att=7, head=1, k_conv=3)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net32_resnet38_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay33():

    # renset38
    model = ACmix_ResNet(layers=[3,3,3,3], num_classes=2, k_att=5, head=1, k_conv=3)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net33_resnet38_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay34():

    # renset38
    model = ACmix_ResNet(layers=[3,3,3,3], num_classes=2, k_att=3, head=1, k_conv=3)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net34_resnet38_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay35():

    # renset38
    extra_model = models.resnet50(pretrained=True)
    extra_model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
    model = ACmix_ResNet_Small_Small(extra_model, layers=[3], num_classes=2, k_att=7, head=1, k_conv=3)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net35_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay36():

    # renset38
    extra_model = models.resnet50(pretrained=True)
    extra_model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
    model = ACmix_ResNet_Small(extra_model, layers=[3, 3], num_classes=2, k_att=7, head=1, k_conv=3)
    model = Net29(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net36_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay37():

    model = models.densenet121(pretrained=True)
    model = Net32(model, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net37_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay37_2():

    model = models.resnet50(pretrained=True)
    model = Net32(model, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net37_2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    if 'best' in save_path:
        saved_epoch = save_path.split('_')[-3]
    else:
        saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test', saved_epoch)
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay38():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net33(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_net38_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)


    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net38_train_part_2/densenet121_net38_train_part_2_epoch_2_0.9376623034477234.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_net38_train_part_2_epoch_0_0.8155844211578369.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_net38_train_part_2_epoch_4_0.9298701286315918.pkl'
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay39():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_39_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    # test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'
    # test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/tests_256/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39-saved_train_part_2/XXX.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_3_0.9350649118423462.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_5_0.9558441638946533.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_4_0.932467520236969.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_23_0.997402548789978.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_16_0.9896103739738464.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_1_0.9298701286315918.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_21_1.0.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_2_0.9506493210792542.pkl'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay39_3():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_39_3_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39-saved_train_part_2/densenet121_39_train_part_2_epoch_1_0.9298701286315918.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_39_train_part_2_epoch_3_0.9350649118423462.pkl'

    pkl_file = save_path

    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline2(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_3_train_part_2/XXX.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_3_train_part_2_epoch_13_0.916883111000061.pkl'
    
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_train_data_part_w_weight_decay39_2():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_39_2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''

    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline3(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_2_train_part_2/XXX.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_39_2_train_part_2_epoch_10_0.9168831168831169.pkl'
    
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


# 论文最好结果
def experim_train_data_part_w_weight_decay39_final():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_39_final_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_final_train_part_2/XXX.pkl'
    save_path = os.path.dirname(save_path) + '/densenet121_39_final_train_part_2_epoch_4_0.9428571428571428.pkl'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)




def experim_train_data_part_w_weight_decay39_final_final():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_39_final_final2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39-saved_train_part_2/densenet121_39_train_part_2_epoch_2_0.9506493210792542.pkl'
    # pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_final_final_train_part_2/densenet121_39_final_final_train_part_2_epoch_3_0.987012987012987.pkl'

    # train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_final_final_train_part_2/XXX.pkl'
    # # save_path = os.path.dirname(save_path) + '/densenet121_39_final_final_train_part_2_epoch_3_0.987012987012987_best.pkl'
    # # save_path = os.path.dirname(save_path) + '/densenet121_39_final_final_train_part_2_epoch_5_0.9402597402597402.pkl'
    # # save_path = os.path.dirname(save_path) + '/densenet121_39_final_final_train_part_2_epoch_7_0.961038961038961.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_39_final_final_train_part_2_epoch_14_0.9922077922077922.pkl'

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_final_final2_train_part_2/densenet121_39_final_final2_train_part_2_epoch_3_0.987012987012987.pkl'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_net35_w_deep_latent():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    # model = Net35_W_DeepLatent(model, dense_model, num_extra_feats=10, latent_from_dim=10, latent_to_dim=100)

    # model = Net35_W_DeepLatent(model, dense_model, num_extra_feats=10, latent_from_dim=10, latent_to_dim=50)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # model_name = 'net35_w_deep_latent_train_part_2'
    # model_name = 'net35_w_deep_latent_2_train_part_2'


    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''

    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net35_w_deep_latent_train_part_2/XXX.pkl'
    # save_path = os.path.dirname(save_path) + '/'
  
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay39_pretrained():

    res_model = models.resnet50(pretrained=True)
    res_model = NetWrapper(res_model)
    res_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet_wrapper_train_part_2/resnet_wrapper_train_part_2_epoch_50_1.0.pkl'
    res_model.load_state_dict(torch.load(res_model_ckpt_path))
    res_model = res_model.cuda(device)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35Pretrained(res_model, dense_model, num_extra_feats=10, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net35pretrained2_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net35pretrained_train_part_2/XXX.pkl'
    save_path = os.path.dirname(save_path) + '/net35pretrained_train_part_2_epoch_15_1.0.pkl'

    epoch_pos = -2
    pkl_file = save_path

    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    # save_path = train_model_w_topo_mask_offline(*train_settings)
    save_path = train_model_w_topo_mask_offline3(*train_settings)

    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net35pretrained_train_part_2/XXX.pkl'
    save_path = os.path.dirname(save_path) + '/net35pretrained_train_part_2_epoch_15_1.0.pkl'
    # save_path = os.path.dirname(save_path) + '/net35pretrained_train_part_2_epoch_14_0.9974025974025974.pkl'

    # if 'best' in save_path:
    #     saved_epoch = save_path.split('_')[-3]
    # else:
    #     saved_epoch = save_path.split('_')[-2]

    # test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, train_img_paths[200:220], train_img_labels[200:220], use_gpu, save_path, criterion, 'test')
    # test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


   
def experim_train_data_part_w_weight_decay40():

    model = models.resnet50(pretrained=True)
    model = Net36(model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_40_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


def experim_train_data_part_w_weight_decay41():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_41_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_41_train_part_2'
    # save_path = os.path.join(save_dir, 'densenet121_41_train_part_2_epoch_27_0.9688311219215393.pkl')
    # save_path = os.path.join(save_dir, 'densenet121_41_train_part_2_epoch_24_0.9948051571846008_best.pkl')
    # saved_epoch = 0
    # if 'best' in save_path:
    #     saved_epoch = save_path.split('_')[-3]
    # else:
    #     saved_epoch = save_path.split('_')[-2]

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)




def experim_train_data_part_w_weight_decay42():

    model = models.resnet50(pretrained=True)

    dense_model = models.densenet121(pretrained=True)
    dense_model = Net32(dense_model)
    dense_model_ckpt_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net37_train_part_2/densenet121_net37_train_part_2_epoch_15_0.9870129823684692_best.pkl'
    dense_model.load_state_dict(torch.load(dense_model_ckpt_path))
    dense_model = dense_model.cuda(device)

    model = Net35_3(model, dense_model, num_extra_feats=10)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'densenet121_42_train_part_2'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_3999_train_part_2/densenet121_39_train_part_2_epoch_1_0.9298701286315918.pkl'
    # save_path = os.path.dirname(save_path) + '/densenet121_3999_train_part_2_epoch_22_0.9766233563423157.pkl'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



if __name__ == "__main__":

    # experim_train_data_part_w_weight_decay()

    # experim_train_data_part_w_weight_decay2()

    # experim_train_data_part_w_weight_decay3()

    # experim_train_data_part_w_weight_decay4()

    # experim_train_data_part_w_weight_decay5()

    # experim_train_data_part_w_weight_decay6()

    # experim_train_data_part_w_weight_decay7()

    # experim_train_data_part_w_weight_decay8()

    # experim_train_data_part_w_weight_decay9()

    # experim_train_data_part_w_weight_decay10()

    # experim_train_data_part_w_weight_decay11()

    # experim_train_data_part_w_weight_decay12()

    # experim_train_data_part_w_weight_decay13()


    # experim_train_data_part_w_weight_decay14()


    # experim_train_data_part_w_weight_decay15()

    # experim_train_data_part_w_weight_decay16()

    # experim_train_data_part_w_weight_decay17()

    # experim_train_data_part_w_weight_decay18()

    # experim_train_data_part_w_weight_decay20()

    # experim_train_data_part_w_weight_decay21()

    # experim_train_data_part_w_weight_decay23()

    # experim_train_data_part_w_weight_decay22()

    # experim_train_data_part_w_weight_decay22_plus()


    # experim_train_data_part_w_weight_decay24()


    # experim_train_data_part_w_weight_decay24_plus()

    # experim_train_data_part_w_weight_decay24_plus2()

    # experim_train_data_part_w_weight_decay25()


    # experim_train_data_part_w_weight_decay21()


    # experim_train_data_part_w_weight_decay26()
    
    # experim_train_data_part_w_weight_decay27()

    # experim_train_data_part_w_weight_decay28()

    # experim_train_data_part_w_weight_decay29()

    # experim_train_data_part_w_weight_decay31()

    # experim_train_data_part_w_weight_decay30()

    # experim_train_data_part_w_weight_decay32()

    # experim_train_data_part_w_weight_decay33()

    # experim_train_data_part_w_weight_decay34()

    # experim_train_data_part_w_weight_decay35()

    # experim_train_data_part_w_weight_decay37()

    # experim_train_data_part_w_weight_decay38()

    # experim_train_data_part_w_weight_decay21()

    # experim_train_data_part_w_weight_decay39()

    # experim_train_data_part_w_weight_decay40()

    # experim_train_data_part_w_weight_decay41()

    # experim_train_data_part_w_weight_decay39_2()

    # experim_train_data_part_w_weight_decay42()


    # experim_train_data_part_w_weight_decay39_final()

    # experim_train_data_part_w_weight_decay39_final_final()

    # experim_train_data_part_w_weight_decay3_2()

    # experim_train_data_part_w_weight_decay_wrapper()

    # experim_train_data_part_w_weight_decay_wrapper2()

    # experim_train_data_part_w_weight_decay_net17simple()

    # experim_train_data_part_w_weight_decay_net17simple2()

    # experim_train_data_part_w_weight_decay_net17simple3()

    # experim_train_data_part_w_weight_decay_net17simple_pretrained()

    # experim_train_data_part_w_weight_decay37_2()

    # experim_train_data_part_w_weight_decay39_pretrained()




    # experim_net35_w_deep_latent()

    # experim_net17simple_pretrained_w_deep_latent()

    experim_net17simple_pretrained_w_deep_latent2()











    
    












    


    