import sys
from turtle import mode
from numpy import save
from pip import main
from train_corrected import test_model_concat_fusion, test_model_corrected, test_model_w_loss_corrected, train_model_concat_fusion, train_model_corrected, train_model_inception, train_model_w_loss_corrected
from ResNet_ImageNet import ACmix_ResNet_Small, ACmix_ResNet_Small_Small
from train29 import *
from test_final import *
import random
# from densenet_plus import Net32Plus
from torchstat import stat



def setup_seed(seed):
	#  下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    
    # torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。


# 论文最好结果
def experim_1():

    backbone = NetMerge(has_second_fc=False)
    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net_merge/net_merge_epoch_43_0.9922077922077922.pkl'
    backbone.load_state_dict(torch.load(pkl_file), strict=False)

    model = NetMergeWExtra(backbone=backbone, num_extra_feats=10, num_mid_feats=1000, num_classes=2)

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net_merge_w_extra_3'

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

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/final_model/XXX.pkl'
    # save_path = os.path.dirname(save_path) + '/'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



# 论文最好结果
def experim_2():

    model = NetMerge()

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.5, 0.5])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model_name = 'net_merge'

    feats_csv_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file, 2, suffix='jpg')

    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net_merge/net_merge_epoch_26_0.9558441558441558.pkl'
   
    train_settings = (model, train_img_paths, train_img_labels, num_epochs, criterion, optimizer, scheduler, use_gpu, img_feats_dict, model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_topo_mask_offline(*train_settings)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net_merge/XXX.pkl'
    # save_path = os.path.dirname(save_path) + '/net_merge_epoch_25_0.9532467532467532.pkl'
    # save_path = os.path.dirname(save_path) + '/net_merge_epoch_26_0.9558441558441558.pkl'

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_3():

    # model = models.resnet50(pretrained=True)
    # model = Net17(model, num_extra_feats=10)

    # model = Net32(model)

    # model = Net32Plus()

    # model = ResNet1()


    model = ResNet2()

    num_epochs = 100

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # model_name = 'experim_3_resnet1'
    model_name = 'experim_3_resnet2'


    train_feats_csv = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/filtered_features_10.csv'
    train_img_feats_dict = get_img_feats_dict(train_feats_csv, 2, suffix='jpg')
    val_feats_csv = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/val/filtered_features_10.csv'
    val_img_feats_dict = get_img_feats_dict(val_feats_csv, 2, suffix='jpg')


    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/roi'
    val_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/val/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/test/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    val_img_paths, val_img_labels = get_available_data_by_order(data_dir=val_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)


    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, val_img_paths, val_img_labels,
     num_epochs, criterion, optimizer, scheduler, use_gpu, 
     train_img_feats_dict, val_img_feats_dict, 
     model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_validation_part(*train_settings)

    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_4():

    seed = 10
    setup_seed(seed)


    # model = ResNet1()
    # model = ResNet2()
    # model = Net32Plus()

    # model = ResNet3()
    # model = DenseNet1()

    # model = ResNet4()

    # model = ResNet7()

    # model = ResNet8()

    # model = Net32Plus_W_DeepLatent(latent_from_dim=10, latent_to_dim=10)
    # model = Net32Plus_W_DeepLatent(latent_from_dim=10, latent_to_dim=50)
    # model = Net32Plus_W_DeepLatent(latent_from_dim=10, latent_to_dim=100)

    # model = Net32Plus_W_DeepLatent2(latent_to_dim=50)
    # model = Net32Plus_W_DeepLatent2(latent_to_dim=10)

    # model = ResNet9(num_extra_feats=10, num_mid_feats=10, num_classes=2)

    
    # model = models.resnet50(pretrained=True)
    # model = Net17(model, num_extra_feats=10)

    # model = ResNet5()
    # model = ResNet10()

    # model = ResNet10(num_extra_feats=10, num_mid_feats=32, num_classes=2)
    # model = ResNet10(num_extra_feats=10, num_mid_feats=16, num_classes=2)

    # model = ResNet11()
    
    # model = ResNet10(num_extra_feats=10, num_mid_feats=32, num_classes=2)

    # model = ResNet13(num_mid_feats=32, num_classes=2)


    model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))


    model = ResNet14(model=model1, num_extra_feats=10, num_mid_feats=32, num_classes=2)
    # model = ResNet15(model=model1, num_extra_feats=10, num_mid_feats=32, num_classes=2)

    num_epochs = 60
   
    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0008, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # model_name = 'experim_4_resnet1'
    # model_name = 'experim_4_resnet2'
    # model_name = 'experim_4_net32_plus_on_test2'
    # model_name = 'experim_4_resnet3'
    # model_name = 'experim_4_densenet1'
    # model_name = 'experim_4_resnet4'
    # model_name = 'experim_4_resnet7'
    # model_name = 'experim_4_resnet8'
    # model_name = 'experim_4_deep_latent_1'
    # model_name = 'experim_4_deep_latent_2'
    # model_name = 'experim_4_deep_latent_3'
    # model_name = 'experim_4_deep_latent2_1'
    # model_name = 'experim_4_resnet9'
    # model_name = 'experim_4_resnet9_1'
    # model_name = 'experim_4_net17'
    # model_name = 'experim_4_resnet51'
    # model_name = 'experim_4_resnet10'
    # model_name = 'experim_4_resnet10_2'
    # model_name = 'experim_4_resnet10_3'
    # model_name = 'experim_4_resnet10_4'
    # model_name = 'experim_4_resnet10_5'

    # model_name = 'experim_4_resnet11'

    # model_name = 'experim_4_resnet13'

    # model_name = 'experim_4_resnet14'
    # model_name = 'experim_4_resnet14_2'
    # model_name = 'experim_4_resnet14_3'
    # model_name = 'experim_4_resnet14_4'
    # model_name = 'experim_4_resnet14_5'
    model_name = 'experim_4_resnet14_6'

    # model_name = 'experim_4_resnet15'


    train_feats_csv = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/filtered_features_10.csv'
    train_img_feats_dict = get_img_feats_dict(train_feats_csv, 2, suffix='jpg')
    val_feats_csv = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/filtered_features_10.csv'
    val_img_feats_dict = get_img_feats_dict(val_feats_csv, 2, suffix='jpg')


    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/roi'
    val_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/roi'
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/roi'

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    val_img_paths, val_img_labels = get_available_data_by_order(data_dir=val_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2
    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, val_img_paths, val_img_labels,
     num_epochs, criterion, optimizer, scheduler, use_gpu, 
     train_img_feats_dict, val_img_feats_dict, 
     model_name,  pkl_file, epoch_pos)
    save_path = train_model_w_validation_part(*train_settings)

  
    test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_model_w_topo_mask_online(*test_settings)

    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)



def experim_5():

    seed = 10
    setup_seed(seed)

    # model_name = 'experim_4_resnet16'
    # model = ResNet16()

    # model_name = 'experim_4_resnet16_2'
    # model = ResNet16(num_mid_features=64)

    # model_name = 'experim_4_resnet16_3'
    # model = ResNet16(num_mid_features=32) #add relu

    # model_name = 'experim_4_resnet16_4'
    # model = ResNet16(num_mid_features=512)

    # model_name = 'experim_4_resnet16_5'
    # model = ResNet16(num_mid_features=128)

    # model_name = 'experim_4_resnet16_6'
    # model = ResNet16(num_mid_features=16)

    # model_name = 'experim_4_resnet18'
    # model = ResNet18(num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_4_on_different_data'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4(model1, model2, num_mid_feats=32, num_classes=2)
  

    # model_name = 'experim_4_densenet2'
    # model = DenseNet2(num_mid_features=32, num_classes=2)


    # model_name = 'experim_4_resnet17_3_2'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_3(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_resnet17_4'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_resnet17_4_loss_changed'
    # model = ResNet17_4_Loss_Changed(num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_4_loss_changed_paper'
    # model = ResNet17_4_Loss_Changed(num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_densenet2_change_pool'
    # model = DenseNet2_W_WeightPool(num_mid_features=32, num_classes=2)

    # model_name = 'experim_4_resnet17_4_refined_2'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4_Refined(model1, model2, num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_4_refined2_0'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4_Refined2(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_densenet3'
    # model = DenseNet3(num_mid_features=32, num_classes=2)


    # model_name = 'experim_4_resnet17_4_refined3_0'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model3 = DenseNet3(num_mid_features=32, num_classes=2)
    # model3.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet3/experim_4_densenet3_epoch_21_0.9688311688311688_0.8372093023255814_best.pkl'))

    # model = ResNet17_4_Refined3(model1, model2, model3, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_resnet17_4_refined3_1'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model3 = VggNet2(num_classes=2)
    # model3.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_vgg2/experim_4_vgg2_epoch_0_0.5974025974025974_0.7441860465116279_best.pkl'))
   
    # model = ResNet17_4_Refined3(model1, model2, model3, num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_4_refined4_0'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model3 = VggNet(num_classes=2)
    # model3.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_vgg/experim_4_vgg_epoch_5_0.9688311688311688_0.7441860465116279_best.pkl'))
   
    # model = ResNet17_4_Refined4(model1, model2, model3, num_mid_feats=32, num_classes=2)

    # model = 'experim_4_inception_3'
    # model = InceptionNet(num_classes=2)

    # model_name = 'experim_4_vgg'
    # model = VggNet(num_classes=2)

    # model_name = 'experim_4_vgg2'
    # model = VggNet2(num_classes=2, num_mid_features=32)

    # model_name = 'experim_4_resnet17_w_trfm'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4_W_TRFM(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_resnet17_4_2'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    
    # model = ResNet17_4(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_resnet17_44'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4(model1, model2, num_mid_feats=32, num_classes=2)



    # model_name = 'experim_4_resnet17_4_refined_data2'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet13_data2/resnet13_data2_epoch_28_0.956140350877193_0.7674418604651163_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet2_data2/densenet2_data2_epoch_56_0.9707602339181286_0.813953488372093_best.pkl'))
    
    # model = ResNet17_4_Refined(model1, model2, num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet13_new_data'
    # model = ResNet13(num_mid_feats=32, num_classes=2)


    # model_name = 'resnet13_change_pool'
    # model = ResNet13_Change_Pool(num_mid_feats=32, num_classes=2)

    # model_name = 'resnet13_change_pool2'
    # model = ResNet13_Change_Pool2(num_mid_feats=32, num_classes=2)

    # model_name = 'resnet13_change_pool3'
    # model = ResNet13_Change_Pool3(num_mid_feats=32, num_classes=2)

    # model_name = 'resnet13_change_pool3_30'
    # model = ResNet13_Change_Pool3(num_mid_feats=32, num_classes=2)

    # model_name = 'resnet13_data2'
    # model = ResNet13(num_mid_feats=32, num_classes=2)


    # model_name = 'densenet2_data2'
    # model = DenseNet2()

    # model_name = 'resnet13_change_pool4'
    # model = ResNet13_Change_Pool4(num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_5'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_5(model1, model2, num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_6'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_6(model1, model2, num_mid_feats=32, num_classes=2)

    # model_name = 'experim_4_resnet17_4_plus'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4_Plus(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_resnet17_4_plus3'
    # model1 = ResNet13(num_mid_feats=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17_4_Plus3(model1, model2, num_mid_feats=32, num_classes=2)
 

    # model_name = 'experim_4_resnet17_4'
    # model2 = ResNet13(num_mid_feats=32, num_classes=2)
    # model2.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet13/experim_4_resnet13_epoch_38_0.9636363636363636_0.7441860465116279_best.pkl'))

    # model1 = DenseNet2(num_mid_features=32, num_classes=2)
    # model1.load_state_dict(torch.load('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet2/experim_4_densenet2_epoch_58_0.9558441558441558_0.7906976744186046_best.pkl'))
    
    # model = ResNet17(model1, model2, num_mid_feats=32, num_classes=2)


    # model_name = 'experim_4_aspp_1'
    # model = ASPP(in_channels=3, out_channels=32, num_classes=2)
    

    # model_name = 'experim_4_aspp_2'
    # model = ASPP2(in_channels=3, out_channels=32, num_classes=2)


#########################################################################################################
    # weight_for_loss = 1.0
    # times = '2'
    # model_name = 'experim_4_resnet17_4_refined_w_loss_paper_' + str(weight_for_loss) + '_' + times
    # model = ResNet17_4_Refined_W_Loss(num_mid_feats=32, num_classes=2)

    ########### 模型返回三个值的时候stat函数没有办法正常工作，所以稍微改动了下用于计算参数量
    # weight_for_loss = 1.0
    # times = '1'
    # model_name = 'experim_4_resnet17_4_refined_w_loss_paper_' + str(weight_for_loss) + '_' + times
    # model = ResNet17_4_Refined_W_Loss_Single_Output(num_mid_feats=32, num_classes=2)

    # stat(model, (1, 3, 128, 128))


    weight_for_loss = 1.0
    times = '3'
    model_name = 'experim_4_resnet17_4_refined_w_loss_new2_paper2_' + str(weight_for_loss) + '_' + times
    model = ResNet17_4_Refined_W_Loss_New2(num_mid_feats=32, num_classes=2)

    # stat(model, (1, 3, 128, 128))

    # sys.exit(0)


############################################################################################################

    # weight_for_loss = 1.0
    # times = '2'
    # model_name = 'experim_4_densenet121_paper2_' + str(weight_for_loss) + '_' + times
    # model = DenseNet2(num_mid_features=32, num_classes=2)

    # stat(model, (1, 3, 128, 128))

    # sys.exit(0)

##########################################################################################################

    # weight_for_loss = 1.0
    # times = '2'
    # model_name = 'experim_4_resnet18_paper2_' + str(weight_for_loss) + '_' + times
    # model = ResNet13(num_mid_feats=32, num_classes=2)

    # stat(model, (1, 3, 128, 128))

    # sys.exit(0)

#########################################################################

    # weight_for_loss = 1.0
    # times = '1'
    # model_name = 'experim_4_proposed_wo_multi_output_paper2_' + str(weight_for_loss) + '_' + times

    # model1 = ResNet13(num_mid_feats=32, num_classes=2)

    # model2 = DenseNet2(num_mid_features=32, num_classes=2)
    
    # model = ResNet17_4_Refined(model1, model2, num_mid_feats=32, num_classes=2)

    # stat(model, (1, 3, 128, 128))

    # sys.exit(0)

#########################################################################

    # weight_for_loss = 1.0
    # times = '1'
    # model_name = 'experim_4_fdffe_paper2_' + str(weight_for_loss) + '_' + times
    # model = FDFFE(num_mid_features=500, num_classes=2)
    # model = FDFFE(num_mid_features=100, num_classes=2)
    # stat(model, (1, 10))

    # sys.exit(0)


###########################################################################

    # weight_for_loss = 1.0
    # times = '2'
    # model_name = 'experim_4_resnet17_4_ConcatFusion_w_loss_paper2_' + str(weight_for_loss) + '_' + times
    # model = ResNet17_4_ConcatFusion_W_Loss(num_mid_feats=32, num_classes=2)
    # stat(model, (3, 128, 128))

    # sys.exit(0)

##############################################################################

    # model_name = 'experim_4_inception_paper2'
    # model = InceptionNet(num_classes=2)

    # stat(model, (3, 299, 299))

    # sys.exit(0)

################################################################################

    # model_name = 'experim_4_inception2_paper'
    # model = InceptionNet2(num_classes=2)

##################################################################################

    # model_name = 'experim_4_vggnet_paper2'
    # model = VggNet(num_classes=2)

    # print("model:", model)

    # stat(model, (3, 128, 128))

    # sys.exit(0)


# ##################################################################################

    # model_name = 'experim_4_vggnet2_paper'
    # model = VggNet2(num_classes=2)

##################################################################################

    # model_name = 'experim_4_vggnet16_paper'
    # model = VggNet16(num_classes=2)

    # stat(model, (3, 128, 128))

##################################################################################

#     model_name = 'experim_4_vggnet16_2_paper'
#     model = VggNet16_2(num_classes=2)

# ##################################################################################

    # model_name = 'experim_4_efficient_paper2'
    # model = EfficientNet(num_classes=2)

    # stat(model, (3, 128, 128))


# ##################################################################################

    # model_name = 'experim_4_efficient_2_paper'
    # model = EfficientNet2(num_classes=2)

    # stat(model, (3, 128, 128))

##################################################################################

    # model_name = 'experim_4_alexnet_paper'
    # model = AlexNet(num_classes=2)

    # stat(model, (3, 128, 128))


#################################################################################

    # model_name = 'experim_4_alexnet_2_paper'
    # model = AlexNet2(num_classes=2)

    # stat(model, (3, 224, 224))


#################################################################################

    # 学习率使用0.0005，不用池化 + SDWFM（d=32）
    # model_name = 'resnet13_change_pool3_paper2'
    # model = ResNet13_Change_Pool3(num_mid_feats=32, num_classes=2)

    # stat(model, (1, 3, 128, 128)

###################################################################################

    # 学习率使用0.0008，使用池化 + SDWFM(d=30)
    # model_name = 'resnet13_change_pool3_2_paper'
    # model = ResNet13_Change_Pool3_2(num_mid_feats=32, num_classes=2)

###############################################################################

    # num_trans_dim = 100
    # model_name = 'DenseNet2_W_WeightPool_' + str(num_trans_dim) + '_paper' 
    # model =  DenseNet2_W_WeightPool(num_mid_features=32, num_classes=2, num_trans_dim=num_trans_dim)

    # stat(model, (1, 3, 128, 128))


##############################################################################

    # num_trans_dim = 100
    # model_name = 'DenseNet2_W_WeightPool2_' + str(num_trans_dim) + '_paper' 
    # model =  DenseNet2_W_WeightPool2(num_mid_features=32, num_classes=2, num_trans_dim=num_trans_dim)

    # stat(model, (1, 3, 128, 128))

########################################################################################

    # num_trans_dim = 90
    # model_name = 'DenseNet2_W_WeightPool3_' + str(num_trans_dim) + '_paper' 
    # model =  DenseNet2_W_WeightPool_3(num_mid_features=32, num_classes=2, num_trans_dim=num_trans_dim)

    # stat(model, (1, 3, 128, 128))

#######################################################################################

    num_epochs = 35
   
    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])

    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0008, momentum=0.9, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    datasets_str = 'fixed_split_3'
    train_feats_csv = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/'+ datasets_str + '/train/filtered_features_10.csv'
    train_img_feats_dict = get_img_feats_dict(train_feats_csv, 2, suffix='jpg')
    val_feats_csv = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/' + datasets_str + '/val/filtered_features_10.csv'
    val_img_feats_dict = get_img_feats_dict(val_feats_csv, 2, suffix='jpg')

    roi_data_str = 'roi'
    # roi_data_str = 'roi_299'
    train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/' + datasets_str + '/train/' + roi_data_str
    val_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/' + datasets_str + '/val/' + roi_data_str
    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/' + datasets_str + '/test/' + roi_data_str

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    # print("len of train data:", len(train_img_labels))
    # # 去除第51、54、53、52号（从0开始计数）
    # train_img_paths = train_img_paths[:50] + train_img_paths[200:250]
    # train_img_labels = train_img_labels[:50] + train_img_labels[200:250]

    val_img_paths, val_img_labels = get_available_data_by_order(data_dir=val_dir)
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    epoch_pos = -2

    pkl_file = ''
    train_settings = (model, train_img_paths, train_img_labels, val_img_paths, val_img_labels,
     num_epochs, criterion, optimizer, scheduler, use_gpu, 
     train_img_feats_dict, val_img_feats_dict, 
     model_name,  pkl_file, epoch_pos)

    # save_path = train_model_w_validation_part(*train_settings)
    # save_path = train_model_w_validation_part22(*train_settings, w_loss=weight_for_loss)
    # save_path = train_model_w_validation_part2(*train_settings, w_loss=weight_for_loss)
    # save_path = train_model_w_validation_part3(*train_settings)
    # save_path = train_model_w_validation_part2(*train_settings, w_loss=weight_for_loss)


    # save_path = train_model_corrected(*train_settings)
    # save_path = train_model_inception(*train_settings)
    # save_path = train_model_w_loss_corrected(*train_settings, w_loss=weight_for_loss)
    # save_path = train_model_concat_fusion(*train_settings, w_loss=weight_for_loss)



    # test_settings = (model, val_img_paths, val_img_labels, use_gpu, save_path, criterion, 'test')
    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2/experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2_epoch_34_0.9444444444444444_0.8604651162790697.pkl'
    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_paper_1.0/experim_4_resnet17_4_refined_w_loss_paper_1.0_epoch_32_0.935672514619883_0.8372093023255814.pkl'
    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_new2_paper_1.0_1/experim_4_resnet17_4_refined_w_loss_new2_paper_1.0_1_epoch_34_0.9444444444444444_0.8604651162790697.pkl'
    save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2/experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2_epoch_34_0.9444444444444444_0.8604651162790697.pkl'
    # test_settings = (model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, 'test')
    test_settings = (model, val_img_paths, val_img_labels, use_gpu, save_path, criterion, 'val')
    # test_settings = (model, val_img_paths, val_img_labels, use_gpu, save_path, criterion, 'test')
    # test_settings = (model, train_img_paths, train_img_labels, use_gpu, save_path, criterion, 'test')


    # saved_epoch = 24
    # test_model_w_correct_auc(*test_settings, saved_epoch)

    # test_model_w_topo_mask_online(*test_settings)
    # test_model_w_topo_mask_online3(*test_settings)
    # test_model_w_topo_mask_online2(*test_settings, w_loss=weight_for_loss)
    # test_model_w_topo_mask_online21(*test_settings, w_loss=weight_for_loss)
    # test_model_w_topo_mask_online2(*test_settings, w_loss=weight_for_loss)

    # test_model_corrected(*test_settings)
    test_model_w_loss_corrected(*test_settings, w_loss=weight_for_loss)
    # test_model_concat_fusion(*test_settings, w_loss=weight_for_loss)



    print("&" * 30)
    print("save_path:", save_path)
    print("&" * 30)


        
if __name__ == "__main__":
    # experim_1()

    # experim_2()
    
    # experim_3()

    # experim_4()

    experim_5()
    












    


    