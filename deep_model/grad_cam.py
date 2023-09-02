# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
from asyncio import FastChildWatcher
import imghdr
from typing import ForwardRef
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
from densenet_plus import DenseNet2, DenseNet2_W_WeightPool, ResNet13, ResNet13_Change_Pool3, ResNet17_4_ConcatFusion_W_Loss, ResNet17_4_Refined_W_Loss, ResNet17_4_Refined_W_Loss_New2, VggNet



def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in, key='train'):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(128, 128))

    train_consistent_mean = torch.Tensor([0.4247, 0.4247, 0.4247])
    train_consistent_std = torch.Tensor([0.1309, 0.1309, 0.1309])
    val_consistent_mean = torch.Tensor([0.3906, 0.3906, 0.3906])
    val_consistent_std = torch.Tensor([0.1235, 0.1235, 0.1235])
    test_consistent_mean = torch.Tensor([0.4310, 0.4310, 0.4310])
    test_consistent_std = torch.Tensor([0.1327, 0.1327, 0.1327])

    img_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std),
        ]),
    }
    img_input = img_transform(img, img_transforms[key])
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def forward_hook(module, input, output):
    fmap_block.append(output)


def backward_hook2(module, grad_in, grad_out):
    grad_block2.append(grad_out[0].detach())


def backward_hook3(module, grad_in, grad_out):
    grad_block3.append(grad_out[0].detach())


def forward_hook2(module, input, output):
    fmap_block2.append(output)


def forward_hook3(module, input, output):
    fmap_block3.append(output)



def show_cam_on_image(img, mask, out_dir, img_name = 'cam.jpg', raw_img_name = 'raw.jpg'):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255  # 归一化操作
    cam = heatmap + np.float32(img)  # 热力图与原图进行叠加
    cam = cam / np.max(cam)  #再次进行归一化操作

    path_cam_img = os.path.join(out_dir, img_name)
    path_raw_img = os.path.join(out_dir, raw_img_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
    print("save successfully!")


def comp_class_vec(output_vec, index=None, num_classes=10):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(output_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, num_classes).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output_vec)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam -= np.min(cam)
    # cam /= (np.max(cam) - np.min(cam))
    print("min cam:", np.min(cam))
    cam /= np.max(cam) 


    return cam


def grad_cam_on_paper_proposed( model_name = 'proposed', img_name = 'B9RMLO.jpg',  nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'
   
    path_img = os.path.join(base_dir, key + '_imgs', img_name)
    path_net = os.path.join(base_dir, nets_folder_name, 'experim_4_resnet17_4_refined_w_loss_paper_1.0_epoch_32_0.935672514619883_0.8372093023255814_best.pkl')
    output_dir = os.path.join(base_dir, model_name + '_' + key + "_outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    net = ResNet17_4_Refined_W_Loss(num_mid_feats=32, num_classes=2, num_trans_dim=8)
    net.train(False)
    # net.cuda('cuda:7')
    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
    net.model1.layer4.register_forward_hook(forward_hook)
    net.model1.layer4.register_backward_hook(backward_hook)

    net.model2.features.denseblock4.register_forward_hook(forward_hook2)
    net.model2.features.denseblock4.register_backward_hook(backward_hook2)

    # forward
    output = net(img_input)[0]
    print("output", output)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()


    ##############################################
    # 生成resnet18对应的热力图
    #

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_resnet18_proposed.jpg', raw_img_name = img_name)


    ##############################################
    # 生成densenet121对应的热力图
    #
    
    # 生成cam
    grads_val = grad_block2[0].cpu().data.numpy().squeeze()
    fmap = fmap_block2[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_densenet121_proposed.jpg', raw_img_name = img_name)



def grad_cam_on_paper_proposed2( model_name = 'proposed', img_name = 'B9RMLO.jpg',  nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'
   
    path_img = os.path.join(base_dir, key + '_imgs', img_name)
    path_net = os.path.join(base_dir, nets_folder_name, 'experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2_epoch_34_0.9444444444444444_0.8604651162790697.pkl')
    output_dir = os.path.join(base_dir, model_name + '_' + key + "_outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    # net = ResNet17_4_Refined_W_Loss(num_mid_feats=32, num_classes=2, num_trans_dim=8)
    net = ResNet17_4_Refined_W_Loss_New2(num_mid_feats=32, num_classes=2)
    # net.cuda('cuda:7')
    net.train(False)

    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
    net.model1.layer4.register_forward_hook(forward_hook)
    net.model1.layer4.register_backward_hook(backward_hook)

    net.model2.features.denseblock4.register_forward_hook(forward_hook2)
    net.model2.features.denseblock4.register_backward_hook(backward_hook2)

    # forward
    output = net(img_input)[0]
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()


    ##############################################
    # 生成resnet18对应的热力图
    #

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_resnet18_proposed.jpg', raw_img_name = img_name)


    ##############################################
    # 生成densenet121对应的热力图
    #
    
    # 生成cam
    grads_val = grad_block2[0].cpu().data.numpy().squeeze()
    fmap = fmap_block2[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_densenet121_proposed.jpg', raw_img_name = img_name)


def grad_cam_on_paper_proposed3( model_name = 'proposed', img_name = 'B9RMLO.jpg',  nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'

    path_img = os.path.join(base_dir, key + '_imgs', img_name)
    path_net = os.path.join(base_dir, nets_folder_name, 'experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2_epoch_34_0.9444444444444444_0.8604651162790697.pkl')
    output_dir = os.path.join(base_dir, model_name + '_' + key + "_outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    # net = ResNet17_4_Refined_W_Loss(num_mid_feats=32, num_classes=2, num_trans_dim=8)
    net = ResNet17_4_Refined_W_Loss_New2(num_mid_feats=32, num_classes=2)
    # net.cuda('cuda:7')
    net.train(False)

    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
    net.model1.layer4.register_forward_hook(forward_hook)
    net.model1.layer4.register_backward_hook(backward_hook)

    net.model2.features.denseblock4.register_forward_hook(forward_hook2)
    net.model2.features.denseblock4.register_backward_hook(backward_hook2)

    net.fc.register_forward_hook(forward_hook3)
    net.fc.register_backward_hook(backward_hook3)

    # forward
    output = net(img_input)[0]
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()

    grads_val = grad_block3[0].cpu().data.numpy().squeeze()
    fmap = fmap_block3[0].cpu().data.numpy().squeeze()



#  可视化论文模型的代码
def grad_cam_on_paper_proposed4( model_name = 'proposed', img_name = 'B9RMLO.jpg',  nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'

    path_img = os.path.join(base_dir, key + '_imgs', img_name)
    path_net = os.path.join(base_dir, nets_folder_name, 'experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2_epoch_34_0.9444444444444444_0.8604651162790697.pkl')
    output_dir = os.path.join(base_dir, model_name + '_' + key + "_outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    print("img:", path_img)
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    net = ResNet17_4_Refined_W_Loss_New2(num_mid_feats=32, num_classes=2)
    net.train(False)
    # net.train(True)

    # print(net)

    # sys.exit(0)

    # net.cuda('cuda:7')
    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
    net.model1.layer4.register_forward_hook(forward_hook)
    net.model1.layer4.register_backward_hook(backward_hook)

    net.model2.features.denseblock4.register_forward_hook(forward_hook2)
    net.model2.features.denseblock4.register_backward_hook(backward_hook2)

    # forward
    output = net(img_input)[0]
    print("output", output)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    fc_weights = net(img_input)[3][0]
    fc_weights = fc_weights.cpu().data.numpy()
    res_fc_weight = fc_weights[:, 0]
    dense_fc_weight = fc_weights[:, 1]

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()


    res_grads_val = grad_block[0].cpu().data.numpy().squeeze()
    res_fmap = fmap_block[0].cpu().data.numpy().squeeze()
    res_weights = np.mean(res_grads_val, axis=(1, 2))  

    dense_grads_val = grad_block2[0].cpu().data.numpy().squeeze()
    dense_fmap = fmap_block2[0].cpu().data.numpy().squeeze()
    dense_weights = np.mean(dense_grads_val, axis=(1,2))

    cam = np.zeros(res_fmap.shape[1:], dtype=np.float32)  # cam shape (H, W)

    # for ith_ch in range(res_weights.shape[0]):
    #     cam += res_weights[ith_ch] * res_fc_weight[ith_ch] * res_fmap[ith_ch, :, :] + \
    #             dense_weights[ith_ch] * dense_fc_weight[ith_ch] * dense_fmap[ith_ch, :, :]


    # 获取resnet18 fc层的权重
    res18_fc_w = net.model1.fc.weight #(32,512)
    res18_fc_w = res18_fc_w.detach().numpy()
    
    # 获取densenet121 fc层的权重
    dense121_fc_w = net.model2.classifier.weight  # (32, 1024)
    dense121_fc_w = dense121_fc_w.detach().numpy()

    # 获取整个模型最后一层fc的权重 
    fc_w = net.fc.weight  #　(2, 32) 
    fc_w = fc_w.detach().numpy()


    res_fmap_updated = np.transpose(res_fmap,(1,2,0))
    res_fmap_updated =  np.matmul(res_fmap_updated, res18_fc_w.transpose(1,0))

    dense_fmap_updated = np.transpose(dense_fmap, (1,2,0))
    dense_fmap_updated = np.matmul(dense_fmap_updated, dense121_fc_w.transpose(1,0))

    weighted_merge_fmap = np.zeros_like(dense_fmap_updated, dtype=np.float32)
    for ch in range(res_fmap_updated.shape[-1]):
        weighted_merge_fmap[:,:, ch] = res_fc_weight[ch] * res_fmap_updated[:, :, ch] + dense_fc_weight[ch] * dense_fmap_updated[:,:, ch]
    
    if idx == '0':
        add_weight = fc_w[0]
    else:
        add_weight = fc_w[1]

    for ch in range(weighted_merge_fmap.shape[-1]):
        cam += weighted_merge_fmap[:, :, ch] * add_weight[ch]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam -= np.min(cam)
    # print("min cam:", np.min(cam))
    # print("max cam:", np.max(cam))
    cam /= (np.max(cam) - np.min(cam))

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_merged.jpg', raw_img_name = img_name)



#  可视化错误分类结果的代码
def grad_cam_on_paper_proposed5(path_img, output_dir, model_name = 'proposed', nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'

    path_net = os.path.join(base_dir, nets_folder_name, 'experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2_epoch_34_0.9444444444444444_0.8604651162790697.pkl')

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    print("img:", path_img)
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    net = ResNet17_4_Refined_W_Loss_New2(num_mid_feats=32, num_classes=2)
    net.train(False)
    # net.train(True)

    # print(net)

    # sys.exit(0)

    # net.cuda('cuda:7')
    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
    net.model1.layer4.register_forward_hook(forward_hook)
    net.model1.layer4.register_backward_hook(backward_hook)

    net.model2.features.denseblock4.register_forward_hook(forward_hook2)
    net.model2.features.denseblock4.register_backward_hook(backward_hook2)

    # forward
    output = net(img_input)[0]
    print("output", output)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    fc_weights = net(img_input)[3][0]
    fc_weights = fc_weights.cpu().data.numpy()
    res_fc_weight = fc_weights[:, 0]
    dense_fc_weight = fc_weights[:, 1]

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()


    res_grads_val = grad_block[0].cpu().data.numpy().squeeze()
    res_fmap = fmap_block[0].cpu().data.numpy().squeeze()
    res_weights = np.mean(res_grads_val, axis=(1, 2))  

    dense_grads_val = grad_block2[0].cpu().data.numpy().squeeze()
    dense_fmap = fmap_block2[0].cpu().data.numpy().squeeze()
    dense_weights = np.mean(dense_grads_val, axis=(1,2))

    cam = np.zeros(res_fmap.shape[1:], dtype=np.float32)  # cam shape (H, W)

    # 获取resnet18 fc层的权重
    res18_fc_w = net.model1.fc.weight #(32,512)
    res18_fc_w = res18_fc_w.detach().numpy()
    
    # 获取densenet121 fc层的权重
    dense121_fc_w = net.model2.classifier.weight  # (32, 1024)
    dense121_fc_w = dense121_fc_w.detach().numpy()

    # 获取整个模型最后一层fc的权重 
    fc_w = net.fc.weight  #　(2, 32) 
    fc_w = fc_w.detach().numpy()


    res_fmap_updated = np.transpose(res_fmap,(1,2,0))
    res_fmap_updated =  np.matmul(res_fmap_updated, res18_fc_w.transpose(1,0))

    dense_fmap_updated = np.transpose(dense_fmap, (1,2,0))
    dense_fmap_updated = np.matmul(dense_fmap_updated, dense121_fc_w.transpose(1,0))

    weighted_merge_fmap = np.zeros_like(dense_fmap_updated, dtype=np.float32)
    for ch in range(res_fmap_updated.shape[-1]):
        weighted_merge_fmap[:,:, ch] = res_fc_weight[ch] * res_fmap_updated[:, :, ch] + dense_fc_weight[ch] * dense_fmap_updated[:,:, ch]
    
    if idx == '0':
        add_weight = fc_w[0]
    else:
        add_weight = fc_w[1]

    for ch in range(weighted_merge_fmap.shape[-1]):
        cam += weighted_merge_fmap[:, :, ch] * add_weight[ch]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam -= np.min(cam)
    # print("min cam:", np.min(cam))
    # print("max cam:", np.max(cam))
    cam /= (np.max(cam) - np.min(cam))

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_merged.jpg', raw_img_name = img_name)



def grad_cam_on_paper_proposed_w_concat_fusion( model_name = 'proposed', img_name = 'B9RMLO.jpg',  nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'

    path_img = os.path.join(base_dir, key + '_imgs', img_name)
    path_net = os.path.join(base_dir, nets_folder_name, 'experim_4_resnet17_4_ConcatFusion_w_loss_paper2_1.0_2_epoch_28_0.8888888888888888_0.8372093023255814.pkl')
    output_dir = os.path.join(base_dir, model_name + '_' + key + "_outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    print("img:", path_img)
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    net = ResNet17_4_ConcatFusion_W_Loss(num_mid_feats=32, num_classes=2)
    net.train(False)

    # net.cuda('cuda:7')
    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
    net.model1.layer4.register_forward_hook(forward_hook)
    net.model1.layer4.register_backward_hook(backward_hook)

    net.model2.features.denseblock4.register_forward_hook(forward_hook2)
    net.model2.features.denseblock4.register_backward_hook(backward_hook2)


    # forward
    output = net(img_input)[0]
    print("output", output)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()

    res_grads_val = grad_block[0].cpu().data.numpy().squeeze()
    res_fmap = fmap_block[0].cpu().data.numpy().squeeze()
    res_weights = np.mean(res_grads_val, axis=(1, 2))  

    dense_grads_val = grad_block2[0].cpu().data.numpy().squeeze()
    dense_fmap = fmap_block2[0].cpu().data.numpy().squeeze()
    dense_weights = np.mean(dense_grads_val, axis=(1,2))

    cam = np.zeros(res_fmap.shape[1:], dtype=np.float32)  # cam shape (H, W)


    # 获取resnet18 fc层的权重
    res18_fc_w = net.model1.fc.weight #(32,512)
    res18_fc_w = res18_fc_w.detach().numpy()

    # 获取densenet121 fc层的权重
    dense121_fc_w = net.model2.classifier.weight  # (32, 1024)
    dense121_fc_w = dense121_fc_w.detach().numpy()

    # 获取整个模型最后一层fc的权重 
    fc_w = net.fc.weight  #　(2, 64) 
    fc_w = fc_w.detach().numpy()


    res_fmap_updated = np.transpose(res_fmap,(1,2,0))
    res_fmap_updated =  np.matmul(res_fmap_updated, res18_fc_w.transpose(1,0)) #(4,4,512) * (512,32) 

    dense_fmap_updated = np.transpose(dense_fmap, (1,2,0))
    dense_fmap_updated = np.matmul(dense_fmap_updated, dense121_fc_w.transpose(1,0))


    merged_fmap = np.concatenate((res_fmap_updated, dense_fmap_updated), -1) # (4, 4, 64)
    if idx == 0:
        fc_w_index = 0
    elif idx == 1:
        fc_w_index = 1
    else:
        fc_w_index = -2

    for ch in range(merged_fmap.shape[-1]):
        cam += fc_w[fc_w_index][ch] * merged_fmap[:, :, ch]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam -= np.min(cam)
    # print("min cam:", np.min(cam))
    # print("max cam:", np.max(cam))
    cam /= (np.max(cam) - np.min(cam))

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_merged.jpg', raw_img_name = img_name)



def grad_cam_on_single_model(model, checkpoints_path, model_name = 'proposed', img_name = 'B9RMLO.jpg',  nets_folder_name = 'nets',  key='train'):

    base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results'
   
    path_img = os.path.join(base_dir, key + '_imgs', img_name)
    print(checkpoints_path)
    path_net = os.path.join(base_dir, nets_folder_name, checkpoints_path)
    print("path net is:" + path_net)
    output_dir = os.path.join(base_dir, model_name + '_' + key + "_outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    classes = ('benign', 'malignant')

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img, key)
    net = model
    # net.cuda('cuda:7')
    net.train(False)
    net.load_state_dict(torch.load(path_net, map_location=lambda storage, loc: storage))

    # 注册hook
  

    # forward
    output = net(img_input)[0]
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, num_classes=2)
    class_loss.backward()


    ##############################################
    # 生成resnet18对应的热力图
    #

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 128))) / 255
    # img_show是原图，cam是对应的热力图，output_dir是对应的输出文件夹
    # img_show和cam都归一化到了0-1，且大小都是128 * 128
    show_cam_on_image(img_show, cam, output_dir, img_name=img_name + '_cam_' + model_name + '.jpg', raw_img_name = img_name)




if __name__ == '__main__':

    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/train_imgs'
    # key = 'train'

    # fmap_block = list()
    # grad_block = list()

    # fmap_block2 = list()
    # grad_block2 = list()

    # for img_name in os.listdir(imgs_dir):
    #     grad_cam_on_paper_proposed(model_name = 'proposed', img_name = img_name, key=key)


    ###################################################################

    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/test_imgs'
    # key = 'test'

    # fmap_block = list()
    # grad_block = list()

    # fmap_block2 = list()
    # grad_block2 = list()

    # for img_name in os.listdir(imgs_dir):
    #     grad_cam_on_paper_proposed(model_name = 'proposed', img_name = img_name, key=key)


    #######################################################################

    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/val_imgs'
    # key = 'val'

    # fmap_block = list()
    # grad_block = list()

    # fmap_block2 = list()
    # grad_block2 = list()

    # for img_name in os.listdir(imgs_dir):
    #     grad_cam_on_paper_proposed(model_name = 'proposed', img_name = img_name, key=key)


    #######################################################

    # model = ResNet13_Change_Pool3(num_mid_feats=32, num_classes=2)
    # model_name = 'resnet_change_pool3'

    # key = 'train'
    # key = 'val'
    # # key = 'test'
    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    # checkpoints_path = 'resnet13_change_pool3_paper_epoch_9_0.935672514619883_0.8604651162790697_best.pkl'

    # fmap_block = list()
    # grad_block = list()

    # model.model[7].register_forward_hook(forward_hook)
    # model.model[7].register_backward_hook(backward_hook)

    # print(model)

    # for img_name in os.listdir(imgs_dir):
    #     grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)

    #####################################################################

    # model = DenseNet2_W_WeightPool(num_mid_features=32, num_classes=2, num_trans_dim=100)
    # model_name = 'DenseNet2_W_WeightPool'

    # key_list = ['train','val','test' ]
   
    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     checkpoints_path = 'DenseNet2_W_WeightPool_100_paper_epoch_30_0.9707602339181286_0.8837209302325582_best.pkl'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model[11].register_forward_hook(forward_hook)
    #     model.model[11].register_backward_hook(backward_hook)

    #     print(model)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)

    
    ##################################################################################

    # model = DenseNet2(num_mid_features=32, num_classes=2)
    # model_name = 'densenet121_paper_1.0_1'

    # key_list = ['train','val','test' ]
    # checkpoints_path = 'experim_4_densenet121_paper_1.0_1_epoch_32_0.9385964912280702_0.7674418604651163_best.pkl'

    # print(model)

    # # sys.exit(0)
   
    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model.features.register_forward_hook(forward_hook)
    #     model.model.features.register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)
    
    ########################################################################################################

    # model = ResNet13(num_mid_feats=32, num_classes=2)
    # model_name = 'resnet18_paper_1.0_1'

    # key_list = ['train','val','test' ]

    # checkpoints_path = 'experim_4_resnet18_paper_1.0_1_epoch_28_0.956140350877193_0.7674418604651163_best.pkl'

    # print(model)

    # # sys.exit(0)

    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model.layer4.register_forward_hook(forward_hook)
    #     model.model.layer4.register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)


    ############################################################################################################

    # key = 'val'
    # key = 'test'
    # key = 'train'
    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    # fmap_block = list()
    # grad_block = list()

    # fmap_block2 = list()
    # grad_block2 = list()

    # fmap_block3 = list()
    # grad_block3 = list()

    # for img_name in os.listdir(imgs_dir):
    #     # grad_cam_on_paper_proposed2(model_name = 'proposed', img_name = img_name, key=key)
    #     # print(img_name)
        
    #     grad_cam_on_paper_proposed4(model_name = 'proposed', img_name = img_name, key=key)

    #############################################################################################################

    # model = ResNet13(num_mid_feats=32, num_classes=2)
    # model_name = 'resnet18_paper2_1.0_1'

    # key_list = ['train','val','test' ]

    # checkpoints_path = 'experim_4_resnet18_paper2_1.0_1_epoch_4_0.7514619883040936_0.813953488372093.pkl'

    # print(model)

    # # sys.exit(0)

    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model.layer4.register_forward_hook(forward_hook)
    #     model.model.layer4.register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)
    
    # sys.exit(0)

    ###############################################################################################################

    # model = DenseNet2(num_mid_features=32, num_classes=2)
    # model_name = 'densenet121_paper2_1.0_2'

    # key_list = ['train','val','test' ]

    # checkpoints_path = 'experim_4_densenet121_paper2_1.0_2_epoch_24_0.8713450292397661_0.8372093023255814.pkl'

    # print(model)

    # # sys.exit(0)

    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model.features.denseblock3.register_forward_hook(forward_hook)
    #     model.model.features.denseblock3.register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)

    # sys.exit(0)

    #################################################################################################################

    # model = ResNet13_Change_Pool3(num_mid_feats=32, num_classes=2)
    # model_name = 'resnet13_change_pool3_paper2'

    # key_list = ['train','val','test' ]

    # checkpoints_path = 'resnet13_change_pool3_paper2_epoch_16_0.9210526315789473_0.8372093023255814.pkl'

    # print(model)

    # # sys.exit(0)

    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model[-3].register_forward_hook(forward_hook)
    #     model.model[-3].register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)

    # sys.exit(0)

    #################################################################################################################################

    # model_name = 'resnet17_4_ConcatFusion_w_loss_paper2'

    # key_list = ['train','val','test' ]

    
    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     fmap_block2 = list()
    #     grad_block2 = list()

    #     for img_name in os.listdir(imgs_dir):

    #         grad_cam_on_paper_proposed_w_concat_fusion(model_name = model_name, img_name = img_name, key=key)

    # sys.exit(0)

    ###########################################################################################################################################

    # model = VggNet(num_classes=2)
    # model_name = 'vggnet_paper2'

    # key_list = ['train','val','test' ]

    # checkpoints_path = 'experim_4_vggnet_paper2_epoch_27_1.0_0.8372093023255814.pkl'

    # print(model)

    # # sys.exit(0)

    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model.features.register_forward_hook(forward_hook)
    #     model.model.features.register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)

    # sys.exit(0)

    #######################################################################################################################################

    # model = None
    # model_name = ''

    # key_list = ['train','val','test' ]

    # checkpoints_path = ''

    # # print(model)

    # # sys.exit(0)

    # for key in key_list:
    #     imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

    #     fmap_block = list()
    #     grad_block = list()

    #     model.model.layer4.register_forward_hook(forward_hook)
    #     model.model.layer4.register_backward_hook(backward_hook)

    #     for img_name in os.listdir(imgs_dir):
    #         grad_cam_on_single_model(model, checkpoints_path = checkpoints_path , model_name = model_name, img_name = img_name, key=key)


    ###############################################################################################################################

    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/proposed_wrong_pred_imgs/val_malignant_mis_classified'
    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/proposed_wrong_pred_imgs/val_benign_mis_classified'
    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/proposed_wrong_pred_imgs/test_malignant_mis_classified'
    # imgs_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/proposed_wrong_pred_imgs/test_benign_mis_classified'
    # key = 'test'
    # key = 'val'

    # fmap_block = list()
    # grad_block = list()

    # fmap_block2 = list()
    # grad_block2 = list()

    # fmap_block3 = list()
    # grad_block3 = list()

    # for img_name in os.listdir(imgs_dir):
        
    #     path_img = os.path.join(imgs_dir, img_name)
    #     grad_cam_on_paper_proposed5(path_img=path_img, output_dir=imgs_dir, model_name = 'proposed', key=key)


    ######################################################################################################################################################

    # densenet 加上加权池化后的可视化结果

    model = DenseNet2_W_WeightPool(num_mid_features=32, num_classes=2, num_trans_dim=119)
    model_name = 'DenseNet2_W_WeightPool_119_paper'

    key_list = ['train','val','test' ]

    checkpoints_path = 'DenseNet2_W_WeightPool_119_paper_epoch_21_0.9327485380116959_0.7906976744186046.pkl'

    # print(model)

    # sys.exit(0)

    for key in key_list:
        imgs_dir = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/grad_cam_results/' + key + '_imgs'

        fmap_block = list()
        grad_block = list()

        model.model[-5].register_forward_hook(forward_hook)
        model.model[-5].register_backward_hook(backward_hook)

        for img_name in os.listdir(imgs_dir):
            grad_cam_on_single_model(model, checkpoints_path , model_name = model_name, img_name = img_name, key=key)

    sys.exit(0)

    



    





    














