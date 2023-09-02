from tkinter import N
from sklearn import metrics
import os
import pickle


def cal_metrics(labels_arr, preds_arr):

    auc_score = metrics.roc_auc_score(labels_arr, preds_arr)
    fpr, tpr, thresholds = metrics.roc_curve(labels_arr, preds_arr)

    true_positive_nums = 0
    false_positive_nums = 0
    true_negative_nums = 0
    false_negative_nums = 0
    for i in range(len(labels_arr)):
        pred_label = preds_arr[i]
        actual_label = labels_arr[i]
        if pred_label == 1 and actual_label == 1:
            true_positive_nums += 1
        elif pred_label == 1 and actual_label == 0:
            false_positive_nums += 1
        elif pred_label == 0 and actual_label == 0:
            true_negative_nums += 1
        else:
            false_negative_nums += 1
    
    # print("preds arr:", preds_arr)
    # print("labels arr:", labels_arr)
    if (true_positive_nums + false_negative_nums) != 0:
        sensitivity = true_positive_nums / (true_positive_nums + false_negative_nums)
    else:
        sensitivity = -1.0

    if (true_negative_nums + false_positive_nums) != 0:
        specificity = true_negative_nums / (true_negative_nums + false_positive_nums)
    else:
        specificity =  -1.0

    if (true_positive_nums + false_positive_nums) != 0:
        ppv = true_positive_nums / (true_positive_nums + false_positive_nums)
    else:
        ppv =  -1.0

    if (true_negative_nums + false_negative_nums) != 0:
        npv = true_negative_nums / (true_negative_nums + false_negative_nums)
    else:
        npv =  -1.0

    return auc_score, sensitivity, specificity, ppv, npv


import pylab as plt

# 固定三组数据一起显示
def ks(y_predicted1, y_true1, y_predicted2, y_true2, y_predicted3, y_true3):
  Font={'size':18, 'family':'Times New Romance'}
  
  label1=y_true1
  label2=y_true2
  label3=y_true3
  fpr1,tpr1,thres1 = metrics.roc_curve(label1, y_predicted1)
  fpr2,tpr2,thres2 = metrics.roc_curve(label2, y_predicted2)
  fpr3,tpr3,thres3 = metrics.roc_curve(label3, y_predicted3)
  roc_auc1 = metrics.auc(fpr1, tpr1)
  roc_auc2 = metrics.auc(fpr2, tpr2)
  roc_auc3 = metrics.auc(fpr3, tpr3)
  
  plt.figure(figsize=(6,6))
  plt.plot(fpr1, tpr1, 'b', label = 'Stacking = %0.3f' % roc_auc1, color='Red')
  plt.plot(fpr2, tpr2, 'b', label = 'XGBoost = %0.3f' % roc_auc2, color='k')
  plt.plot(fpr3, tpr3, 'b', label = 'Random Forest = %0.3f' % roc_auc3, color='RoyalBlue')
  plt.legend(loc = 'lower right', prop=Font)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate', Font)
  plt.xlabel('False Positive Rate', Font)
  plt.tick_params(labelsize=15)
  plt.show()
  return abs(fpr1 - tpr1).max(),abs(fpr2 - tpr2).max(),abs(fpr3 - tpr3).max()


# 可以任意组一起显示
# 传入的y_predicted_group均为列表

def ks_with_random_num(y_predicted_group, y_label_group, y_legend_group, color_group, linestyle_group, save_path):

  group_num = len(y_predicted_group)
  print("group num:", group_num)
  print("color group:", color_group)
  print("line group:", linestyle_group)
  fpr_group = [None] * group_num
  tpr_group = [None] * group_num
  threshold_group = [None] * group_num
  roc_auc_group = [None] * group_num

  Font={'size':10, }

  for i in range(group_num):
      fpr, tpr, threshold = metrics.roc_curve(y_label_group[i], y_predicted_group[i])
      fpr_group[i] = fpr
      tpr_group[i] = tpr
      threshold_group[i] = threshold

      roc_auc_s = metrics.auc(fpr, tpr)
    #   roc_auc_s = metrics.roc_auc_score(y_label_group[i], y_predicted_group[i])
      roc_auc_group[i] = roc_auc_s


  plt.figure(figsize=(10,8))
  for i in range(group_num):
      plt.plot(fpr_group[i], tpr_group[i], label = y_legend_group[i] + ' = %0.3f' % roc_auc_group[i], 
            color=color_group[i], linestyle=linestyle_group[i])

  plt.legend(loc = 'lower right', prop=Font)
#   plt.plot([0, 1], [0, 1],'--', color='black')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate', Font)
  plt.xlabel('False Positive Rate', Font)
  plt.tick_params(labelsize=15)
  plt.savefig(save_path)
  plt.show()


if __name__ == "__main__":
    
    # pickle_dir_1 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_39_final_final2_train_part_2/saved_dicts'
    # pickle_file_group = [os.path.join(pickle_dir_1, '3_train_dicts.pickle'), 
    #                     os.path.join(pickle_dir_1, '3_test_dicts.pickle')]

    # pickle_dir_2 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densenet121_net17_2_train_part_2/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_2, '40_train_dicts.pickle'), 
    #                     os.path.join(pickle_dir_2, '40_test_dicts.pickle')]

    # pickle_dir_3 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet_wrapper_train_part_2/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_3, '43_train_dicts.pickle'), 
    #                     os.path.join(pickle_dir_3, '43_test_dicts.pickle')]
        
    # pickle_dir_4 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/densnet_wrapper_train_part_2/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_4, '59_train_dicts.pickle'), 
    #                     os.path.join(pickle_dir_4, '59_test_dicts.pickle')]

    # y_predicted_group = []
    # y_label_group = []
    # y_legend_group = ['proposed(train)', 'proposed(test)', 'resnet50 + handcrafted_features(train)', 'resnet50 + handcrafted_features(test)', 
    #                   'resnet50(train)', 'resnet50(test)',  'densenet121(train)', 'densenet121(test)']
    # # 'RoyalBlue'
    # color_group = ['Red', 'Red', 'Purple', 'Purple', 'green', 'green', 'yellow', 'yellow']
    # linestyle_group = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']

    # for i in range(len(pickle_file_group)):
    #     pickle_file_path = pickle_file_group[i]
    #     with open(pickle_file_path, 'rb') as fp:
    #         dict = pickle.load(fp)
    #     y_predicted_group.append(dict['preds'])
    #     y_label_group.append(dict['labels'])

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/figs_saved/roc_curve.jpg'
    # ks_with_random_num(y_predicted_group=y_predicted_group, y_label_group=y_label_group, 
    #                 y_legend_group=y_legend_group, color_group=color_group,  linestyle_group= linestyle_group, 
    #                 save_path=save_path)


    ############################################################################################

    # pickle_dir_1 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/net17simple_train_part_2/saved_dicts'
    # pickle_file_group = [os.path.join(pickle_dir_1, '14_train_dicts.pickle'), 
    #                     os.path.join(pickle_dir_1, '14_test_dicts.pickle')]

    # y_predicted_group = []
    # y_label_group = []
    # y_legend_group = ['proposed(train)', 'proposed(test)', 'resnet50 + manual_features(train)', 'resnet50 + manual_features(test)', 
    #                   'resnet50(train)', 'resnet50(test)',  'densenet121(train)', 'densenet121(test)']
    # # 'RoyalBlue'
    # color_group = ['Red', 'Red', 'Purple', 'Purple', 'green', 'green', 'yellow', 'yellow']
    # linestyle_group = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']

    # for i in range(len(pickle_file_group)):
    #     pickle_file_path = pickle_file_group[i]
    #     with open(pickle_file_path, 'rb') as fp:
    #         dict = pickle.load(fp)
    #     y_predicted_group.append(dict['preds'])
    #     y_label_group.append(dict['labels'])

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/figs_saved/roc_curve2.jpg'
    # ks_with_random_num(y_predicted_group=y_predicted_group, y_label_group=y_label_group, 
    #                 y_legend_group=y_legend_group, color_group=color_group,  linestyle_group= linestyle_group, 
    #                 save_path=save_path)

    ##########################################################################################################################################

    # pickle_dir_1 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_paper_1.0/saved_dicts'
    # pickle_file_group = [ os.path.join(pickle_dir_1, '32_val_dicts.pickle'), 
    #                     os.path.join(pickle_dir_1, '-1_test_dicts.pickle')]


    # pickle_dir_2 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_ConcatFusion_w_loss_paper_1.0_1/saved_dicts'
    # pickle_file_group += [ os.path.join(pickle_dir_2, '28_val_dicts.pickle'), 
    #                          os.path.join(pickle_dir_2, '-1_test_dicts.pickle')]


    # pickle_dir_3 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_fdffe_paper_1.0_1/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_3, '12_val_dicts.pickle'), 
    #                       os.path.join(pickle_dir_3, '-1_test_dicts.pickle')]
        

    # pickle_dir_4 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_vggnet_paper/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_4, '24_val_dicts.pickle'), 
    #                      os.path.join(pickle_dir_4, '-1_test_dicts.pickle')]

    
    # pickle_dir_5 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_vggnet16_paper/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_5, '19_val_dicts.pickle'), 
    #                          os.path.join(pickle_dir_5, '-1_test_dicts.pickle')]


    # pickle_dir_6 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_inception_paper/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_6, '32_val_dicts.pickle'), 
    #                      os.path.join(pickle_dir_6, '-1_test_dicts.pickle')]


    # pickle_dir_7 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_efficient_paper/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_7, '28_val_dicts.pickle'), 
    #                      os.path.join(pickle_dir_7, '-1_test_dicts.pickle')]
        


    # pickle_dir_8 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_alexnet_paper/saved_dicts'
    # pickle_file_group += [ os.path.join(pickle_dir_8, '30_val_dicts.pickle'), 
    #                      os.path.join(pickle_dir_8, '-1_test_dicts.pickle')]


    # pickle_dir_9 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/resnet13_change_pool3_paper/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_9, '9_val_dicts.pickle'), 
    #                          os.path.join(pickle_dir_9, '-1_test_dicts.pickle')]


    # pickle_dir_10 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/DenseNet2_W_WeightPool_100_paper/saved_dicts'
    # pickle_file_group += [os.path.join(pickle_dir_10, '30_val_dicts.pickle'), 
    #                          os.path.join(pickle_dir_10, '-1_test_dicts.pickle')]


    # y_predicted_group = []
    # y_label_group = []
    # y_legend_group = ['proposed(validation)', 'proposed(test)', 'resnet18 and densenet121 w concatenation(validation)', 'resnet18 and densenet121 w concatenation(test)', 
    #                 'FDFFE(validation)', 'FDFFE(test)', 'vgg11(validation)', 'vgg11(test)', 
    #                 'vgg16(validation)', 'vgg16(test)', 'inception-v3(validation)', 'inception-v3(test)', 
    #                 'efficientnet-b0(validation)', 'efficientnet-b0(test)', 'alexnet(validation)', 'alexnet(test)', 
    #                 'resnet18 w weighted pooling(validation)', 'resnet18 w weighted pooling(test)', 'densenet121 w weighted pooling(validation)', 'densenet121 w weighted pooling(test)', 
    #                   ]


    # # 'RoyalBlue'
    # color_group = ['Red', 'Red', 'Purple', 'Purple', 'green', 'green', 'blue', 'blue', 'orange', 'orange', 
    #                 'cyan', 'cyan', 'pink', 'pink',  'magenta', 'magenta', 'brown','brown', 'olive', 'olive'
    #              ]
    # linestyle_group = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
    #         'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed'
    # ]


    # for i in range(len(pickle_file_group)):
    #     pickle_file_path = pickle_file_group[i]
    #     with open(pickle_file_path, 'rb') as fp:
    #         dict = pickle.load(fp)
    #     y_predicted_group.append(dict['preds'])
    #     y_label_group.append(dict['labels'])


    ##############################################################################################
    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/figs_saved/roc_curve_paper.jpg'
    # ks_with_random_num(y_predicted_group=y_predicted_group, y_label_group=y_label_group, 
    #                 y_legend_group=y_legend_group, color_group=color_group,  linestyle_group= linestyle_group, 
    #                 save_path=save_path)


    # pickle_dir_1 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2/saved_dicts'
    # pickle_file_group = [ os.path.join(pickle_dir_1, '34_val_dicts.pickle'), 
    #                     os.path.join(pickle_dir_1, '-1_test_dicts.pickle')]

    # y_predicted_group = []
    # y_label_group = []
    # y_legend_group = ['proposed(validation)', 'proposed(test)']
    

    # # 'RoyalBlue'
    # color_group = ['Red', 'Red']
    # linestyle_group = ['solid', 'dashed']


    # for i in range(len(pickle_file_group)):
    #     pickle_file_path = pickle_file_group[i]
    #     with open(pickle_file_path, 'rb') as fp:
    #         dict = pickle.load(fp)
    #     y_predicted_group.append(dict['preds'])
    #     y_label_group.append(dict['labels'])


    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/figs_saved/roc_curve_paper2.jpg'
    # ks_with_random_num(y_predicted_group=y_predicted_group, y_label_group=y_label_group, 
    #                 y_legend_group=y_legend_group, color_group=color_group,  linestyle_group= linestyle_group, 
    #                 save_path=save_path)


    #######################################################################################################################################

    name_dict = {}
    pickle_file_group_dict = {}
    color_dict = {}

    pickle_dir_1 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_refined_w_loss_new2_paper2_1.0_2/saved_dicts'
    pickle_file_group_1 = [ os.path.join(pickle_dir_1, '34_val_dicts.pickle'), 
                        os.path.join(pickle_dir_1, '-1_test_dicts.pickle')]
    name_1 = ['proposed(val)', 'proposed(test)']
    name_dict['1'] = name_1
    pickle_file_group_dict['1'] = pickle_file_group_1
    color_dict['1'] = ['Red', 'Red']

    


    pickle_dir_2 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_densenet121_paper2_1.0_2/saved_dicts'
    pickle_file_group_2 = [ os.path.join(pickle_dir_2, '24_val_dicts.pickle'), 
                        os.path.join(pickle_dir_2, '-1_test_dicts.pickle')]
    name_2 = ['densenet121(val)', 'densenet121(test)']
    name_dict['2'] = name_2
    pickle_file_group_dict['2'] = pickle_file_group_2
    color_dict['2'] = ['Purple', 'Purple']





    pickle_dir_3 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_fdffe_paper2_1.0_1/saved_dicts'
    pickle_file_group_3 = [os.path.join(pickle_dir_3, '12_val_dicts.pickle'), 
                          os.path.join(pickle_dir_3, '-1_test_dicts.pickle')]
    name_3 = ['FDFFE(val)', 'FDFFE(test)']
    name_dict['3'] = name_3
    pickle_file_group_dict['3'] = pickle_file_group_3
    color_dict['3'] = ['green', 'green',]
    




    pickle_dir_4 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_vggnet_paper2/saved_dicts'
    pickle_file_group_4 = [os.path.join(pickle_dir_4, '27_val_dicts.pickle'), 
                         os.path.join(pickle_dir_4, '-1_test_dicts.pickle')]
    name_4 = [   'vgg11(val)', 'vgg11(test)']
    name_dict['4'] = name_4
    pickle_file_group_dict['4'] = pickle_file_group_4
    color_dict['4'] = ['blue', 'blue']

    


    pickle_dir_5 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_vggnet16_paper/saved_dicts'
    pickle_file_group_5 = [os.path.join(pickle_dir_5, '17_val_dicts.pickle'), 
                             os.path.join(pickle_dir_5, '-1_test_dicts.pickle')]
    name_5 = [ 'vgg16(val)', 'vgg16(test)', ]
    name_dict['5'] = name_5
    pickle_file_group_dict['5'] = pickle_file_group_5
    color_dict['5'] = ['orange', 'orange',]




    pickle_dir_6 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_efficient_paper2/saved_dicts'
    pickle_file_group_6 = [os.path.join(pickle_dir_6, '31_val_dicts.pickle'), 
                         os.path.join(pickle_dir_6, '-1_test_dicts.pickle')]
    name_6 = [ 'efficientnet-b0(val)', 'efficientnet-b0(test)', ]
    name_dict['6'] = name_6
    pickle_file_group_dict['6'] = pickle_file_group_6
    color_dict['6'] = ['cyan', 'cyan']




    pickle_dir_7 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_alexnet_paper2/saved_dicts'
    pickle_file_group_7 = [os.path.join(pickle_dir_7, '33_val_dicts.pickle'), 
                         os.path.join(pickle_dir_7, '-1_test_dicts.pickle')]    
    name_7 = [   'alexnet(val)', 'alexnet(test)', ]
    name_dict['7'] = name_7
    pickle_file_group_dict['7'] = pickle_file_group_7
    color_dict['7'] = ['pink', 'pink',]




    pickle_dir_8 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/resnet13_change_pool3_paper2/saved_dicts'
    pickle_file_group_8 = [ os.path.join(pickle_dir_8, '16_val_dicts.pickle'), 
                         os.path.join(pickle_dir_8, '-1_test_dicts.pickle')]
    name_8 = [  'resnet18 w weighted pooling(val)', 'resnet18 w weighted pooling(test)',]
    name_dict['8'] = name_8
    pickle_file_group_dict['8'] = pickle_file_group_8
    color_dict['8'] = ['magenta', 'magenta',]




    pickle_dir_9 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_proposed_wo_multi_output_paper2_1.0_1/saved_dicts'
    pickle_file_group_9 = [os.path.join(pickle_dir_9, '16_val_dicts.pickle'), 
                             os.path.join(pickle_dir_9, '-1_test_dicts.pickle')]
    name_9 = ['proposed w/o multi-output(val)', 'proposed w/o multi-output(test)']
    name_dict['9'] = name_9
    pickle_file_group_dict['9'] = pickle_file_group_9
    color_dict['9'] = ['brown','brown',]




    pickle_dir_10 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet17_4_ConcatFusion_w_loss_paper2_1.0_2/saved_dicts'
    pickle_file_group_10 = [os.path.join(pickle_dir_10, '28_val_dicts.pickle'), 
                             os.path.join(pickle_dir_10, '-1_test_dicts.pickle')]
    name_10 = ['resnet18 and densenet121 w concatenation fusion(val)', 'resnet18 and densenet121 w concatenation fusion(test)']
    name_dict['10'] = name_10
    pickle_file_group_dict['10'] = pickle_file_group_10
    color_dict['10'] = ['olive', 'olive']




    pickle_dir_11 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/experim_4_resnet18_paper2_1.0_1/saved_dicts'
    pickle_file_group_11 = [os.path.join(pickle_dir_11, '4_val_dicts.pickle'), 
                             os.path.join(pickle_dir_11, '-1_test_dicts.pickle')]
    name_11 = ['resnet18(val)', 'resnet18(test)']
    name_dict['11'] = name_11
    pickle_file_group_dict['11'] = pickle_file_group_11
    color_dict['11'] = ['RoyalBlue', 'RoyalBlue',]



    pickle_dir_12 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/DenseNet2_W_WeightPool_120_paper/saved_dicts'
    pickle_file_group_12 = [os.path.join(pickle_dir_12, '34_val_dicts.pickle'), 
                             os.path.join(pickle_dir_12, '-1_test_dicts.pickle')]
    name_12 = [ 'densenet121 w weighted pooling(val)', 'densenet121 w weighted pooling(test)', ]
    name_dict['12'] = name_12
    pickle_file_group_dict['12'] = pickle_file_group_12
    color_dict['12'] = ['Turquoise', 'Turquoise']

    
    pickle_dir_13 = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/DenseNet2_W_WeightPool_119_paper/saved_dicts'
    pickle_file_group_13 = [os.path.join(pickle_dir_13, '21_val_dicts.pickle'), 
                             os.path.join(pickle_dir_13, '-1_test_dicts.pickle')]
    name_13 = [ 'densenet121 w weighted pooling(val)', 'densenet121 w weighted pooling(test)', ]
    name_dict['13'] = name_13
    pickle_file_group_dict['13'] = pickle_file_group_13
    color_dict['13'] = ['DarkGreen', 'DarkGreen']

    
    pickle_dir_14 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/extra_saved/saved_dicts'
    pickle_file_group_14 = [os.path.join(pickle_dir_14, 'knc_val_dicts.pickle'), 
                             os.path.join(pickle_dir_14, 'knc_test_dicts.pickle')]
    name_14 = [ 'knn (val)', 'knn (test)', ]
    name_dict['14'] = name_14
    pickle_file_group_dict['14'] = pickle_file_group_14
    color_dict['14'] = ['RoyalBlue', 'RoyalBlue']


    pickle_dir_15 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/extra_saved/saved_dicts'
    pickle_file_group_15 = [os.path.join(pickle_dir_15, 'rf_model_val_dicts.pickle'), 
                             os.path.join(pickle_dir_15, 'rf_model_test_dicts.pickle')]
    name_15 = [ 'random forest (val)', 'random forest (test)', ]
    name_dict['15'] = name_15
    pickle_file_group_dict['15'] = pickle_file_group_15
    color_dict['15'] = ['Gray', 'Gray']


    pickle_dir_16 = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/extra_saved/saved_dicts'
    pickle_file_group_16 = [os.path.join(pickle_dir_16, 'svc_on_whole_10_val_dicts.pickle'), 
                             os.path.join(pickle_dir_16, 'svc_on_whole_10_test_dicts.pickle')]
    name_16 = [ 'svm (val)', 'svm (test)', ]
    name_dict['16'] = name_16
    pickle_file_group_dict['16'] = pickle_file_group_16
    color_dict['16'] = ['lavender', 'lavender']


    y_predicted_group = []
    y_label_group = []
    y_legend_group = [] 
    pickle_file_group = []
    color_group = []
                 
    keys = ['1', '11', '2', '9']
    keys = ['1', '10', '3', '4', '5', '6', '7']
    keys = ['1', '8', '13']
    keys = ['1', '11', '2', '9', '10', '3', '4', '5', '6', '7', '8', '13']
    keys = ['1', '10', '16', '15', '14', '3', '4', '5', '6', '7']
    keys = ['1', '16', '15', '14']
    keys = ['1', '10', '3', '4', '5', '6', '7']
    keys = ['1', '11', '2', '9', '10', '16', '15', '14', '3', '4', '5', '6', '7', '8', '13']


    for key in keys:
        y_legend_group += name_dict[key]
        pickle_file_group += pickle_file_group_dict[key]
        color_group += color_dict[key]
    data_len = len(keys) * 2
        

    linestyle_group = ['solid', 'dashed'] * data_len
    
    for i in range(len(pickle_file_group)):
        pickle_file_path = pickle_file_group[i]
        with open(pickle_file_path, 'rb') as fp:
            dict = pickle.load(fp)
        y_predicted_group.append(dict['preds'])
        y_label_group.append(dict['labels'])

    save_path = '/mnt/520_v2/lxy/BenignOrMaligantDiagnosis/deep_model/figs_saved/roc_curve_paper_final_merged_2.jpg'
    ks_with_random_num(y_predicted_group=y_predicted_group, y_label_group=y_label_group, 
                    y_legend_group=y_legend_group, color_group=color_group,  linestyle_group= linestyle_group, 
                    save_path=save_path)

                