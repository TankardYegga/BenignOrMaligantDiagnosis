from train_feats_collecting import *
import os

if __name__ == "__main__":

    save_path = global_var.base_data_aug_2_prefix  + '/filtered_features_10.csv'
    train_dir = global_var.base_data_aug_2_prefix + '/roi'
    collect_feats(train_dir, save_path)

    # save_path = global_var.base_data_aug_prefix  + '/filtered_features_10.csv'
    # train_dir = global_var.base_data_aug_prefix + '/roi'
    # collect_feats(train_dir, save_path)

    save_path = global_var.base_data_aug_3_prefix  + '/filtered_features_10.csv'
    train_dir = global_var.base_data_aug_3_prefix + '/roi'
    collect_feats(train_dir, save_path)


