import os
import shutil

def move(original_dir, ref_dir, moved_dir):
    for img in os.listdir(ref_dir):
        ori_path = os.path.join(original_dir, img)
        moved_to_path = os.path.join(moved_dir, img)
        shutil.copy2(ori_path, moved_to_path)


if __name__ == '__main__':
    # original_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected/roi'
    # moved_to_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi2'
    # ref_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    # move(original_dir, ref_dir, moved_to_dir)

    # original_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected/topo_mask'
    # moved_to_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/topo_mask2'
    # ref_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/topo_mask'
    # move(original_dir, ref_dir, moved_to_dir)

    original_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected/roi'
    moved_to_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi2'
    ref_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data/roi'
    move(original_dir, ref_dir, moved_to_dir)
