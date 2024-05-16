import os
import json
from PIL import Image
import numpy as np
import random
from shutil import copy2, move

ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, "mapillary_dataset")
ORIGINAL_20000_IMG = os.path.join(DATASET_DIR, "original_20000", "images")
ORIGINAL_20000_INS = os.path.join(DATASET_DIR, "original_20000", "instances")

DS_STORE = '.DS_Store'

intersection_class_ids = [0, 19, 20, 21, 33, 38, 48, 54, 55, 57, 61]
instance_class_ids = [0, 1, 8, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]

# write a list of files to .txt
def Write_TXT(file_names, dst, txt_name):
    if not os.path.exists(dst):
            os.makedirs(dst)
    with open(os.path.join(dst, txt_name), 'w') as the_file:
        for file_name in file_names:
            the_file.write(file_name.rsplit(".", 1)[0])
            the_file.write('\n')

def Read_TXT(txt_path):
    with open(txt_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

# Python code t get difference of two lists
# Using set()
def Diff(li1, li2):
    return (list(set(li1) - set(li2)))

# Copy a list of files to a desired directory
def Copy_Files(dataset_name, subset_name, file_names):
    for file_name in file_names:
        src = os.path.join(ORIGINAL_20000_IMG, file_name + ".jpg")
        dst = os.path.join(DATASET_DIR, dataset_name, subset_name, "images")
        if not os.path.exists(dst):
            os.makedirs(dst)
        copy2(src, dst)

        ins_name = file_name.rsplit(".", 1)[0] + ".png"
        src = os.path.join(ORIGINAL_20000_INS, ins_name)
        dst = os.path.join(DATASET_DIR, dataset_name, subset_name, "instances")
        if not os.path.exists(dst):
            os.makedirs(dst)
        copy2(src, dst)

def Read_Dataset(txt_path):
    if os.path.exists(txt_path):
        all_files = Read_TXT(txt_path)
    else:
        IMG_DIR = os.path.join(txt_path.rsplit(".", 1)[0], "images")
        print(IMG_DIR)
        all_files = next(os.walk(IMG_DIR))[2]
        if DS_STORE in all_files:
            all_files.remove(DS_STORE)
        Write_TXT(all_files, IMG_DIR, txt_path)
    return all_files

# Select desired number of examples whose ground truth masks are non-trivial (defined by threshold)
# this function was modified from dataset_clean.ipynb
def Select_Examples(available_files, num_files_needed, threshold, class_ids):
    accepted = []
    rejected = []
    idx = 0
    while len(accepted) < num_files_needed:
        if idx >= len(available_files):
            print("not enough avaialble files!")
            break
            
        file_name = available_files[idx]
        if file_name != '.DS_Store':
            IMG_PATH = os.path.join(ORIGINAL_20000_IMG, file_name)
            ins_name = file_name.rsplit(".", 1)[0] + ".png"
            INS_PATH = os.path.join(ORIGINAL_20000_INS, ins_name)
            instance_image = Image.open(INS_PATH)
            
            # convert labeled data to numpy arrays for better handling
            instance_array = np.array(instance_image, dtype=np.uint16)

            instances = np.unique(instance_array)
            instaces_count = instances.shape[0]

            label_ids = instances // 256
            label_id_count = np.unique(label_ids).shape[0]

            mask_count = 0
            for instance in instances:
                label_id = instance // 256
                if label_id in class_ids:
                    m = np.zeros((instance_array.shape[0], instance_array.shape[1]), dtype=np.uint8)
                    m[instance_array == instance] = 1
                    m_size = np.count_nonzero(m == 1)

                    # only load mask greater than threshold size, 
                    # otherwise bounding box with area zero causes program to crash
                    if m_size > threshold:
                        mask_count = mask_count + 1
            if mask_count == 0:
                rejected.append(file_name)
            else:
                accepted.append(file_name)
        idx = idx + 1
        print('Accepted {}/{}, rejected {}\r'.format(len(accepted), num_files_needed, len(rejected)), end='', )
        

        with open(os.path.join(DATASET_DIR, "progress.txt"), 'w') as the_file:
            the_file.write('Accepted {}/{}, rejected {}\r'.format(len(accepted), num_files_needed, len(rejected)))
            the_file.write('\n')
            
    return accepted, rejected

def Build_Dataset(available_files, dataset_name, subset_name, threshold, size, rebuild = False, class_ids = intersection_class_ids):
    accepted = []
    rejected = []

    subset_dir = os.path.join(DATASET_DIR, dataset_name, subset_name)
    accepted_txt = os.path.join(subset_dir, "accepted.txt")
    rejected_txt = os.path.join(subset_dir, "rejected.txt")

    if rebuild:
        accepted, rejected = Select_Examples(available_files, size, threshold, class_ids)

        Write_TXT(accepted, subset_dir, "accepted.txt")
        Write_TXT(rejected, subset_dir, "rejected.txt")

        Copy_Files(subset_dir, "accepted", accepted)
        Copy_Files(subset_dir, "rejected", rejected)
    else:
        accepted = Read_Dataset(accepted_txt)
        rejected = Read_Dataset(rejected_txt)
    
    used = accepted + rejected
    available_files = Diff(available_files, used)

    print('{} accepted {} and rejected {}'.format(dataset_name, len(accepted), len(rejected)))
    print('{} images still available'.format(len(available_files)))
    
    return available_files, accepted, rejected

txt_20k = os.path.join(DATASET_DIR, "original_20000.txt")
all_files = Read_Dataset(txt_20k)
print('{} images were found in the mapillary dataset'.format(len(all_files)))

AWS_RUN_1 = os.path.join(DATASET_DIR, "AWS_run_1")

txt_path = os.path.join(DATASET_DIR, "AWS_run_1", "train_4096.txt")
train_file_names = Read_Dataset(txt_path)

txt_path = os.path.join(DATASET_DIR, "AWS_run_1", "dev_512.txt")
dev_file_names = Read_Dataset(txt_path)

used_files = list(set(train_file_names + dev_file_names))
    
print('{} images were used in AWS run 1 (train + dev)'.format(len(used_files)))

# take difference between two lists
available_files = Diff(all_files, used_files)
print('{} images still available'.format(len(available_files)))

# make sure that the filenames have a fixed order before shuffling
available_files.sort()  

# fix the random seed
random.seed(0)

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(available_files) 

available_files, AWS_run_2_train, AWS_run_2_train_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_2", 
                                                                           subset_name = "train_4096",
                                                                           threshold = 32 * 32, 
                                                                           size = 4096, 
                                                                           rebuild = False)

available_files, AWS_run_2_dev, AWS_run_2_dev_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_2", 
                                                                           subset_name = "dev_512",
                                                                           threshold = 32 * 32, 
                                                                           size = 512, 
                                                                           rebuild = False)

txt_20k = os.path.join(DATASET_DIR, "original_20000.txt")
all_files = Read_Dataset(txt_20k)
print('{} images were found in the mapillary dataset'.format(len(all_files)))

# make sure that the filenames have a fixed order before shuffling
all_files.sort()  

# fix the random seed
random.seed(230)

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(all_files) 

available_files, AWS_run_3_dev, AWS_run_3_dev_rejected = Build_Dataset(available_files = all_files,
                                                                           dataset_name = "AWS_run_3", 
                                                                           subset_name = "dev_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False)

available_files, AWS_run_3_test, AWS_run_3_test_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_3", 
                                                                           subset_name = "test_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False)

available_files, AWS_run_3_train, AWS_run_3_train_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_3", 
                                                                           subset_name = "train_16384",
                                                                           threshold = 32 * 32, 
                                                                           size = 16384, 
                                                                           rebuild = False)

txt_20k = os.path.join(DATASET_DIR, "original_20000.txt")
all_files = Read_Dataset(txt_20k)
print('{} images were found in the mapillary dataset'.format(len(all_files)))

# make sure that the filenames have a fixed order before shuffling
all_files.sort()  

# fix the random seed
random.seed(230)

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(all_files) 

available_files, AWS_run_4_dev, AWS_run_4_dev_rejected = Build_Dataset(available_files = all_files,
                                                                           dataset_name = "AWS_run_4", 
                                                                           subset_name = "dev_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False,
                                                                           class_ids = instance_class_ids)

available_files, AWS_run_4_test, AWS_run_4_test_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_4", 
                                                                           subset_name = "test_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False,
                                                                           class_ids = instance_class_ids)

available_files, AWS_run_4_train, AWS_run_4_train_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_4", 
                                                                           subset_name = "train_16384",
                                                                           threshold = 32 * 32, 
                                                                           size = 16384, 
                                                                           rebuild = False,
                                                                           class_ids = instance_class_ids)

dataset_name = "AWS_run_4"

# copy files from original_20000 if messed up
# Copy_Files(os.path.join(DATASET_DIR, dataset_name, "train_16384"), "accepted", AWS_run_4_train)

# divide images into 8 folders
files_to_move = AWS_run_4_train
part_size = 2048
directory = os.path.join(DATASET_DIR, dataset_name, "train_16384", "accepted", "images")
num_parts = int(len(files_to_move) / part_size)

num_files = next(os.walk(directory))[1]
if (len(num_files) == len(files_to_move)):
    for i in range(num_parts):
        part = files_to_move[:part_size]
        part_dir = os.path.join(directory, "part_" + str(i))
        if not os.path.exists(part_dir):
            os.makedirs(part_dir)
        for file in part:
            src = os.path.join(directory, file + '.jpg')
            move(src, part_dir)
        files_to_move = Diff(files_to_move, part)

# check the number of images in each folder
directory = os.path.join(DATASET_DIR, dataset_name, "train_16384", "accepted", "images")
folders = next(os.walk(directory))[1]
folders.sort()  
for folder in folders:
    part_dir = os.path.join(directory, folder)
    files = next(os.walk(part_dir))[2]
    if DS_STORE in files:
        files.remove(DS_STORE)
    print("num_files in {}: {}".format(folder, len(files)))

# # generate datasets sequentially withut rejecting anything (not recommended)
# TRAIN_BATCH_SIZE = 4096
# DEV_BATCH_SIZE = 512
# TEST_SET_SIZE = len(available_files) - (TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 3

# start = 0
# AWS_run_2_train = available_files[:TRAIN_BATCH_SIZE]
# AWS_run_2_dev   = available_files[TRAIN_BATCH_SIZE : TRAIN_BATCH_SIZE + DEV_BATCH_SIZE]

# AWS_run_3_train = available_files[TRAIN_BATCH_SIZE + DEV_BATCH_SIZE : TRAIN_BATCH_SIZE * 2 + DEV_BATCH_SIZE]
# AWS_run_3_dev   = available_files[TRAIN_BATCH_SIZE * 2 + DEV_BATCH_SIZE : (TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 2]


# AWS_run_4_train = available_files[(TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 2 : TRAIN_BATCH_SIZE * 3 + DEV_BATCH_SIZE * 2]
# AWS_run_4_dev   = available_files[TRAIN_BATCH_SIZE * 3 + DEV_BATCH_SIZE *2 : (TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 3]

# assert(len(AWS_run_2_train) == 4096)
# assert(len(AWS_run_3_train) == 4096)
# assert(len(AWS_run_4_train) == 4096)

# assert(len(AWS_run_2_dev) == 512)
# assert(len(AWS_run_3_dev) == 512)
# assert(len(AWS_run_4_dev) == 512)

# test_set = available_files[(TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 3 :]
# print("{} images in the test set".format(len(test_set)))



