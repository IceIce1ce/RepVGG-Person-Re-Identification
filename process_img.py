import os
from shutil import copyfile

### https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/prepare.py
query_path = 'data/Market-1501/query'
query_save_path = 'data/Market-1501/query_test'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

gallery_path = 'data/Market-1501/bounding_box_test'
gallery_save_path = 'data/Market-1501/test'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)
for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

train_path = 'data/Market-1501/bounding_box_train'
train_save_path = 'data/Market-1501/train'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)