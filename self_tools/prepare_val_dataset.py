import os
import glob
import xml.etree.cElementTree as ET
import shutil

anno_root = '/opt/tiger/minist/datasets/ILSVRC/Annotations/CLS-LOC'
img_root = '/opt/tiger/minist/datasets/ILSVRC/Data/CLS-LOC'
split = 'val'

anno_folder = os.path.join(anno_root, split)
img_folder = os.path.join(img_root, split)

anno_file_paths = glob.glob(anno_folder + '/*')

print(len(anno_file_paths))

count = 0
for anno in anno_file_paths:
    tree = ET.ElementTree(file=anno)
    root = tree.getroot()
    file_name = root[1].text
    objs = root.findall('object')
    max_area = 0
    class_name = None
    for obj in root.findall('object'):
        class_name_t = obj.find('name').text
        xmin = int(obj[4][0].text)
        ymin = int(obj[4][1].text)
        xmax = int(obj[4][2].text)
        ymax = int(obj[4][3].text)
        area = (ymax - ymin) * (xmax - xmin)
        if area > max_area:
            max_area = area
            class_name = class_name_t
    # print(class_name, max_area)
    class_folder = os.path.join(img_folder, class_name)
    if not os.path.exists(class_folder) :
        os.makedirs(class_folder)
    # print(os.path.join(img_folder, file_name+'.JPEG'), class_folder)
    shutil.copyfile(os.path.join(img_folder, file_name+'.JPEG'), os.path.join(class_folder, file_name+'.JPEG'))