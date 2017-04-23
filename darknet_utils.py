
import os
import numpy as np
dst_dir = '/home/maddoxw/temp/cowc'
anno_dir = os.path.join(dst_dir, 'annotations')

img_files = []
for path, _, files in os.walk(anno_dir):
    for file in files:
        if file.endswith('txt'):
            continue
        img_files.append(os.path.join(path, file))

np.random.shuffle(img_files)

n_splits = 5
n_split = 1
n_files = len(img_files)
n_test = n_files / n_splits + 1
i_files = n_test
test_dict = {i+1: [] for i in range(n_splits)}
for ii, img_file in enumerate(img_files):
    if ii >= i_files:
        n_split += 1
        i_files += n_test
    test_dict[n_split].append(img_file)

for n_split, test_list in test_dict.iteritems():
    with open(os.path.join(dst_dir, 'test' + str(n_split) + '.txt'), 'w') as ofs:
        for test_file in test_list:
            ofs.write(test_file + '\n')
