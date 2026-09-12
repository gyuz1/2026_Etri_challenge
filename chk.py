import mmcv, sys

p = 'projects/configs/VAD/VAD_etri_tiny_stage1_best_nolcf.py'
c = mmcv.Config.fromfile(p)
print('fromfile train ann:', c.data.train.ann_file)
print('fromfile val   ann:', c.data.val.ann_file)
print('fromfile test  ann:', c.data.test.ann_file)
print('fromfile data_root:', c.data.train.get('data_root'))

dumped = 'work_dirs/stage1_best_nolcf/VAD_etri_tiny_stage1_best_nolcf.py'
a = c.pretty_text.splitlines()
b = open(dumped).read().splitlines()
import difflib
d = [l for l in difflib.unified_diff(a, b, 'fromfile', 'dumped', lineterm='', n=1)]
print('=== diff lines:', len(d))
for l in d[:60]:
    print(l)
