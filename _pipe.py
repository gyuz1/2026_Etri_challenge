import mmcv, sys, json
train_cfg = mmcv.Config.fromfile(sys.argv[1])
eval_cfg  = mmcv.Config.fromfile(sys.argv[2])

def pipe(cfg, key):
    d = cfg.data[key]
    return d.get('pipeline', [])

def norm(p):
    out = []
    for t in p:
        t = dict(t)
        ty = t.pop('type')
        out.append((ty, t))
    return out

tr = norm(pipe(train_cfg, 'train'))
ev = norm(pipe(eval_cfg, 'test'))
print("학습 train 파이프라인 (%d단계):" % len(tr))
for ty,_ in tr: print("   ", ty)
print("\n평가 test 파이프라인 (%d단계):" % len(ev))
for ty,_ in ev: print("   ", ty)

# 이미지 기하/정규화에 영향 주는 단계만 뽑아 인자까지 비교
KEY = ('UndistortMultiViewImage','CropMultiViewImage','RandomScaleImageMultiViewImage',
       'NormalizeMultiviewImage','PadMultiViewImage','ResizeMultiview3D',
       'LoadETRIGeometryCache','PhotoMetricDistortionMultiViewImage')
def geo(p): return {ty: args for ty,args in p if ty in KEY}
g_tr, g_ev = geo(tr), geo(ev)
print("\n=== 기하/정규화 단계 비교 ===")
for k in sorted(set(g_tr) | set(g_ev)):
    a, b = g_tr.get(k), g_ev.get(k)
    if a is None: print("  [평가에만] %s %s" % (k, b)); continue
    if b is None: print("  [학습에만] %s %s" % (k, a)); continue
    if a != b:
        print("  [인자 다름] %s" % k)
        for kk in sorted(set(a)|set(b)):
            if a.get(kk) != b.get(kk):
                print("       %s: train=%r  eval=%r" % (kk, a.get(kk), b.get(kk)))
    else:
        print("  [동일] %s" % k)
