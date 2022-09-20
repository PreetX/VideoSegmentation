changes={
  "backbone.stem.conv1.norm":"backbone.stem.norm1",
  "backbone.res2.0.shortcut.norm":"backbone.res2.0.shortcut_norm",
  "backbone.res2.0.conv1.norm":"backbone.res2.0.norm1",
  "backbone.res2.0.conv2.norm":"backbone.res2.0.norm2",
  "backbone.res2.0.conv3.norm":"backbone.res2.0.norm3",
  "backbone.res2.1.conv1.norm":"backbone.res2.1.norm1",
  "backbone.res2.1.conv2.norm":"backbone.res2.1.norm2",
  "backbone.res2.1.conv3.norm":"backbone.res2.1.norm3",
  "backbone.res2.2.conv1.norm":"backbone.res2.2.norm1",
  "backbone.res2.2.conv2.norm":"backbone.res2.2.norm2",
  "backbone.res2.2.conv3.norm":"backbone.res2.2.norm3",
  "backbone.res3.0.shortcut.norm":"backbone.res3.0.shortcut_norm",
  "backbone.res3.0.conv1.norm":"backbone.res3.0.norm1",
  "backbone.res3.0.conv2.norm":"backbone.res3.0.norm2",
  "backbone.res3.0.conv3.norm":"backbone.res3.0.norm3",
  "backbone.res3.1.conv1.norm":"backbone.res3.1.norm1",
  "backbone.res3.1.conv2.norm":"backbone.res3.1.norm2",
  "backbone.res3.1.conv3.norm":"backbone.res3.1.norm3",
  "backbone.res3.2.conv1.norm":"backbone.res3.2.norm1",
  "backbone.res3.2.conv2.norm":"backbone.res3.2.norm2",
  "backbone.res3.2.conv3.norm":"backbone.res3.2.norm3",
  "backbone.res3.3.conv1.norm":"backbone.res3.3.norm1",
  "backbone.res3.3.conv2.norm":"backbone.res3.3.norm2",
  "backbone.res3.3.conv3.norm":"backbone.res3.3.norm3",
  "backbone.res4.0.shortcut.norm":"backbone.res4.0.shortcut_norm",
  "backbone.res4.0.conv1.norm":"backbone.res4.0.norm1",
  "backbone.res4.0.conv2.norm":"backbone.res4.0.norm2",
  "backbone.res4.0.conv3.norm":"backbone.res4.0.norm3",
  "backbone.res4.1.conv1.norm":"backbone.res4.1.norm1",
  "backbone.res4.1.conv2.norm":"backbone.res4.1.norm2",
  "backbone.res4.1.conv3.norm":"backbone.res4.1.norm3",
  "backbone.res4.2.conv1.norm":"backbone.res4.2.norm1",
  "backbone.res4.2.conv2.norm":"backbone.res4.2.norm2",
  "backbone.res4.2.conv3.norm":"backbone.res4.2.norm3",
  "backbone.res4.3.conv1.norm":"backbone.res4.3.norm1",
  "backbone.res4.3.conv2.norm":"backbone.res4.3.norm2",
  "backbone.res4.3.conv3.norm":"backbone.res4.3.norm3",
  "backbone.res4.4.conv1.norm":"backbone.res4.4.norm1",
  "backbone.res4.4.conv2.norm":"backbone.res4.4.norm2",
  "backbone.res4.4.conv3.norm":"backbone.res4.4.norm3",
  "backbone.res4.5.conv1.norm":"backbone.res4.5.norm1",
  "backbone.res4.5.conv2.norm":"backbone.res4.5.norm2",
  "backbone.res4.5.conv3.norm":"backbone.res4.5.norm3",
#   "roi_heads.res5.0.shortcut.norm":"roi_heads.res5.1.shortcut_norm",
#   "roi_heads.res5.0.conv1.norm":"roi_heads.res5.1.norm1",
#   "roi_heads.res5.0.conv2.norm":"roi_heads.res5.1.norm2",
#   "roi_heads.res5.0.conv3.norm":"roi_heads.res5.1.norm3",
#   "roi_heads.res5.1.conv1.norm":"roi_heads.res5.2.norm1",
#   "roi_heads.res5.1.conv2.norm":"roi_heads.res5.2.norm2",
#   "roi_heads.res5.1.conv3.norm":"roi_heads.res5.2.norm3",
#   "roi_heads.res5.2.conv1.norm":"roi_heads.res5.3.norm1",
#   "roi_heads.res5.2.conv2.norm":"roi_heads.res5.3.norm2",
#   "roi_heads.res5.2.conv3.norm":"roi_heads.res5.3.norm3",
#   "roi_heads.res5.0.shortcut":"roi_heads.res5.1.shortcut",
#   "roi_heads.res5.0.conv1":"roi_heads.res5.1.conv1",
#   "roi_heads.res5.0.conv2":"roi_heads.res5.1.conv2",
#   "roi_heads.res5.0.conv3":"roi_heads.res5.1.conv3",
#   "roi_heads.res5.1.conv1":"roi_heads.res5.2.conv1",
#   "roi_heads.res5.1.conv2":"roi_heads.res5.2.conv2",
#   "roi_heads.res5.1.conv3":"roi_heads.res5.2.conv3",
#   "roi_heads.res5.2.conv1":"roi_heads.res5.3.conv1",
#   "roi_heads.res5.2.conv2":"roi_heads.res5.3.conv2",
#   "roi_heads.res5.2.conv3":"roi_heads.res5.3.conv3",
}

import pickle

weights = pickle.load(open("D:/Samsung/Delta-Detectron2/faster-rcnn_resnet50.pkl", 'rb'))
keys = [k for k in weights['model']]
change=list()
for k in keys:
    if 'backbone' in k:
        if not len(weights['model'][k].shape)==4:
            continue
        print(weights['model'][k].shape)
        weights['model'][k] = weights['model'][k].transpose((1,2,3,0))
        change.append(k)
#     if 'roi_heads' in k:
#         if not len(weights['model'][k].shape)==4:
#             continue
#         weights['model'][k] = weights['model'][k].transpose((1,2,3,0))
print(change)
# change_keys = [k for k in changes]

# for k in change_keys:
#     value = changes[k]
#     weights['model'][value+'.weight'] = weights['model'][k+'.weight']
#     try:
#         weights['model'][value+'.bias'] = weights['model'][k+'.bias']
#         weights['model'][value+'.running_mean'] = weights['model'][k+'.running_mean']
#         weights['model'][value+'.running_var'] = weights['model'][k+'.running_var']
#     except:
#         # print(k+'.weight')
#         # print(weights['model'][k+'.weight'].shape)
#         continue

# for k in change_keys:
#     del weights['model'][k+'.weight']
#     try:
#         del weights['model'][k+'.bias']
#         del weights['model'][k+'.running_mean']
#         del weights['model'][k+'.running_var']
#     except:
#         # print(k)
#         continue

pickle.dump(weights, open('faster-rcnn_resnet50_channel_last.pkl', 'wb'))