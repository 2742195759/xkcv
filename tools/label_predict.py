# use the normal multilabel methods to predict the label and 
# check the accuracy. if the accuracy is high then ok

# TODO 0.19, try to find other way to get the true result

from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import MLARAM
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os


# copy from the loader function
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        image = Image.open(path)
        image = image.convert('RGB')
        return image

classifier = MLARAM()

dirpath = '../data/ava_dataset/'
imagepath = dirpath + 'images/'
style_dir_path = dirpath + 'style_image_lists/'
train_id = style_dir_path + 'train.jpgl'
train_tag = style_dir_path + 'train.lab'
test_id = style_dir_path + 'test.jpgl'
test_tags = style_dir_path + 'test.multilab'

resnet18 = models.resnet18(pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_transforms = transforms.Compose([
    transforms.Resize((600,600)),
#    transforms.RandomSizedCrop(600),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

if not os.path.exists('../cache/ava_input_np.npy'): #XXX target' order is the same, so not this problem
    data = datasets.ImageFolder('../data/ava_style/train/', image_transforms)
    print (data.class_to_idx)
    train_loader = torch.utils.data.DataLoader(
        data, 
        batch_size=2, shuffle=True,
        num_workers=2, pin_memory=True
    )
    resnet18.eval()
    inputs = []
    targets = []
    i = 0
    print ('[Load]')
    for batchx, batchy in train_loader: # the output size is (batch_size, 1000)
        inputs.append(resnet18(batchx).detach().numpy())
        targets.extend(batchy.numpy())

    input_np = np.concatenate(inputs, axis=0)
    target_np = np.array(targets, dtype=np.int64)
    np.save('../cache/ava_input_np.npy', input_np)
    np.save('../cache/ava_target_np.npy', target_np)

input_np = np.load('../cache/ava_input_np.npy')
target_np = np.load('../cache/ava_target_np.npy')
classifier.fit(input_np, np.eye(14)[target_np])

### after the train process, start the eval and predict

test_inputs = []
gts = []
# import pdb
# pdb.set_trace()

for iid, tag_line in zip(open(test_id).readlines(), open(test_tags).readlines()):
    iid, tag_line = iid.strip(), tag_line.strip()
    imagefile = imagepath + iid + '.jpg'
    image = image_transforms(default_loader(imagefile)).unsqueeze(0)
    image_np = resnet18(image).detach().numpy()

    test_inputs.append(image_np)
    gts.append(np.array([int(i) for i in tag_line.split(' ')]))
   # break

test_input_np = np.concatenate(test_inputs, axis=0)
gts_np = np.array(gts, dtype=np.int64)

#import pdb
#pdb.set_trace()

acc_metric = accuracy_score(gts_np, classifier.predict(test_input_np)) 
# metric for the function in multilabel classification function
preds_np = classifier.predict(test_input_np)
if not isinstance(preds_np, np.array) : 
    preds_np = preds_np.toarray()

hamm_metric = 0.0
jacc_metric = 0.0
macc_metric = 0.0
mrec_metric = 0.0
f1 = 0.0

for gt, pred in zip(gts_np, preds_np):
    gt_bool, pred_bool = gt.astype(np.bool), pred.astype(np.bool) 
    hamm_metric = hamm_metric + (gt_bool^pred_bool).sum()
    jacc_metric = jacc_metric + ((gt_bool & pred_bool).sum() * 1.0 / (gt_bool | pred_bool).sum() if (gt_bool | pred_bool).sum() > 0 else 1)
    single_acc = ((gt_bool & pred_bool).sum() * 1.0 / pred_bool.sum()) if (pred_bool).sum() > 0 else 1
    single_rec = ((gt_bool & pred_bool).sum() * 1.0 / gt_bool.sum()) if (gt_bool).sum() > 0 else 1
    macc_metric = macc_metric + single_acc
    mrec_metric = mrec_metric + single_rec
    f1 = f1 + 2.0 / (1.0 / single_acc + 1.0 / single_rec)

    print ("")
    print ("")
    print (gt, pred)

print ('[Metric] subset_accu: {h}'.format(h=acc_metric))
print ('[Metric] hamm:        {h}'.format(h=hamm_metric*1.0/(len(gts_np)*14)))
print ('[Metric] jacc:        {h}'.format(h=jacc_metric*1.0/len(gts_np)))
print ('[Metric] macc:        {h}'.format(h=macc_metric*1.0/len(gts_np)))
print ('[Metric] mrec:        {h}'.format(h=mrec_metric*1.0/len(gts_np)))
print ('[Metric] f1:          {h}'.format(h=f1*1.0/len(gts_np)))

import pdb
pdb.set_trace()
a = 1
