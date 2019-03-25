import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='GCANet')
parser.add_argument('--task', default='dehaze', help='dehaze | derain')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--indir', default='valahazy/test1')
parser.add_argument('--outdir', default='t27/')
opt = parser.parse_args()
assert opt.task in ['dehaze', 'derain']
## forget to regress the residue for deraining by mistake,
## which should be able to produce better results


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

opt.only_residual =1
opt.model = 'ck2/last2_epoch_27.pth'
opt.use_cuda = 1
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
test_img_paths = make_dataset(opt.indir)

if opt.network == 'GCANet':
    from architecture import RRDB_Net
    net = RRDB_Net()
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

if opt.use_cuda:
    torch.cuda.set_device(opt.gpu_id)
    net.cuda()
else:
    net.float()

cc=torch.load(opt.model)
net.load_state_dict(cc['g_state_dict'])
net.eval()

for img_path in test_img_paths:
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    img_data = img_data/255.0
    #edge_data = edge_compute(img_data)
    in_data = img_data.unsqueeze(0)
    in_data =in_data.cuda() if opt.use_cuda else in_data.float()
    with torch.no_grad():
        pred = net(Variable(in_data))
    if opt.only_residual:
        out_img_data = (((pred.data[0].cpu().float()))*255).round().clamp(0, 255)
    else:
        out_img_data = pred.data[0].cpu().float().round().clamp(0, 255)
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '.png' ))
