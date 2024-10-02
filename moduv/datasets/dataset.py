import random, json, sys
import os.path as osp
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import PIL.Image as pil
import pycocotools.mask as MaskUtil

from .ScaleJitter import ScaleJitter
from moduv.utils import read_img

proj_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, proj_dir)

def get_img_path(data_dir, segment_name, frame_idx):
    return osp.join(data_dir, segment_name, f'{frame_idx:06}.jpg')

class Dataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 filenames,
                 anno_dir,
                 height,
                 width,
                 is_train=False,
                 flip_freq=0.0,     # horizontal flip
                 sj_freq=0.0,       # scale jitter
                 sj_min=1.0,
                 sj_max=1.0,):
        super(Dataset, self).__init__()

        self.data_dir = data_dir
        self.filenames = filenames
        self.anno_dir = anno_dir
        self.height = height
        self.width = width
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()
        self.flip_freq = flip_freq
        self.sj_freq = sj_freq
        self.scale_jitter = ScaleJitter(output_size=[self.height, self.width], aug_scale_min=sj_min, aug_scale_max=sj_max) if self.sj_freq > 0 else None

        self.ss_thrd = 1        # short side threshold
        self.empty_mask = torch.zeros(0,self.height,self.width)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):        
        inputs = {}
        do_flip = self.is_train and random.random() < self.flip_freq

        line = self.filenames[index].split()
        segment_name = line[0]; frame_index = int(line[1])
        
        img = read_img(get_img_path(self.data_dir, segment_name, frame_index))
        if do_flip:
            img = img.transpose(pil.FLIP_LEFT_RIGHT)
        image = self.to_tensor(img)
        
        # Verify loaded image dimensions
        assert image.shape[-2] == self.height and image.shape[-1] == self.width
        
        target = {}
        if self.is_train:   # only fill targets if training
            
            json_path = osp.join(self.anno_dir, segment_name, '{:06d}.json'.format(frame_index))
            with open(json_path, 'r') as fh:
                anno = json.load(fh)

            masks = [torch.from_numpy(MaskUtil.decode(m)) for m in anno['masks']]
            masks = torch.stack(masks) if len(masks) > 0 else self.empty_mask 
            boxes = torch.tensor(anno['boxes']).float() if len(anno['boxes']) > 0 else torch.zeros(0,4).float()

            load_height = masks.shape[-2]
            load_width = masks.shape[-1]
            if load_height != self.height or load_width != self.width:
                masks = F.interpolate(masks[:,None], (self.height, self.width), mode='nearest')[:,0]

                boxes[:, 0] *= self.width / load_width
                boxes[:, 2] *= self.width / load_width
                boxes[:, 1] *= self.height / load_height
                boxes[:, 3] *= self.height / load_height
            
            if do_flip and boxes.size(0) > 0:
                masks = torch.flip(masks, [-1]) # flip left/right on last dim
                boxes[:, 0] = self.width - 1 - boxes[:, 0]
                boxes[:, 2] = self.width - 1 - boxes[:, 2]
                boxes = boxes[:, (2,1,0,3)]

            short_side = torch.min(torch.stack([boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1]]), 0).values
            valid = short_side > self.ss_thrd

            target['masks'] = masks[valid]
            target['boxes'] = boxes[valid]
            target['labels'] = torch.ones(target['boxes'].size(0)).type(torch.long) if 'labels' not in anno else torch.tensor(anno['labels']).type(torch.long) 

        if self.is_train and random.random() < self.sj_freq:
            image, target = self.scale_jitter(image, target)
        
        inputs["image"] = image
        inputs['target'] = target
        inputs['index'] = index
        
        return inputs
