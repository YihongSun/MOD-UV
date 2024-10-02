import json, sys
import os.path as osp

proj_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, proj_dir)

import numpy as np
import torch
from tqdm import tqdm

from moduv.model import Model
from moduv.datasets.dataset import get_img_path
from moduv.options import MODUVOptions
from moduv.utils import readlines, osp_join, read_img
import pycocotools.mask as MaskUtil


def get_pseudo_labels(output, mask_thrd=0.5):
    """ Compute model predictions for a given image, assuming score threshold has been set accordingly
    """
    boxes = output['boxes'].cpu().numpy()       #(n, 4)
    masks = output['masks'].cpu()               #(n, 1, h, w)
    scores = output['scores'].cpu().numpy()     #(n)

    enc_masks = [dict(MaskUtil.encode(np.asfortranarray((m[0]>mask_thrd).type(torch.uint8).numpy()))) for m in masks] # from torch
    for em in enc_masks:
        em['counts'] = em['counts'].decode()
    return {'masks' : enc_masks, 'boxes': boxes.tolist(), 'scores': scores.tolist()}

''' python3 moduv/pseudo_labels/save_pred.py --train_files ./moduv/splits/waymo/example_train_files.txt --m2m_ckpt ./ckpts/moduv_m2m.pth
'''
if __name__ == "__main__":

    options = MODUVOptions()
    opt = options.parse()

    device = torch.device(f"cuda:{opt.cuda_id}" if torch.cuda.is_available() else "cpu")
    model = Model(backbone_weights='moco').eval().to(device)
    model.load(opt.m2m_ckpt)
    model.set_score_thrd(opt.m2m_conf_thrd)

    out_dir = osp.join(opt.save_dir, opt.name, 'L1_pseudo_labels')
    print(f'Saving to {out_dir} ...')
    for train_instance in tqdm(readlines(opt.train_files)):
        segment_name, frame_idx = train_instance.split(); frame_idx = int(frame_idx)

        img_path = get_img_path(opt.data_dir, segment_name, frame_idx)
        with torch.no_grad():
            output = model(read_img(img_path))[0][0]
        L1 = get_pseudo_labels(output)

        out_path = osp_join(out_dir, segment_name, f'{frame_idx:06}.json')
        with open(out_path, 'w') as fh:
            json.dump(L1, fh)
