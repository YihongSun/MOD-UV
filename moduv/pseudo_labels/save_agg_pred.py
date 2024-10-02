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

def filter_mask(masks, ref_bin_mask, filt_thrd):
    overlap_size = torch.logical_and(masks, ref_bin_mask).sum((-1,-2))
    org_size = masks.sum((-2,-1))
    return overlap_size/org_size < filt_thrd

def compute_iou_matrix(Amasks, Bmasks):
    Asum = Amasks.sum((-2,-1))                                  # (A, H, W) -> (A)
    Bsum = Bmasks.sum((-2,-1))                                  # (B, H, W) -> (B)
    inter = (Amasks[:, None] * Bmasks[None, :]).sum((-2,-1))    # (A, H, W) x (B, H, W) -> (A, B)
    union = (Asum[:, None] + Bsum[None, :]) - inter             # (A, B)
    return inter / (union + 1e-6)

def cleanout_masks(masks, descending, filt_thrd):
    if len(masks.shape) == 4:
        masks = masks[:,0]
    mask_sizes = masks.sum((-2,-1))
    argsort = mask_sizes.argsort(descending=descending)
    keep_ind = torch.ones(masks.shape[0], dtype=torch.bool).to(masks.device)
    for i in range(masks.shape[0]):
        rest_ind = argsort[i+1:]
        rest_mask = torch.any(masks[rest_ind], 0)
        overlap_frac = torch.logical_and(masks[argsort[i]], rest_mask).sum((-1,-2)) / mask_sizes[argsort[i]]
        if overlap_frac > filt_thrd:
            keep_ind[argsort[i]] = False
    return keep_ind

def aggregate_masks(lout, sout, match_thrd=0.5, self_filt_thrd=0.75, cross_filt_thrd=0.5, mask_thrd=0.5):

    # Clean out Large outputs
    clean_ind = cleanout_masks(lout['masks'], descending=False, filt_thrd=self_filt_thrd)    # remove small masks if overlapped with larger ones 
    lout['masks'] = lout['masks'][clean_ind]
    lout['boxes'] = lout['boxes'][clean_ind]
    lout['scores'] = lout['scores'][clean_ind]

    # Clean out Small outputs
    clean_ind = cleanout_masks(sout['masks'], descending=True, filt_thrd=self_filt_thrd)    # remove large masks if overlapped with smaller ones 
    sout['masks'] = sout['masks'][clean_ind]
    sout['boxes'] = sout['boxes'][clean_ind]
    sout['scores'] = sout['scores'][clean_ind]
    
    # Fix shape if necessary
    lout['masks'] = lout['masks'][:,None] if len(lout['masks'].shape) == 3 else lout['masks']
    sout['masks'] = sout['masks'][:,None] if len(sout['masks'].shape) == 3 else sout['masks']
            
    large_masks = lout['masks'][:,0] > mask_thrd
    small_masks = sout['masks'][:,0] > mask_thrd
    
    remain_l_ind = torch.ones_like(lout['scores'], dtype=torch.bool)
    remain_s_ind = torch.ones_like(sout['scores'], dtype=torch.bool)
    overlap_l_ind = torch.zeros_like(lout['scores'], dtype=torch.bool)
    overlap_s_ind = torch.zeros_like(sout['scores'], dtype=torch.bool)
    
    # first find overlapping predictions
    if large_masks.size(0) > 0 and small_masks.size(0) > 0:
        
        iou_mat = compute_iou_matrix(large_masks, small_masks)

        l_idx, s_idx = torch.where(iou_mat == iou_mat.max()); l_idx = l_idx[0]; s_idx = s_idx[0]        
        while iou_mat[l_idx, s_idx] > match_thrd:
            iou_mat[l_idx, s_idx] = 0
            
            # choose which mask is better
            if lout['scores'][l_idx] >= sout['scores'][s_idx]:
                overlap_l_ind[l_idx] = True
            else:
                overlap_s_ind[s_idx] = True

            # matched so no longer need to processed further
            remain_l_ind[l_idx] = False
            remain_s_ind[s_idx] = False
            
            l_idx, s_idx = torch.where(iou_mat == iou_mat.max()); l_idx = l_idx[0]; s_idx = s_idx[0]
    
    # Filter remaining large masks with all small masks
    filt_ind = filter_mask(large_masks[remain_l_ind], ref_bin_mask=torch.any(small_masks,0), filt_thrd=cross_filt_thrd)
    remain_l_ind[[torch.where(remain_l_ind)[0][~filt_ind]]] = False
    
    # Filter remaining small masks with all remaining large masks 
    filt_ind = filter_mask(small_masks[remain_s_ind], ref_bin_mask=torch.any(torch.cat((large_masks[overlap_l_ind], small_masks[overlap_s_ind], large_masks[remain_l_ind])),0), filt_thrd=cross_filt_thrd)
    remain_s_ind[[torch.where(remain_s_ind)[0][~filt_ind]]] = False
      
    boxes = torch.cat([lout['boxes'][overlap_l_ind], sout['boxes'][overlap_s_ind], lout['boxes'][remain_l_ind], sout['boxes'][remain_s_ind]]).cpu().numpy()       #(n, 4)
    masks = torch.cat([lout['masks'][overlap_l_ind], sout['masks'][overlap_s_ind], lout['masks'][remain_l_ind], sout['masks'][remain_s_ind]]).cpu()               #(n, h, w)
    scores = torch.cat([lout['scores'][overlap_l_ind], sout['scores'][overlap_s_ind], lout['scores'][remain_l_ind], sout['scores'][remain_s_ind]]).cpu().numpy()  #(n)
    
    enc_masks = [dict(MaskUtil.encode(np.asfortranarray((m[0]>mask_thrd).type(torch.uint8).numpy()))) for m in masks] # from torch
    for em in enc_masks:
        em['counts'] = em['counts'].decode()

    return {'masks' : enc_masks, 'boxes': boxes.tolist(), 'scores': scores.tolist()}


''' python3 moduv/pseudo_labels/save_agg_pred.py --train_files ./moduv/splits/waymo/example_train_files.txt --l2s_l_ckpt ./ckpts/moduv_l2s_l.pth --l2s_s_ckpt ./ckpts/moduv_l2s_s.pth
'''
if __name__ == "__main__":

    options = MODUVOptions()
    opt = options.parse()

    MATCH_IOU_THRD = 0.5
    SELF_FILT_THRD = 0.75
    CROSS_FILT_THRD = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_L = Model(backbone_weights='moco').eval().to(device)
    model_L.load(opt.l2s_l_ckpt)
    model_L.set_score_thrd(opt.l2s_l_conf_thrd)

    model_S = Model(backbone_weights='moco').eval().to(device)
    model_S.load(opt.l2s_s_ckpt)
    model_S.set_score_thrd(opt.l2s_s_conf_thrd)

    out_dir = osp.join(opt.save_dir, opt.name, 'L2_pseudo_labels')
    print(f'Saving to {out_dir} ...')
    for train_instance in tqdm(readlines(opt.train_files)):
        segment_name, frame_idx = train_instance.split(); frame_idx = int(frame_idx)
        
        img_path = get_img_path(opt.data_dir, segment_name, frame_idx)
        with torch.no_grad():
            output_L = model_L(read_img(img_path))[0][0]
            output_S = model_S(read_img(img_path))[0][0]

        L2 = aggregate_masks(output_L, output_S, match_thrd=MATCH_IOU_THRD, self_filt_thrd=SELF_FILT_THRD, cross_filt_thrd=CROSS_FILT_THRD)

        out_path = osp_join(out_dir, segment_name, f'{frame_idx:06}.json')
        with open(out_path, 'w') as fh:
            json.dump(L2, fh)
