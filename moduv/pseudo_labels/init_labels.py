import json, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN  
from scipy.sparse import coo_matrix
import pycocotools.mask as MaskUtil
from tqdm import tqdm
import warnings
from sklearn.exceptions import EfficiencyWarning
warnings.simplefilter(action='ignore', category=EfficiencyWarning)

proj_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, proj_dir)

from moduv.options import MODUVOptions
from moduv.utils import readlines, osp_join



class BackprojectDepth:
    """ Transform a depth image into a point cloud via input camera intrinsics
    """
    def __init__(self, height, width):
        self.batch_size = 1
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        B = depth.size(0)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords[:B])
        cam_points = depth.view(B, 1, -1) * cam_points[:B]
        cam_points = torch.cat([cam_points, self.ones[:B]], 1)

        return cam_points

class MaskExtractor:
    """ Extract instance level mask from motion segmentation and monocular depth predictions
    """
    def __init__(self, nbh_size, mot_thrd, eps, minsamp, height, width):
        self.nbh_size = nbh_size
        self.mot_thrd = mot_thrd
        self.eps = eps
        self.minsamp = minsamp
        self.height = height
        self.width = width
        self.pointcloud = BackprojectDepth(height, width)
    
    def info(self,):
        return '_'.join([
            f'n={self.nbh_size}',
            f't={self.mot_thrd}',
            f'e={self.eps}',
            f'r={self.minsamp}',
        ])
        
    def get_nbh(self, org_ind=None):

        h = self.height; w = self.width; n = self.nbh_size

        if org_ind is None:
            org_x = np.concatenate([np.arange(0, w) for y in range(h)])
            org_y = np.arange(0, h).repeat(w)
            org_ind = np.stack((org_y, org_x)).transpose((1, 0)) # (h*w, 2)

        n_coord = np.array([[x, y] for x in range(-n//2, n-n//2) for y in range(-n//2, n-n//2)]) # (n^2, 2)
        
        src_ind = np.repeat(org_ind, n_coord.shape[0], axis=0)
        tgt_ind = (org_ind.reshape(-1, 1, 2) + n_coord.reshape(1, -1, 2)).reshape(-1, 2) # (h*w*n^2, 2)

        valid = np.logical_and(np.logical_and(tgt_ind[:,0] >= 0, tgt_ind[:,0] < h), np.logical_and(tgt_ind[:,1] >= 0, tgt_ind[:,1] < w))

        src_ind = np.array([a[0]*w+a[1] for a in src_ind[valid]])
        tgt_ind = np.array([a[0]*w+a[1] for a in tgt_ind[valid]])

        return src_ind, tgt_ind
    
    def extract(self, motion_mask, depth, inv_K):
        h = self.height; w = self.width

        org_ind = torch.stack(torch.where(motion_mask.squeeze()>self.mot_thrd)).permute(1,0).cpu().numpy()
        src_ind, tgt_ind = self.get_nbh(org_ind=org_ind)

        motion_mask_feature = motion_mask.reshape(h*w)
        tgt_ind_valid = (motion_mask_feature[tgt_ind] > self.mot_thrd).cpu()
        src_ind = src_ind[tgt_ind_valid]
        tgt_ind = tgt_ind[tgt_ind_valid]
        
        cam_points = self.pointcloud.forward(depth, inv_K)
        proj_3d = cam_points.reshape(1, 4, h, w)[:,:3].permute(0,2,3,1).reshape(-1, 3)
        cam_d_3d = torch.sqrt(torch.sum((proj_3d)**2,1))
        dist = torch.abs(cam_d_3d[src_ind] - cam_d_3d[tgt_ind]).cpu().numpy()

        sort_ind = np.argsort(dist, kind='mergesort')
        d = dist[sort_ind]
        x = src_ind[sort_ind]
        y = tgt_ind[sort_ind]
        X = coo_matrix((d, (x, y)), shape=[h*w, h*w])

        clusters = DBSCAN(eps=self.eps, 
                        min_samples=int(self.minsamp), 
                        metric='precomputed', 
                        algorithm='auto', 
                        n_jobs=None).fit(X)

        out = {'masks' : list(), 'boxes' : list()}
        for obj_idx in range(0, clusters.labels_.max()+1):
            obj_mask = (clusters.labels_ == obj_idx).reshape(h, w).astype(np.uint8)
            if not np.any(obj_mask>0):
                continue    
            
            y, x = np.where(obj_mask != 0)
            top, bottom, left, right = int(np.min(y)), int(np.max(y)), int(np.min(x)), int(np.max(x))
            height, width = bottom-top, right-left
            obj_bbox, bh, bw = [left, top, right, bottom], height, width

            if bh <= 5 or bw <= 5:
                continue
        
            enc_masks = dict(MaskUtil.encode(np.asfortranarray(obj_mask)))
            enc_masks['counts'] = enc_masks['counts'].decode()

            out['masks'].append(enc_masks)
            out['boxes'].append(obj_bbox)

        return out

''' python3 moduv/pseudo_labels/init_labels.py --train_files ./moduv/splits/waymo/example_train_files.txt --dynamo_dir ./save/dynamo_out-waymo/
'''
if __name__ == "__main__":

    options = MODUVOptions()
    opt = options.parse()

    out_dir = osp.join(opt.save_dir, opt.name, 'L0_pseudo_labels')
    print(f'Saving to {out_dir} ...')
    for train_instance in tqdm(readlines(opt.train_files)):
        segment_name, frame_idx = train_instance.split(); frame_idx = int(frame_idx)
        out_path = osp_join(out_dir, segment_name, f'{frame_idx:06}.json')
        if osp.exists(out_path):
            continue

        dynamo_load_path = osp.join(opt.dynamo_dir, segment_name, f'{frame_idx:06}.npz')
        dynamo_out = np.load(dynamo_load_path)
        mask =  dynamo_out['mot_mask']  # (320, 480)
        depth = dynamo_out['depth']     # (320, 480)
        inv_K = dynamo_out['inv_K']     # (4, 4)

        height, width = mask.shape
        mask_extractor = MaskExtractor(nbh_size=10, mot_thrd=0.1, eps=0.03, minsamp=100, height=height, width=width)
        L0 = mask_extractor.extract(torch.from_numpy(mask), torch.from_numpy(depth)[None, None], torch.from_numpy(inv_K)[None])

        with open(out_path, 'w') as fh:
            json.dump(L0, fh)
        

