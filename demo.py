import os
import os.path as osp
import imageio.v3 as iio
import torch
from tqdm import tqdm

from moduv.options import MODUVOptions
from moduv.utils import read_img, visualize_ann
from moduv.model import Model

def get_img_fnames(dir):
    """ Find all images (.jpg or .png) under a directory
    """
    all_files = []
    for root, directories, files in os.walk(dir):
        all_files += [osp.join(root, f) for f in files if f.endswith('.jpg') or f.endswith('.png')]
    return all_files


''' python3 demo.py --load_ckpt ./ckpts/moduv_final.pth --vis_conf_thrd 0.5 --input ./data/
'''
if __name__ == "__main__":

    # Load Options
    options = MODUVOptions()
    opt = options.parse()

    if opt.load_ckpt is None:
        raise Exception('Error: model checkpoint path is not specified.')
    if opt.input is None:
        raise Exception('Error: inference input is not specified.')
    if not osp.exists(opt.out):
        os.makedirs(opt.out)

    # Load Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(backbone_weights='scratch').eval().to(device)
    model.load(opt.load_ckpt)

    # Get inference paths
    img_paths = get_img_fnames(opt.input) if osp.isdir(opt.input) else [opt.input]
    
    print('Using Confidence Thrd:', opt.vis_conf_thrd)
    print('# Inference Images:   ', len(img_paths))
    print('Using Device:         ', device)
    
    # Run inference
    for img_path in tqdm(img_paths):
        output = model(read_img(img_path))[0]
        vis = visualize_ann(iio.imread(img_path), output[0], m_conf_thrd=opt.vis_conf_thrd, bool_words=False)
        iio.imwrite(osp.join(opt.out, img_path.strip('/').replace('/', '-')), vis)
