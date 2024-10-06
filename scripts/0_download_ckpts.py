import os

ckpt_path = None
assert ckpt_path is not None, 'Please reach out for checkpoint access according to the README.'

os.system(f'gdown {ckpt_path}')
os.system('unzip ckpts.zip')
os.system('rm ckpts.zip')
