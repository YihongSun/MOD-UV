import os

pseudo_label_path = None
assert pseudo_label_path is not None, 'Please reach out for pseudo-labels access according to the README.'

os.system(f'gdown {pseudo_label_path}')
os.system('mkdir save/example')
os.system('mkdir save/example/L0_pseudo_labels')
os.system('unzip init_pseudo_labels.zip -d save/example/L0_pseudo_labels/')
os.system('rm init_pseudo_labels.zip')
