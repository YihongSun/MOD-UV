import argparse

class MODUVOptions:
    def __init__(self):
        self.p = argparse.ArgumentParser(description="MOD-UV options")

        # Experiment options
        self.p.add_argument("--name", "-n",
							type=str,
							help="the name of the experiment",
                            default="example")
        self.p.add_argument("--round_name",
							type=str,
							help="the name of the self-training round",
                            default="R1")
        self.p.add_argument("--seed",
							type=int,
							help="random seed",
                            default=0)
        self.p.add_argument("--cuda_id",
							type=int,
							help="cuda id",
                            default=0)
        self.p.add_argument("--train_files",
							type=str,
							help="the path of the training files",
                            default="./moduv/splits/waymo/train_files.txt")
        self.p.add_argument("--data_dir", 
							type=str,
							help="data dir",
                            default="./data/waymo")
        self.p.add_argument("--save_dir", 
							type=str,
							help="save dir",
                            default="./save/")
        self.p.add_argument("--height",
							type=int,
							help="image height",
                            default=1280)
        self.p.add_argument("--width",
							type=int,
							help="image width",
                            default=1920)
		
        # Training options
        self.p.add_argument("--epochs",
							type=int,
							help="number of epochs",
                            default=20)
        self.p.add_argument("--anno_dir", 
							type=str,
							help="directory that stores training labels",
                            default=None)
        self.p.add_argument("--num_workers",
							type=int,
							help="number of dataloader workers",
                            default=2)
        self.p.add_argument('--epoch-size', 
							type=int, 
							help='manual epoch size (will match dataset size if 0)',
							default=8000)
        self.p.add_argument("--learning_rate",
							type=float,
							help="learning rate",
							default=1e-4)
        self.p.add_argument("--scheduler_step_size",
							type=int,
							help="step size of the scheduler",
							default=10)
        self.p.add_argument("--grad_agg",
							type=float,
							help="weight update aggregation (disable =1)",
							default=1)
        self.p.add_argument("--flip_aug_rate",
							type=float,
							help="threshold used to filter next round self-training labels",
							default=0.5)
        self.p.add_argument("--sjitter_rate",
							type=float,
							help="fraction of time to use scale jittering",
							default=1.0)
        self.p.add_argument("--sjitter_min",
							type=float,
							help="minimum jittering scale factor",
							default=0.5)
        self.p.add_argument("--sjitter_max",
							type=float,
							help="maximum jittering scale factor",
							default=1.0)
        self.p.add_argument("--log_freq",
							type=int,
							help="number of iterations in between print lines",
                            default=1000)
        self.p.add_argument("--save_freq",
							type=int,
							help="number of epochs in between model saves",
                            default=1)
        
        # Model options
        self.p.add_argument("--backbone_weights",
							type=str,
							help="specify the backbone weights",
							choices=["scratch", "moco", "imagenet"],
							default="moco")
        self.p.add_argument("--backbone_path",
							type=str,
							help="specify the moco weights path",
							default='./ckpts/moco_v2_800ep_pretrain.pth.tar')
        
        # Pseudo Labels options
        self.p.add_argument("--dynamo_dir", 
							type=str,
							help="directory that stores motion mask and depth outputs",
                            default="./save/dynamo_out-waymo/")
        self.p.add_argument("--m2m_ckpt",
							type=str,
							help="path to model trained after m2m",
                            default=None)
        self.p.add_argument("--m2m_conf_thrd", 
							type=float,
							help="confidence threshold used when computing pseudo-labels for next round self-training",
                            default=0.5)
        self.p.add_argument("--l2s_l_ckpt",
							type=str,
							help="path to large only model trained after l2s",
                            default=None)
        self.p.add_argument("--l2s_l_conf_thrd", 
							type=float,
							help="confidence threshold used when computing pseudo-labels for next round self-training",
                            default=0.9)
        self.p.add_argument("--l2s_s_ckpt",
							type=str,
							help="path to small only model trained after l2s",
                            default=None)
        self.p.add_argument("--l2s_s_conf_thrd", 
							type=float,
							help="confidence threshold used when computing pseudo-labels for next round self-training",
                            default=0.8)
        

        # Inference options
        self.p.add_argument("--load_ckpt", "-l",
							type=str,
							help="path to model to load",
                            default=None)
        self.p.add_argument("--input", 
							type=str,
							help="path to inference img or directory",
                            default=None)
        self.p.add_argument("--out", 
							type=str,
							help="directory to store inference outputs",
                            default="./out/")
        self.p.add_argument("--vis_conf_thrd", 
							type=float,
							help="confidence threshold used when visualizing _confident_ predictions",
                            default=0.5)

    
    def parse(self, **kwargs):
        self.options = self.p.parse_args(**kwargs)
        	
        return self.options