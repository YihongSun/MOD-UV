import sys, json, os, time, random
import os.path as osp

proj_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.insert(0, proj_dir)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import readlines, sec_to_hm_str, osp_join
from moduv.model import Model
from moduv.datasets.dataset import Dataset
from moduv.options import MODUVOptions

class Trainer:
    def __init__(self, options):
        self.opt = options
        # Set seed and run assertions
        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)


        print('\n=============== Trainer Initialization ===============')

        self.device = torch.device("cuda:{}".format(self.opt.cuda_id) if torch.cuda.is_available() else "cpu")
        self.model = Model(backbone_weights=self.opt.backbone_weights, backbone_path=self.opt.backbone_path).to(self.device) 
        self.optim = optim.Adam(self.model.parameters(), self.opt.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, self.opt.scheduler_step_size, 0.5)

        self.log_path = os.path.join(self.opt.save_dir, self.opt.name, self.opt.round_name)

        print('Using train file path: ' + self.opt.train_files)
        self.train_filenames = readlines(self.opt.train_files)

        # set up loaders
        self.setup_train_loader()
        self.num_steps_per_epoch = len(self.train_loader)
        self.num_total_steps = self.num_steps_per_epoch * self.opt.epochs
        self.save_opts()
        print(f'Number of training items: {len(self.train_dataset)}')
        print('=============== Trainer Initialization ===============\n')

    # ========== Standard functions ========== #

    def train(self):
        """ Run the entire training pipeline 
        """

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.epochs):
            print()
            self.run_epoch()
            self.setup_train_loader() # reset train loader after each epoch

            if ((self.epoch + 1) % self.opt.save_freq == 0) or (self.epoch == self.opt.epochs - 1):
                self.save_model()

    def run_epoch(self):
        """ Run a single epoch of training and validation 
        """

        self.set_train()

        gpu_time, data_loading_time = 0, 0
        before_op_time = time.time()
        
        for batch_idx, inputs in enumerate(self.train_loader):
            
            data_loading_time += (time.time() - before_op_time)
            before_op_time = time.time()

            # === Compute Starts === #
            outputs, losses = self.process_batch(inputs)
            losses['loss'].backward()

            if (self.step + 1) % self.opt.grad_agg == 0:
                self.optim.step()
                self.optim.zero_grad()
            # === Compute Ends === #

            compute_duration = time.time() - before_op_time
            gpu_time += compute_duration

            early_freq = self.opt.log_freq // 10; late_freq = self.opt.log_freq
            if (batch_idx % early_freq == 0 and self.step < late_freq) or self.step % late_freq == 0:
                self.log_time(batch_idx, compute_duration, losses['loss'].cpu().data, data_loading_time, gpu_time)
                gpu_time, data_loading_time = 0, 0
            del outputs

            self.step += 1
            before_op_time = time.time()
        self.lr_scheduler.step()

    def process_batch(self, inputs):
        """ Pass a minibatch through the network and generate images and losses 
        """
        image = [inputs['image'][0].to(self.device)]

        if self.is_train:
            masks = inputs['target']['masks'].to(self.device)       # (B, N, H, W)
            boxes = inputs['target']['boxes'].to(self.device)       # (B, N, 4)
            labels = inputs['target']['labels'].to(self.device)     # (B, N)
            target = [{'masks' : masks[0], 'boxes' : boxes[0], 'labels' : labels[0]}]
        else:
            target = None

        return self.model(image, target)
    
    # ========== Helper functions ========== #

    def setup_train_loader(self):
        """ construct self.train_loader
        """
        if self.opt.epoch_size > 0:
            train_fnames = np.random.choice(self.train_filenames, self.opt.epoch_size, replace=(self.opt.epoch_size>len(self.train_filenames)))
        else:
            train_fnames = self.train_filenames

        self.train_dataset = Dataset(self.opt.data_dir,
                                     train_fnames, 
                                     self.opt.anno_dir,
                                     self.opt.height,
                                     self.opt.width,
                                     is_train=True, 
                                     flip_freq=self.opt.flip_aug_rate, 
                                     sj_freq=self.opt.sjitter_rate,
                                     sj_min=self.opt.sjitter_min,
                                     sj_max=self.opt.sjitter_max)

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=None)
    
    def log_time(self, batch_idx, duration, loss, data_time, gpu_time):
        """ Print a logging statement to the terminal
        """

        samples_per_sec = 1 / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = 'epoch {:>3} | batch {:>6} | examples/s: {:5.1f}' + \
            ' | loss: {:.5f} | time elapsed: {} | time left: {} | CPU/GPU time: {:0.1f}s/{:0.1f}s'
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),
                                  data_time, gpu_time))
    
    def save_opts(self,):
        """ Save options to disk so we know what we ran this experiment with 
        """
        with open(osp_join(self.log_path, 'opt.json'), 'w') as f:
            json.dump(self.opt.__dict__.copy(), f, indent=2)

    def save_model(self, save_name='weights'):
        """ Save model weights and opt to disk
        """
        model_path = osp_join(self.log_path, f'{save_name}_{self.epoch:02}.pth')
        self.model.save(model_path)
    
    def set_train(self):
        """ Convert all models to training mode
        """
        self.model.set_train()
        self.is_train = True

    def set_eval(self):
        """ Convert all models to testing/evaluation mode 
        """
        self.model.set_eval()
        self.is_train = False


if __name__ == "__main__":
    options = MODUVOptions()
    opt = options.parse()
    trainer = Trainer(opt)
    trainer.train()

