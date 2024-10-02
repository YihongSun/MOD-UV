import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):

    def __init__(self, num_classes=2, backbone_weights='moco', backbone_path='./ckpts/moco_v2_800ep_pretrain.pth.tar'):
        super(Model, self).__init__()
        self.num_classes = num_classes  # should be 2 for class-agnostic 
        self.to_tensor = torchvision.transforms.ToTensor()

        assert backbone_weights in {'scratch', 'imagenet', 'moco'}, f'backbone_weights option {backbone_weights} not recognized.' 

        if backbone_weights == 'scratch':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=self.num_classes, weights_backbone=None)
        elif backbone_weights == 'imagenet':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=self.num_classes)
        elif backbone_weights == 'moco':
            # modified from initialization of torchvision.models.detection.maskrcnn_resnet50_fpn
            from torchvision.models import resnet50
            from torchvision.models.detection.backbone_utils import _validate_trainable_layers, _resnet_fpn_extractor
            from torchvision.models.detection.mask_rcnn import MaskRCNN
            from torchvision.ops import misc as misc_nn_ops

            is_trained = True; trainable_backbone_layers = None
            trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
            norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

            weights_backbone = None; progress = True
            backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
            print(f'Loading MoCov2 backbone from {backbone_path} ...')
            obj = torch.load(backbone_path, map_location="cpu")["state_dict"]
            newmodel = {}
            for k, v in obj.items():
                if k.startswith("module.encoder_q."):
                    newmodel[k.replace("module.encoder_q.", "")] = v
            # OK: _IncompatibleKeys(missing_keys=['fc.weight', 'fc.bias'], unexpected_keys=['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias'])
            backbone.load_state_dict(newmodel, strict=False) 

            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
            self.model = MaskRCNN(backbone, num_classes=self.num_classes)


    def forward(self, image, target=None):
        """ Forward pass of the model
        """
        if type(image) is not list:
            model_dev = list(self.model.parameters())[0].device
            image = [self.to_tensor(image).to(model_dev)]

        out = self.model(image, target)

        outputs, losses = list(), dict()
        if target is not None:  # training
            losses['loss'] = sum([out[k] for k in out.keys()])
            losses.update(out)
        else:
            outputs = out
        return outputs, losses

    # ========== Helper functions ========== #

    def set_score_thrd(self, thrd):
        """ Reset score threshold for final predictions
        """
        self.model.roi_heads.score_thresh = thrd
    
    def save(self, model_path):
        """ Save model weights to disk
        """
        torch.save(self.model.state_dict(), model_path)
    
    def load(self, model_path, map_location='cpu'):
        print(f'Loading mrcnn weights from {model_path} ...')

        if not osp.exists(model_path):
            print(f'\t<<< FAILED :: Path {model_path} not found >>>')
            return

        try:
            ckpt = torch.load(model_path, map_location=map_location)
            self.model.load_state_dict(ckpt)
        except:
            print(f'\t<<< FAILED :: state_dict mismatched >>>')

    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.eval()

