import random, torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class ScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=512, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        ratio_height, ratio_width = scaled_size / image_size

        target = target.copy()
        target["size"] = scaled_size

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        if "masks" in target:
            masks = target['masks']
            scaled_masks = F.resize(masks[:, None].float(), scaled_size.tolist(), interpolation=T.InterpolationMode.NEAREST)[:, 0]
            target['masks'] = scaled_masks
        return target

    def crop_target(self, region, target):
        i, j, h, w = region

        # Process Boxes
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area

        # Process Masks
        target['masks'] = target['masks'][:, i:i + h, j:j + w]

        # Remove Zero Area Elements
        visible = area > 0
        target['masks'] = target['masks'][visible]
        target['boxes'] = target['boxes'][visible]
        target['labels'] = target['labels'][visible]

        return target

    def pad_target(self, padding, target):
        left_pad, top_pad, right_pad, bottom_pad = padding

        # Process Boxes
        boxes = target["boxes"]
        shifted_boxes = boxes + torch.as_tensor([left_pad, top_pad, left_pad, top_pad])
        target["boxes"] = shifted_boxes

        # Process Masks
        target['masks'] = torch.nn.functional.pad(target['masks'], (left_pad, right_pad, top_pad, bottom_pad), "constant", 0)
        
        return target

    def __call__(self, image, target=None):
        
        image_size = torch.tensor(image.shape[-2:])
        out_desired_size = self.desired_size
        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (image_size * random_scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())
        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)
        # randomly crop or pad images
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(), out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
                target["size"] = out_desired_size
        else:
            # Apply Random Padding
            h_pad, w_pad = (out_desired_size - scaled_size).tolist()
            top_pad, left_pad = int(h_pad*random.random()), int(w_pad*random.random())
            bottom_pad, right_pad = h_pad - top_pad, w_pad - left_pad

            padding = [left_pad, top_pad, right_pad, bottom_pad]

            output_image = F.pad(scaled_image, padding, padding_mode='constant', fill=0.5)
            if target is not None:
                target = self.pad_target(padding, target)
                target["size"] = out_desired_size

        return output_image, target