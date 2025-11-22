import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def get_color_map(classes, colormap='tab20'):
    cmap = plt.get_cmap(colormap)
    color_list = []
    for i in range(classes):
        rgb = cmap(i / max(classes - 1, 1))[:3]
        rgb255 = tuple([int(x * 255) for x in rgb])
        color_list.append(rgb255)
    return color_list


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       save_img_dir=None, case=None, z_spacing=1, csv_writer=None):
    image = image.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    if image.ndim == 4:
        image = image[0]
    elif image.ndim == 3 and image.shape[0] == 3:
        image = image[0]

    if label.ndim == 4:
        label = label[0]
    elif label.ndim == 3 and label.shape[0] != image.shape[0]:
        label = label[0]

    color_map = get_color_map(classes, colormap='tab20')

    if image.ndim == 3:  # (Z, H, W)
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
            if save_img_dir is not None and case is not None:
                rgb_img = np.zeros((*pred.shape, 3), dtype=np.uint8)
                for k in range(classes):
                    rgb_img[pred == k] = color_map[k]
                rgb_img = np.fliplr(rgb_img)
                rgb_img = np.rot90(rgb_img, k=1)
                rgb_pil = Image.fromarray(rgb_img)
                rgb_pil.save(os.path.join(save_img_dir, f"{case}_slice{ind}_pred.png"))

    elif image.ndim == 2:
        x, y = image.shape[0], image.shape[1]
        slice = image
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
        prediction = pred
        if save_img_dir is not None and case is not None:
            rgb_img = np.zeros((*pred.shape, 3), dtype=np.uint8)
            for k in range(classes):
                rgb_img[pred == k] = color_map[k]
            rgb_img = np.fliplr(rgb_img)
            rgb_img = np.rot90(rgb_img, k=1)
            rgb_pil = Image.fromarray(rgb_img)
            rgb_pil.save(os.path.join(save_img_dir, f"{case}_pred.png"))

    metric_list = []
    dsc_dict = {}
    for i in range(1, classes):
        dice, hd95 = calculate_metric_percase(prediction == i, label == i)
        metric_list.append((dice, hd95))
        dsc_dict[f'class_{i}'] = dice

    if csv_writer is not None and case is not None:
        row = [case] + [dsc_dict[f'class_{i}'] for i in range(1, classes)]
        csv_writer.writerow(row)

    return metric_list
