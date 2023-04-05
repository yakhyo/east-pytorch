import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.eps = eps

    def __call__(self, inputs, targets):
        intersection = torch.sum(inputs * targets)
        denominator = torch.sum(inputs) + torch.sum(targets)

        # calculate the dice loss
        dice_score = (2.0 * intersection + self.eps) / (denominator + self.eps)
        loss = 1 - dice_score

        return loss


class GeoLoss:
    def __call__(self, gt_geo, pred_geo):
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)

        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)

        return iou_loss_map, angle_loss_map


class Loss(nn.Module):
    def __init__(self, weight_angle=10):
        super(Loss, self).__init__()
        self.weight_angle = weight_angle
        self.dice_loss = DiceLoss(eps=1e-5)
        self.geo_loss = GeoLoss()

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        cls_loss = self.dice_loss(pred_score * (1 - ignored_map), gt_score)

        iou_loss_map, angle_loss_map = self.geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss

        return {"geo_loss": geo_loss, "cls_loss": cls_loss}
