import torch

from tqdm import tqdm
import numpy as np
import cv2

from scipy.spatial.distance import cdist


def mean_pixel_accuracy(pred, target, num_classes):
    with torch.no_grad():
        acc_per_class = []
        for cls in range(1, num_classes):
            cls_mask = (target == cls)
            total_cls = cls_mask.sum().item()
            if total_cls == 0:
                continue
            correct_cls = (pred[cls_mask] == cls).sum().item()
            acc_per_class.append(correct_cls / total_cls)
        return sum(acc_per_class) / len(acc_per_class) if acc_per_class else 0.0


def compute_hd_multiclass(pred, gt, num_classes):
    hd_scores = []
    hd95_scores = []
    for cls in range(1,num_classes):
        pred_bin = (pred == cls).astype(np.uint8)
        gt_bin = (gt == cls).astype(np.uint8)

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            continue  # 跳过该类别，不影响分数
        elif pred_bin.sum() == 0 or gt_bin.sum() == 0:
            continue  # 也跳过类别缺失情况

        pred_boundary = pred_bin - cv2.erode(pred_bin, np.ones((3, 3), np.uint8))
        gt_boundary = gt_bin - cv2.erode(gt_bin, np.ones((3, 3), np.uint8))

        pred_coords = np.argwhere(pred_boundary)
        gt_coords = np.argwhere(gt_boundary)

        if len(pred_coords) < 5 or len(gt_coords) < 5:
            continue

        distances = cdist(pred_coords, gt_coords)
        hd = max(distances.min(axis=1).max(), distances.min(axis=0).max())
        forward = distances.min(axis=1)
        backward = distances.min(axis=0)
        hd95 = np.percentile(np.concatenate([forward, backward]), 95)
        hd95_scores.append(hd95)
        hd_scores.append(hd)

    return float(np.mean(hd_scores)) if hd_scores else 0.0,float(np.mean(hd95_scores)) if hd_scores else 0.0


def batch_intersection_union(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = pred[pred == target]
    area_inter = torch.bincount(intersection, minlength=num_classes)
    area_pred  = torch.bincount(pred,        minlength=num_classes)
    area_true  = torch.bincount(target,      minlength=num_classes)
    area_union = area_pred + area_true - area_inter
    return area_inter.float(), area_union.float()

def get_boundary(mask):
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    return mask - erosion

def boundary_f1_score_multiclass(pred, gt, num_classes, tolerance=2):
    bf_scores = []
    for cls in range(1,num_classes):
        pred_bin = (pred == cls).astype(np.uint8)
        gt_bin   = (gt   == cls).astype(np.uint8)
        pred_b = get_boundary(pred_bin)
        gt_b   = get_boundary(gt_bin)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (2*tolerance+1, 2*tolerance+1))
        pred_dil = cv2.dilate(pred_b, se)
        gt_dil   = cv2.dilate(gt_b, se)
        pred_match = pred_b * gt_dil
        gt_match   = gt_b   * pred_dil
        n_pred = pred_b.sum()
        n_gt   = gt_b.sum()
        if   n_pred == 0 and n_gt > 0:
            precision, recall = 1, 0
        elif n_pred > 0 and n_gt == 0:
            precision, recall = 0, 1
        elif n_pred == 0 and n_gt == 0:
            precision, recall = 1, 1
        else:
            precision = pred_match.sum() / (n_pred + 1e-10)
            recall    = gt_match.sum()   / (n_gt   + 1e-10)
        bf = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        bf_scores.append(bf)
    return float(np.mean(bf_scores))


@torch.no_grad()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_classes = net.n_classes
    actual_num_classes = 2 if num_classes == 1 else num_classes

    # 初始化
    total_batches = 0
    valid_dice_batches = 0
    correct_pixels = 0
    total_pixels = 0
    total_mpa = 0
    total_hd = []
    total_bf = []

    per_class_metrics = {
        cls: {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "hd": [],
            "bf": [],
            "hd95": [],
        } for cls in range(1, actual_num_classes)
    }

    autocast_device = device.type if device.type != 'mps' else 'cpu'

    with torch.autocast(autocast_device, enabled=amp):
        with tqdm(total=len(dataloader), desc='Eval', unit='batch', leave=False) as pbar:
            for batch in dataloader:
                images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks_true = batch['mask'].to(device=device, dtype=torch.long)


                outputs = net(images)
                if num_classes == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    preds_label = preds.long().squeeze(1)
                else:
                    preds_label = outputs.argmax(dim=1)

                batch_size = preds_label.shape[0]
                preds_np = preds_label.detach().cpu().numpy()
                masks_np = masks_true.detach().cpu().numpy()

                for b in range(batch_size):
                    pred = preds_np[b]
                    gt = masks_np[b]

                    if np.all(gt == 0):
                        continue  #  跳过整图为背景的样本

                    valid_dice_batches += 1
                    correct_pixels += (pred == gt).sum()
                    total_pixels += gt.size

                    # Mean Pixel Accuracy
                    total_mpa += mean_pixel_accuracy(torch.tensor(pred), torch.tensor(gt), actual_num_classes)

                    for cls in range(1, actual_num_classes):
                        if np.sum(gt == cls) == 0:
                            continue  #  跳过该类在 GT 中未出现

                        pred_bin = (pred == cls).astype(np.uint8)
                        gt_bin   = (gt   == cls).astype(np.uint8)

                        tp = np.logical_and(pred_bin, gt_bin).sum()
                        fp = pred_bin.sum() - tp
                        fn = gt_bin.sum() - tp
                        union = pred_bin.sum() + gt_bin.sum()
                        denom = pred_bin.sum() + gt_bin.sum() - tp

                        dice = 2 * tp / (union + 1e-10) if union > 0 else 1.0
                        iou = tp / (denom + 1e-10) if denom > 0 else 1.0
                        prec = tp / (tp + fp + 1e-10)
                        recall = tp / (tp + fn + 1e-10)

                        per_class_metrics[cls]["dice"].append(dice)
                        per_class_metrics[cls]["iou"].append(iou)
                        per_class_metrics[cls]["precision"].append(prec)
                        per_class_metrics[cls]["recall"].append(recall)

                        # Boundary F1
                        bf = boundary_f1_score_multiclass(pred, gt, actual_num_classes)
                        per_class_metrics[cls]["bf"].append(bf)

                        # HD
                        pred_edge = pred_bin - cv2.erode(pred_bin, np.ones((3, 3), np.uint8))
                        gt_edge = gt_bin - cv2.erode(gt_bin, np.ones((3, 3), np.uint8))
                        pred_coords = np.argwhere(pred_edge)
                        gt_coords = np.argwhere(gt_edge)

                        if len(pred_coords) >= 5 and len(gt_coords) >= 5:
                            distances = cdist(pred_coords, gt_coords)
                            hd = max(distances.min(axis=1).max(), distances.min(axis=0).max())
                            forward = distances.min(axis=1)
                            backward = distances.min(axis=0)
                            hd95 = np.percentile(np.concatenate([forward, backward]), 95)

                            per_class_metrics[cls]["hd"].append(hd)
                            per_class_metrics[cls]["hd95"].append(hd95)

                total_batches += 1
                pbar.update(1)


    # 所有前景类下的所有样本值拉平后求平均
    def classwise_mean(metric_dict, key):
        per_class_means = [
            float(np.nanmean(metric_dict[cls][key]))
            for cls in metric_dict
            if metric_dict[cls][key]  # 至少该类在 val 中出现过
        ]
        return float(np.mean(per_class_means)) if per_class_means else 0.0

    avg_dice = classwise_mean(per_class_metrics, "dice")
    mean_iou = classwise_mean(per_class_metrics, "iou")
    mean_hd = classwise_mean(per_class_metrics, "hd")
    mean_hd95 = classwise_mean(per_class_metrics, "hd95")
    mean_bf = classwise_mean(per_class_metrics, "bf")

    # Global accuracy 和 MPA 按 batch 计算即可
    gacc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    mean_mpa = total_mpa / max(valid_dice_batches, 1)

    net.train()

    # 返回最终指标
    return {
        "dice": avg_dice,
        "miou": mean_iou,
        "mpa": mean_mpa,
        "gacc": gacc,
        "hd": mean_hd,
        "hd95": mean_hd95,
        "bf": mean_bf,
        "per_class": {
            cls: {
                k: float(np.nanmean(v)) if v else 0.0
                for k, v in metrics.items()
            } for cls, metrics in per_class_metrics.items()
        }
    }

