from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import shutil
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import time

from thop import profile
from thop import clever_format 
from tqdm import tqdm
cv2.setNumThreads(0) 

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

 
def Global_Gradient_Error(gt_depth, pred_depth, valid_mask, min_depth=1e-3):

    valid_mask = valid_mask.astype(bool)

    if np.sum(valid_mask) == 0:
        return np.nan

    gt_filled = fill_invalid_with_median(gt_depth, valid_mask, min_depth=min_depth)
    pred_filled = fill_invalid_with_median(pred_depth, valid_mask, min_depth=min_depth)

    log_gt = np.log(gt_filled)
    log_pred = np.log(pred_filled)

    gt_gx = cv2.Sobel(log_gt, cv2.CV_64F, 1, 0, ksize=3)
    gt_gy = cv2.Sobel(log_gt, cv2.CV_64F, 0, 1, ksize=3)

    pred_gx = cv2.Sobel(log_pred, cv2.CV_64F, 1, 0, ksize=3)
    pred_gy = cv2.Sobel(log_pred, cv2.CV_64F, 0, 1, ksize=3)

    error_map = np.abs(pred_gx - gt_gx) + np.abs(pred_gy - gt_gy)

    return np.mean(error_map[valid_mask])


def get_depth_boundary_mask(gt_depth, valid_mask, percentile=90, dilation=2, min_depth=1e-3):

    valid_mask = valid_mask.astype(bool)

    gt_filled = gt_depth.copy().astype(np.float64)

    if np.sum(valid_mask) == 0:
        boundary_mask = np.zeros_like(valid_mask, dtype=bool)
        smooth_mask = np.zeros_like(valid_mask, dtype=bool)
        return boundary_mask, smooth_mask

    median_depth = np.median(gt_filled[valid_mask])
    gt_filled[~valid_mask] = median_depth
    gt_filled[gt_filled < min_depth] = min_depth

    log_gt = np.log(gt_filled)

    gx = cv2.Sobel(log_gt, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_gt, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    grad_values = grad_mag[valid_mask]

    if grad_values.size == 0:
        boundary_mask = np.zeros_like(valid_mask, dtype=bool)
        smooth_mask = valid_mask.copy()
        return boundary_mask, smooth_mask

    threshold = np.percentile(grad_values, percentile)
    boundary_mask = np.logical_and(valid_mask, grad_mag >= threshold)

    if dilation > 0:
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
        boundary_mask = cv2.dilate(boundary_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary_mask = np.logical_and(boundary_mask, valid_mask)

    smooth_mask = np.logical_and(valid_mask, ~boundary_mask)

    return boundary_mask, smooth_mask


 
def Boundary_region_Abs_Rel(gt_depth, pred_depth, region_mask):
    region_mask = region_mask.astype(bool)

    if np.sum(region_mask) == 0:
        return np.nan

    gt_region = gt_depth[region_mask]
    pred_region = pred_depth[region_mask]

    return np.mean(np.abs(gt_region - pred_region) / gt_region)

 
def Boundary_Gradient_Error(gt_depth, pred_depth, boundary_mask, valid_mask, min_depth=1e-3):

    valid_mask = valid_mask.astype(bool)
    boundary_mask = np.logical_and(boundary_mask.astype(bool), valid_mask)

    if np.sum(boundary_mask) == 0:
        return np.nan

    gt_filled = gt_depth.copy().astype(np.float64)
    pred_filled = pred_depth.copy().astype(np.float64)

    if np.sum(valid_mask) == 0:
        return np.nan

    gt_median = np.median(gt_filled[valid_mask])
    pred_median = np.median(pred_filled[valid_mask])

    gt_filled[~valid_mask] = gt_median
    pred_filled[~valid_mask] = pred_median

    gt_filled[gt_filled < min_depth] = min_depth
    pred_filled[pred_filled < min_depth] = min_depth

    log_gt = np.log(gt_filled)
    log_pred = np.log(pred_filled)

    gt_gx = cv2.Sobel(log_gt, cv2.CV_64F, 1, 0, ksize=3)
    gt_gy = cv2.Sobel(log_gt, cv2.CV_64F, 0, 1, ksize=3)

    pred_gx = cv2.Sobel(log_pred, cv2.CV_64F, 1, 0, ksize=3)
    pred_gy = cv2.Sobel(log_pred, cv2.CV_64F, 0, 1, ksize=3)

    error_map = np.abs(pred_gx - gt_gx) + np.abs(pred_gy - gt_gy)

    return np.mean(error_map[boundary_mask])


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def fill_invalid_with_median(depth, valid_mask, min_depth=1e-3):

    valid_mask = valid_mask.astype(bool)
    depth_filled = depth.copy().astype(np.float64)

    if np.sum(valid_mask) == 0:
        depth_filled[depth_filled < min_depth] = min_depth
        return depth_filled

    median_depth = np.median(depth_filled[valid_mask])
    depth_filled[~valid_mask] = median_depth
    depth_filled[depth_filled < min_depth] = min_depth

    return depth_filled


def compute_log_depth_gradient_magnitude(depth, valid_mask, min_depth=1e-3):
    
    depth_filled = fill_invalid_with_median(depth, valid_mask, min_depth=min_depth)
    log_depth = np.log(depth_filled)

    gx = cv2.Sobel(log_depth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_depth, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    return grad_mag


def get_high_gradient_mask(gt_depth, valid_mask, percentile=90, dilation=0, min_depth=1e-3):
   
    valid_mask = valid_mask.astype(bool)

    if np.sum(valid_mask) == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    grad_mag = compute_log_depth_gradient_magnitude(
        gt_depth,
        valid_mask,
        min_depth=min_depth
    )

    grad_values = grad_mag[valid_mask]

    if grad_values.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    threshold = np.percentile(grad_values, percentile)
    high_grad_mask = np.logical_and(valid_mask, grad_mag >= threshold)

    if dilation > 0:
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
        high_grad_mask = cv2.dilate(
            high_grad_mask.astype(np.uint8),
            kernel,
            iterations=1
        ).astype(bool)
        high_grad_mask = np.logical_and(high_grad_mask, valid_mask)

    return high_grad_mask


 
def Edge_Aware_RMSE(gt_depth, pred_depth, valid_mask, edge_weight=2.0, min_depth=1e-3):
   
    valid_mask = valid_mask.astype(bool)

    if np.sum(valid_mask) == 0:
        return np.nan

    grad_mag = compute_log_depth_gradient_magnitude(
        gt_depth,
        valid_mask,
        min_depth=min_depth
    )

    grad_valid = grad_mag[valid_mask]

    if grad_valid.size == 0:
        return np.nan

    norm_factor = np.percentile(grad_valid, 95) + 1e-12
    grad_norm = grad_mag / norm_factor
    grad_norm = np.clip(grad_norm, 0.0, 1.0)

    weight = 1.0 + edge_weight * grad_norm

    sq_error = (pred_depth - gt_depth) ** 2

    edge_aware_rmse = np.sqrt(
        np.sum(weight[valid_mask] * sq_error[valid_mask]) /
        np.sum(weight[valid_mask])
    )

    return edge_aware_rmse

 

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80 
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    opt.load_weights_folder = "/path/weights"

    if opt.ext_disp_to_eval is None: 

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width,
                                           [0], 4, is_train=False)
        
        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False) 

        if opt.encoder_mobilevit == "s":
            encoder = networks.mobile_vit_small(opt.weights_dir)
        elif opt.encoder_mobilevit == "xs":
            encoder = networks.mobile_vit_x_small(opt.weights_dir)
        elif opt.encoder_mobilevit == "xxs":
            encoder = networks.mobile_vit_xx_small(opt.weights_dir)

        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, decoder_channel_scale=opt.decoder_channel_scale)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda(0)
        encoder.eval()
        depth_decoder.cuda(0)
        depth_decoder.eval()

        pred_disps = []
        print("-> Computing predictions with size {}x{}".format(
            opt.width, opt.height))

 
        with torch.no_grad():
            for data in tqdm(dataloader):
                input_color = data[("color", 0, 0)].cuda(0)

                if opt.post_process: 
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                
                output = depth_decoder(encoder(input_color))
               
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp) 
            
        pred_disps = np.concatenate(pred_disps)

    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps: 
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval: 
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    

    if opt.eval_stereo: 
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

 
    errors = []
    
    boundary_region_abs_rel_errors = []
    smooth_region_abs_rel_errors = []
    boundary_gradient_errors = []

    edge_aware_rmse_errors = []
    global_gradient_errors = []

    high_gradient_region_abs_rel_errors = []


    ratios = []

    for i in tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
 
        pred_disp = pred_disps[i]
       
        save_vis = True
        save_dir = "vis_pic/kitti"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(os.path.join(save_dir,'sr')):
            os.makedirs(os.path.join(save_dir,'sr'))

        if save_vis: 
            pic_name = filenames[i].split()[0].split('/')[-1]+'_'+filenames[i].split()[1]+'_sr.png'
            plt.imsave(os.path.join(save_dir,'sr',pic_name), pred_disp, cmap='magma')

            src_img = 'kitti_data_png/'+ filenames[i].split()[0]+'/'+'image_02/data'+'/'+filenames[i].split()[1] +'.png'
            dst_src_img_dir = 'vis_pic/kitti/src_imgs'

            if not os.path.exists(dst_src_img_dir):
                os.makedirs(dst_src_img_dir)

            dst_src_img_path = os.path.join(dst_src_img_dir,pic_name)
            shutil.copy(src_img,dst_src_img_path)
 
      
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height)) 
        pred_depth = 1 / pred_disp

   
        if opt.eval_split == "eigen":
           
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

      
        gt_depth_full = gt_depth.copy()
        pred_depth_full = pred_depth.copy()

        pred_depth_full *= opt.pred_depth_scale_factor

        pred_depth_valid = pred_depth_full[mask]
        gt_depth_valid = gt_depth_full[mask]

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth_valid) / np.median(pred_depth_valid)
            ratios.append(ratio)

            pred_depth_full *= ratio

        pred_depth_full[pred_depth_full < MIN_DEPTH] = MIN_DEPTH
        pred_depth_full[pred_depth_full > MAX_DEPTH] = MAX_DEPTH


        # -------------------------------------------------------------------------
        # Boundary-sensitive evaluation
        # -------------------------------------------------------------------------
        boundary_mask, smooth_mask = get_depth_boundary_mask(
            gt_depth_full,
            mask,
            percentile=90,
            dilation=2,
            min_depth=MIN_DEPTH
        )

        boundary_region_abs_rel = Boundary_region_Abs_Rel(
            gt_depth_full,
            pred_depth_full,
            boundary_mask
        )

 
        smooth_region_abs_rel = Boundary_region_Abs_Rel(
            gt_depth_full,
            pred_depth_full,
            smooth_mask
        )

        
        boundary_gradient_error = Boundary_Gradient_Error(
            gt_depth_full,
            pred_depth_full,
            boundary_mask,
            mask,
            min_depth=MIN_DEPTH
        )


        high_gradient_mask = get_high_gradient_mask(
            gt_depth_full,
            mask,
            percentile=90,
            dilation=0,
            min_depth=MIN_DEPTH
        )

 
        high_gradient_region_abs_rel = Boundary_region_Abs_Rel(
            gt_depth_full,
            pred_depth_full,
            high_gradient_mask
        )

        edge_aware_rmse = Edge_Aware_RMSE(
            gt_depth_full,
            pred_depth_full,
            mask,
            edge_weight=2.0,
            min_depth=MIN_DEPTH
        )

        global_gradient_error = Global_Gradient_Error(
            gt_depth_full,
            pred_depth_full,
            mask,
            min_depth=MIN_DEPTH
        )
 
        boundary_region_abs_rel_errors.append(boundary_region_abs_rel)
        smooth_region_abs_rel_errors.append(smooth_region_abs_rel)
        boundary_gradient_errors.append(boundary_gradient_error)

        high_gradient_region_abs_rel_errors.append(high_gradient_region_abs_rel)
        edge_aware_rmse_errors.append(edge_aware_rmse)
        global_gradient_errors.append(global_gradient_error)
   
        pred_depth_eval = pred_depth_full[mask]
        gt_depth_eval = gt_depth_full[mask]

        errors.append(compute_errors(gt_depth_eval, pred_depth_eval))

    if not opt.disable_median_scaling: 
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

  

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
    ))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")


    mean_boundary_region_abs_rel = np.nanmean(np.array(boundary_region_abs_rel_errors))
    mean_smooth_region_abs_rel = np.nanmean(np.array(smooth_region_abs_rel_errors))
    mean_boundary_gradient_error = np.nanmean(np.array(boundary_gradient_errors))

    print("\n  " + ("{:>8} | " * 7).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
    ))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    print("\n  {:>18} | {:>18} | {:>24} | ".format(
        "boundary_region_abs_rel", "smooth_region_abs_rel", "boundary_gradient_error"
    ))
    print("&{: 18.6f}  &{: 18.6f}  &{: 24.6f}  \\\\".format(
        mean_boundary_region_abs_rel,
        mean_smooth_region_abs_rel,
        mean_boundary_gradient_error
    ))


    mean_high_gradient_region_abs_rel = np.nanmean(np.array(high_gradient_region_abs_rel_errors))
    mean_edge_aware_rmse = np.nanmean(np.array(edge_aware_rmse_errors))
    mean_global_gradient_error = np.nanmean(np.array(global_gradient_errors))
 
    print("\n  {:>22} | {:>18} | {:>18} | ".format(
        "high_gradient_region_abs_rel",
        "edge_aware_rmse",
        "global_gradient_error",
     
    ))

    print("&{: 22.6f}  &{: 18.6f}  &{: 18.6f}  \\\\".format(
        mean_high_gradient_region_abs_rel,
        mean_edge_aware_rmse,
        mean_global_gradient_error,
    ))

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
