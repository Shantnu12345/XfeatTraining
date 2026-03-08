"""
        "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
        https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
         Extended with optional rail-manifold self-supervision (circular / linear).
"""
import argparse
import glob
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import numpy as np
import tqdm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *
from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from modules.dataset.rail_dataset import RailDataset


def _parse_mat3(s):
    vals = [float(v) for v in s.split(',')]
    if len(vals) != 9:
        raise argparse.ArgumentTypeError("Expected 9 comma-separated floats.")
    return vals


def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")
    parser.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, default='/tmp/xfeat_ckpt',
                        help='Path to save the checkpoints. Default is /tmp/xfeat_ckpt.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth', 'xfeat_rail'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps. '
                             'xfeat_rail trains exclusively on the rail self-supervision loss (no MegaDepth/COCO).')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    ## Training scheme --- Fine-tuning control (train only last layers) ---
    parser.add_argument('--pretrained_path', type=str, default='weights/xfeat.pt',
                help='Optional path to pretrained XFeat weights (.pth) to load before training. Default: weights/xfeat.pt')
    parser.add_argument('--finetune_last_layers', action='store_true',
                   help='If set, freeze the whole network and train only the modules listed in --finetune_modules.')
    parser.add_argument('--finetune_modules', type=str,
                   default='block_fusion,heatmap_head,keypoint_head,fine_matcher',
                   help='Comma-separated list of XFeatModel attribute names to unfreeze when --finetune_last_layers is set.')
    parser.add_argument('--reinit_last_layers', action='store_true', default=True,
                   help='Load pretrained weights for backbone, but re-initialize the modules '
                        'listed in --finetune_modules with fresh random weights. All params remain trainable. Default=True')
    # --- Rail self-supervision (simple interface) ---
    parser.add_argument('--rail_mode', type=str, default='circular',
                        choices=['circular', 'linear'],
                        help="Rail manifold type: 'circular' or 'linear'.")
    parser.add_argument('--rail_data_path', type=str, default='',
                        help='Folder with rail images (filenames = timestamps). Empty to disable.')
    parser.add_argument('--rail_lambda', type=float, default=0.0,
                        help='Weight for rail self-supervision loss. 0 to disable.')
    parser.add_argument('--rail_k', type=_parse_mat3, default=None,
                        help='Intrinsics K (row-major 9 floats), shared for all rail images. '
                             'Not required when --device_calib_path is provided.')
    parser.add_argument('--device_calib_path', type=str, default='',
                        help='Path to device_calibration.xml. When provided, the full fisheye '
                             'undistortion model from pycameramodel is used instead of --rail_k.')
    parser.add_argument('--rail_cam_name', type=str, default='',
                        help='Camera name inside the device calibration XML to use for undistortion. '
                             'If empty, the first camera in the XML is used.')
    parser.add_argument('--imu_name', type=str, default='',
                        help='IMU config key inside device_calibration.xml to read ombc/tbc from '
                             '(e.g. "imu_config_None"). Empty = use the primary IMU (recommended).')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    return args


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk,
        the network training itself is quite fast.
    """
    def __init__(self, args):
        self.args = args

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)
        if args.pretrained_path:
            ckpt = torch.load(args.pretrained_path, map_location=self.dev)
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            self.net.load_state_dict(state_dict, strict=False)
            print(f"[Init] Loaded pretrained weights from: {args.pretrained_path}")

        # Optional: reinitialize last layers with random weights
        if getattr(args, 'reinit_last_layers', False):
            for name in (m.strip() for m in str(args.finetune_modules).split(',') if m.strip()):
                if hasattr(self.net, name):
                    self._reinit_module(getattr(self.net, name))

        # ------------------------------
        # Optional fine-tuning: train only selected (late) modules
        # ------------------------------
        if getattr(args, 'finetune_last_layers', False):
            # Freeze everything
            for p_ in self.net.parameters():
                p_.requires_grad = False
            # Unfreeze selected modules by attribute name
            mod_names = [m.strip() for m in str(args.finetune_modules).split(',') if m.strip()]
            for name in mod_names:
                if hasattr(self.net, name):
                    mod = getattr(self.net, name)
                    for p_ in mod.parameters():
                        p_.requires_grad = True
                else:
                    print(f"[FineTune] WARNING: XFeatModel has no attribute '{name}'. Skipping.")
            # Prevent BatchNorm running-stat updates when most of the network is frozen
            for m in self.net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
            n_train = sum(p_.numel() for p_ in self.net.parameters() if p_.requires_grad)
            n_total = sum(p_.numel() for p_ in self.net.parameters())
            print(f"[FineTune] Trainable params: {n_train}/{n_total} ({100.0*n_train/max(n_total,1):.2f}%)")

        # Optimizer / scheduler (uses only params with requires_grad=True)
        self.batch_size = args.batch_size
        self.steps = args.n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=args.gamma_steplr)

        ##################### Synthetic COCO INIT ##########################
        if args.training_type in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = args.synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if args.training_type=='xfeat_default' else args.batch_size),
                                        out_resolution = args.training_res,
                                        warp_resolution = args.training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################

        ##################### MEGADEPTH INIT ##########################
        if args.training_type in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{args.megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{args.megadepth_root_path}/MegaDepth_v1"
            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                                    npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )
            self.data_loader = DataLoader(data,
                                          batch_size=int(self.batch_size * 0.6 if args.training_type=='xfeat_default' else args.batch_size),
                                          shuffle=True)
            self.data_iter = iter(self.data_loader)
        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        ##################### RAIL INIT ##########################
        self.rail_enabled = (args.rail_lambda > 0.0 and args.rail_data_path != '')
        self.rail_lambda = args.rail_lambda
        self.rail_mode = args.rail_mode
        self.rail_iter = None
        self.rail_k = None
        self.rail_cam = None
        # Camera->device extrinsics (R_cd, t_cd): loaded from device_calibration.xml when
        # device_calib_path is provided; otherwise default to identity (no extrinsic offset).
        # Device frame convention: X=up, Y=left, Z=inward.  Rail plane = ZY plane.
        self.rail_r_cd = torch.eye(3, dtype=torch.float32, device=self.dev)
        self.rail_t_cd = torch.zeros(3, dtype=torch.float32, device=self.dev)
        if args.device_calib_path:
            _r_cd_np, _t_cd_np = load_rail_extrinsics_from_calib(
                args.device_calib_path,
                imu_name=args.imu_name if args.imu_name else None
            )
            self.rail_r_cd = torch.from_numpy(_r_cd_np).to(self.dev)
            self.rail_t_cd = torch.from_numpy(_t_cd_np).to(self.dev)
        if self.rail_enabled:
            # --- Intrinsics: prefer device_calib_path (fisheye-aware) over rail_k (pinhole) ---
            if args.device_calib_path:
                import sys as _sys, os as _os
                _pycam_root = _os.path.abspath(
                    _os.path.join(_os.path.dirname(__file__), '..', 'pycameramodel')
                )
                if _pycam_root not in _sys.path:
                    _sys.path.insert(0, _pycam_root)
                from pycameramodel import device as _pycam
                _dev = _pycam.Device(args.device_calib_path)
                cam_keys = list(_dev.cameras.keys())
                if not cam_keys:
                    raise RuntimeError(f"[Rail] No cameras found in {args.device_calib_path}")
                cam_key = args.rail_cam_name if args.rail_cam_name in _dev.cameras else cam_keys[0]
                self.rail_cam = _dev.cameras[cam_key]
                print(f"[Rail] Using pycameramodel fisheye undistortion: "
                      f"calib={args.device_calib_path}, camera='{cam_key}'")
            elif args.rail_k is not None:
                self.rail_k = torch.tensor(args.rail_k, dtype=torch.float32, device=self.dev).view(3, 3)
                print(f"[Rail] Using pinhole K undistortion.")
            else:
                raise RuntimeError(
                    "[Rail] Provide either --device_calib_path (fisheye) "
                    "or --rail_k (pinhole) when rail is enabled."
                )
            rail_ds = RailDataset(root_dir=args.rail_data_path,
                                  length=args.n_steps)
            self.rail_loader = DataLoader(rail_ds, batch_size=1, shuffle=True)
            self.rail_iter = iter(self.rail_loader)
            print(f"[Rail] Enabled: mode={args.rail_mode}, lambda={args.rail_lambda}, "
                  f"data={args.rail_data_path}, images={len(rail_ds.image_paths)}")
        ##################### RAIL INIT END ########################

        if args.training_type == 'xfeat_rail' and not self.rail_enabled:
            raise RuntimeError(
                "[xfeat_rail] --rail_data_path and --rail_lambda (> 0) are required, "
                "plus either --device_calib_path (fisheye) or --rail_k (pinhole)."
            )

        os.makedirs(args.ckpt_save_path, exist_ok=True)
        os.makedirs(args.ckpt_save_path + '/logdir', exist_ok=True)
        self.dry_run = args.dry_run
        self.save_ckpt_every = args.save_ckpt_every
        self.ckpt_save_path = args.ckpt_save_path
        self.writer = SummaryWriter(args.ckpt_save_path + f'/logdir/{args.training_type}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = args.training_type

    @staticmethod
    def _reinit_module(module):
        """Reinitialize all parameters in a module using PyTorch's own defaults."""
        for m in module.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def _get_rail_pair(self, min_matches=30):
        """Get next rail image pair from the dataloader, resetting if exhausted. Uses SIFT matcher to ensure enough matches. Normalize only after match check."""
        attempts = 0
        while attempts < 10000:
            try:
                sample = next(self.rail_iter)
            except StopIteration:
                self.rail_iter = iter(self.rail_loader)
                sample = next(self.rail_iter)
            img0 = sample["image0"].to(self.dev).squeeze(0)
            img1 = sample["image1"].to(self.dev).squeeze(0)
            # Convert to numpy for SIFT
            img0_np = img0.detach().cpu().numpy()
            img1_np = img1.detach().cpu().numpy()
            # Ensure SIFT input is always 2D [H,W]
            if img0_np.ndim == 4:
                img0_np = img0_np[0,0,:,:]
            elif img0_np.ndim == 3:
                img0_np = img0_np[0,:,:]
            if img1_np.ndim == 4:
                img1_np = img1_np[0,0,:,:]
            elif img1_np.ndim == 3:
                img1_np = img1_np[0,:,:]
            img0_np = img0_np.astype(np.uint8)
            img1_np = img1_np.astype(np.uint8)
            if img0_np.ndim != 2 or img1_np.ndim != 2 or img0_np.shape[0] == 0 or img0_np.shape[1] == 0 or img1_np.shape[0] == 0 or img1_np.shape[1] == 0:
                attempts += 1
                continue
            sift = cv2.SIFT_create()
            kp0, des0 = sift.detectAndCompute(img0_np, None)
            kp1, des1 = sift.detectAndCompute(img1_np, None)
            if des0 is not None and des1 is not None:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des0, des1, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                if len(good) >= min_matches:
                    # Normalize to [0,1] float for model
                    img0_norm = img0.float() / 255.0
                    img1_norm = img1.float() / 255.0
                    return img0_norm, img1_norm
            attempts += 1
        raise RuntimeError(f"[Rail] Unable to find image pair with at least {min_matches} SIFT matches after 10000 attempts.")

    def train(self):
        self.net.train()
        # Keep BatchNorm layers frozen when fine-tuning only last layers
        if getattr(self.args, 'finetune_last_layers', False):
            for m in self.net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        difficulty = 0.10
        rc = RAIL_DEFAULTS  # rail config defaults from losses.py
        p1s, p2s, H1, H2 = None, None, None, None
        d = None
        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if getattr(self.args, 'finetune_last_layers', False):
                    for m in self.net.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.eval()
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MD batch
                            d = next(self.data_iter)
                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)
                    if self.augmentor is not None:
                        #Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                # ── Default metrics / loss (skipped in xfeat_rail mode) ──
                acc_coarse_0 = 0.0
                acc_coarse   = 0.0
                acc_coords   = 0.0
                nb_coarse    = 0
                loss_coarse  = 0.0
                loss_coord   = 0.0
                loss_kp_pos  = 0.0
                loss_l1      = 0.0
                acc_pos      = 0.0
                loss         = torch.tensor(0.0, device=self.dev)
                if self.model_name != 'xfeat_rail':
                    if d is not None:
                        for k in d.keys():
                            if isinstance(d[k], torch.Tensor):
                                d[k] = d[k].to(self.dev)
                        p1, p2 = d['image0'], d['image1']
                        positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)
                    if self.augmentor is not None:
                        h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                        _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)
                    #Join megadepth & synthetic data
                    with torch.inference_mode():
                        #RGB -> GRAY
                        if d is not None:
                            p1 = p1.mean(1, keepdim=True)
                            p2 = p2.mean(1, keepdim=True)
                        if self.augmentor is not None:
                            p1s = p1s.mean(1, keepdim=True)
                            p2s = p2s.mean(1, keepdim=True)
                        #Cat two batches
                        if self.model_name in ('xfeat_default'):
                            p1 = torch.cat([p1s, p1], dim=0)
                            p2 = torch.cat([p2s, p2], dim=0)
                            positives_c = positives_s_coarse + positives_md_coarse
                        elif self.model_name in ('xfeat_synthetic'):
                            p1 = p1s ; p2 = p2s
                            positives_c = positives_s_coarse
                        else:
                            positives_c = positives_md_coarse
                    #Check if batch is corrupted with too few correspondences
                    is_corrupted = False
                    for p in positives_c:
                        if len(p) < 30:
                            is_corrupted = True
                    if is_corrupted:
                        continue
                    #Forward pass
                    feats1, kpts1, hmap1 = self.net(p1)
                    feats2, kpts2, hmap2 = self.net(p2)
                    loss_items = []
                    for b in range(len(positives_c)):
                        #Get positive correspondencies
                        pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]
                        #Grab features at corresponding idxs
                        m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                        m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)
                        #grab heatmaps at corresponding idxs
                        h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                        h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                        coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))
                        #Compute losses
                        loss_ds, conf = dual_softmax_loss(m1, m2)
                        loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)
                        loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b], hmap1[b, 0], device=str(self.dev))
                        loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b], hmap2[b, 0], device=str(self.dev))
                        loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                        acc_pos = (acc_pos1 + acc_pos2)/2
                        loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)
                        loss_items.append(loss_ds.unsqueeze(0))
                        loss_items.append(loss_coords.unsqueeze(0))
                        loss_items.append(loss_kp.unsqueeze(0))
                        loss_items.append(loss_kp_pos.unsqueeze(0))
                        if b == 0:
                            acc_coarse_0 = check_accuracy(m1, m2)
                    acc_coarse = check_accuracy(m1, m2)
                    nb_coarse = len(m1)
                    loss = torch.cat(loss_items, -1).mean()
                    loss_coarse = loss_ds.item()
                    loss_coord = loss_coords.item()
                    loss_kp_pos = loss_kp_pos.item()
                    loss_l1 = loss_kp.item()
                ##################### RAIL LOSS ##########################
                rail_loss_val = 0.0
                rail_n_matches = 0
                rail_aux = 0.0
                rail_ldev_val = 0.0
                if self.rail_enabled:
                    rail0, rail1 = self._get_rail_pair()
                    feats_r0, _, hmap_r0 = self.net(rail0)
                    feats_r1, _, hmap_r1 = self.net(rail1)
                    if rc["hard_matching"]:
                        x0_f, x1_f, w = extract_xfeat_matches(
                            feats_r0[0], feats_r1[0], hmap_r0[0, 0], hmap_r1[0, 0],
                            topk=rc["topk"], min_cos=rc["min_cos"]
                        )
                    else:
                        x0_f, x1_f, w = extract_xfeat_matches_soft(
                            feats_r0[0], feats_r1[0], hmap_r0[0, 0], hmap_r1[0, 0],
                            topk=rc["topk"], tau=rc["tau"]
                        )
                    # Feature-map -> pixel coords
                    sy0 = rail0.shape[-2] / feats_r0.shape[-2]
                    sx0 = rail0.shape[-1] / feats_r0.shape[-1]
                    sy1 = rail1.shape[-2] / feats_r1.shape[-2]
                    sx1 = rail1.shape[-1] / feats_r1.shape[-1]
                    x0_px = torch.stack([(x0_f[:, 0] + 0.5) * sx0 - 0.5,
                                         (x0_f[:, 1] + 0.5) * sy0 - 0.5], dim=1)
                    x1_px = torch.stack([(x1_f[:, 0] + 0.5) * sx1 - 0.5,
                                         (x1_f[:, 1] + 0.5) * sy1 - 0.5], dim=1)
                    rail_n_matches = int(len(w))
                    if rail_n_matches >= rc["min_matches"]:
                        J_m, L_dev, _, reg, aux_param, _ = rail_self_supervision_loss(
                            mode=self.rail_mode,
                            x1_px=x0_px, x2_px=x1_px, w=w,
                            cam=self.rail_cam,
                            k0=self.rail_k, k1=self.rail_k,
                            img_h=rail0.shape[-2], img_w=rail0.shape[-1],
                            r_cd=self.rail_r_cd,
                            t_cd=self.rail_t_cd,
                        )
                        loss = loss + self.rail_lambda * J_m + rc["lambda_dev"] * L_dev + reg
                        rail_loss_val = J_m.item()
                        rail_aux = aux_param.item()
                        rail_ldev_val = L_dev.item()
                ##################### RAIL LOSS END ######################

                # Compute Backward Pass
                # In rail-only mode, skip the step if no rail matches were found
                # (loss is still a plain zero scalar with no grad_fn)
                if not loss.requires_grad:
                    self.opt.zero_grad()
                    pbar.update(1)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pt')

                pbar.set_description( 'Loss: {:.9f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f} rail: {:.4f} Ldev: {:.4f} rail_nm: {:d}'.format(
                                        loss.item(), acc_coarse_0, acc_coarse, acc_coords, loss_coarse, loss_coord, loss_l1, nb_coarse, loss_kp_pos, acc_pos, rail_loss_val, rail_ldev_val, rail_n_matches) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)
                if self.rail_enabled:
                    self.writer.add_scalar('Rail/J', rail_loss_val, i)
                    self.writer.add_scalar('Rail/L_dev', rail_ldev_val, i)
                    self.writer.add_scalar('Rail/n_matches', rail_n_matches, i)
                    self.writer.add_scalar('Rail/aux', rail_aux, i)
def main():
    args = parse_arguments()

    # Override defaults for quick local testing (comment out for CLI usage)
    args.training_type = 'xfeat_rail'
    args.rail_mode = 'linear'
    args.rail_data_path = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Linear_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/Camera2_train'
    args.rail_lambda = 0.2
    args.device_calib_path = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Linear_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/device_calibration.xml'
    args.rail_cam_name = 'trackingA'
    args.ckpt_save_path = '/local/mnt/workspace/v3dof/codes/XfeatTraining/modules/training/linear_with_fusion_kpHead_heatHead'
    args.n_steps = 501
    args.finetune_last_layers = True
    args.finetune_modules = 'block_fusion,heatmap_head'

    # args.training_type = 'xfeat_synthetic'
    # args.synthetic_root_path = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Linear_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/Camera2_train'
    # args.ckpt_save_path = '/local/mnt/workspace/v3dof/codes/XfeatTraining/modules/training/ckpt_linear_synth'

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
