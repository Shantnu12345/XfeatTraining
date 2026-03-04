"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time
import sys

def _parse_vec3(s):
    vals = [float(v) for v in s.split(',')]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated floats.")
    return vals

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
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                        help='Select training data recipe.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training.')
    parser.add_argument('--n_steps', type=int, default=160000,
                        help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma for StepLR.')
    parser.add_argument('--training_res', type=int, nargs=2, default=[800, 608],
                        help='Training resolution [W H].')
    parser.add_argument('--device_num', type=str, default="0",
                        help='CUDA device number.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only run a few steps for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')

    # =========================
    # Rail manifold self-supervision args
    # =========================
    parser.add_argument('--rail_img0_path', type=str, default='',
                        help='Path to first real rail image used for self-supervision.')
    parser.add_argument('--rail_img1_path', type=str, default='',
                        help='Path to second real rail image used for self-supervision.')
    parser.add_argument('--rail_lambda', type=float, default=0.0,
                        help='Weight for rail manifold loss term.')
    parser.add_argument('--rail_lambda_dev', type=float, default=0.0,
                        help='Weight for rail manifold deviation term (log J_rail - log J_free).')
    parser.add_argument('--rail_min_matches', type=int, default=64,
                        help='Minimum pseudo matches required to apply rail loss.')
    parser.add_argument('--rail_topk', type=int, default=1024,
                        help='Top-k reliable dense points sampled per image for pseudo matching.')
    parser.add_argument('--rail_min_cos', type=float, default=0.1,
                        help='Minimum cosine similarity to keep a pseudo match.')
    parser.add_argument('--rail_phi_min', type=float, default=-0.35,
                        help='Minimum phi (radians) for 1D rail manifold search.')
    parser.add_argument('--rail_phi_max', type=float, default=0.35,
                        help='Maximum phi (radians) for 1D rail manifold search.')
    parser.add_argument('--rail_phi_grid_steps', type=int, default=41,
                        help='Grid steps for initial phi search.')
    parser.add_argument('--rail_gn_steps', type=int, default=3,
                        help='Number of GN-like refinement steps for phi.')
    parser.add_argument('--rail_lambda_entropy', type=float, default=0.0,
                        help='Entropy regularizer on match weights.')
    parser.add_argument('--rail_lambda_coverage', type=float, default=0.0,
                        help='Coverage regularizer on match weights across image bins.')
    parser.add_argument('--rail_coverage_bins', type=int, default=8,
                        help='Bins per axis for coverage regularizer.')

    parser.add_argument('--rail_radius', type=float, default=1.0,
                        help='Rail radius used by the rail model.')
    parser.add_argument('--rail_r_dc', type=_parse_mat3,
                        default=[1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0],
                        help='Device->camera rotation as 9 comma-separated floats (row-major).')
    parser.add_argument('--rail_t_dc', type=_parse_vec3, default=[0.0, 0.0, 0.0],
                        help='Device->camera translation as 3 comma-separated floats.')

    parser.add_argument('--rail_k0', type=_parse_mat3, default=None,
                        help='Optional intrinsics for image0 as 9 comma-separated floats.')
    parser.add_argument('--rail_k1', type=_parse_mat3, default=None,
                        help='Optional intrinsics for image1 as 9 comma-separated floats.')

    return parser.parse_args()


# --- original imports (kept after argparse for speed of help/--help printing) ---
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk,
        the network training itself is quite fast.
    """

    def __init__(self, megadepth_root_path,
                       synthetic_root_path,
                       ckpt_save_path,
                       model_name='xfeat_default',
                       batch_size=10, n_steps=160_000, lr=3e-4, gamma_steplr=0.5,
                       training_res=(800, 608), device_num="0", dry_run=False,
                       save_ckpt_every=500,
                       # rail args
                       rail_img0_path='',
                       rail_img1_path='',
                       rail_lambda=0.0,
                       rail_lambda_dev=0.0,
                       rail_min_matches=64,
                       rail_topk=1024,
                       rail_min_cos=0.1,
                       rail_phi_min=-0.35,
                       rail_phi_max=0.35,
                       rail_phi_grid_steps=41,
                       rail_gn_steps=3,
                       rail_lambda_entropy=0.0,
                       rail_lambda_coverage=0.0,
                       rail_coverage_bins=8,
                       rail_radius=1.0,
                       rail_r_dc=None,
                       rail_t_dc=None,
                       rail_k0=None,
                       rail_k1=None):

        os.environ["CUDA_VISIBLE_DEVICES"] = device_num
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        # Setup optimizer
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        self.training_res = (training_res[0], training_res[1])
        self.model_name = model_name
        self.ckpt_save_path = ckpt_save_path
        self.save_ckpt_every = save_ckpt_every
        self.dry_run = dry_run

        # tensorboard
        os.makedirs(ckpt_save_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(ckpt_save_path, 'logs'))

        # Setup datasets
        self.augmentor = None
        self.data_loader = None
        self.data_iter = None

        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                root_path=synthetic_root_path,
                img_size=self.training_res,
                gray=True,
                use_tps=True
            )

        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            md_dataset = MegaDepthDataset(
                root_path=megadepth_root_path,
                train=True,
                image_size=self.training_res
            )
            self.data_loader = DataLoader(
                md_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True
            )
            self.data_iter = iter(self.data_loader)

        # =========================
        # Rail settings
        # =========================
        self.rail_lambda = float(rail_lambda)
        self.rail_lambda_dev = float(rail_lambda_dev)
        self.rail_min_matches = int(rail_min_matches)
        self.rail_topk = int(rail_topk)
        self.rail_min_cos = float(rail_min_cos)
        self.rail_phi_min = float(rail_phi_min)
        self.rail_phi_max = float(rail_phi_max)
        self.rail_phi_grid_steps = int(rail_phi_grid_steps)
        self.rail_gn_steps = int(rail_gn_steps)
        self.rail_lambda_entropy = float(rail_lambda_entropy)
        self.rail_lambda_coverage = float(rail_lambda_coverage)
        self.rail_coverage_bins = int(rail_coverage_bins)
        self.rail_radius = float(rail_radius)

        # extrinsics
        if rail_r_dc is None:
            self.rail_r_dc = torch.eye(3, device=self.dev, dtype=torch.float32)
        else:
            self.rail_r_dc = torch.tensor(rail_r_dc, device=self.dev, dtype=torch.float32).view(3, 3)

        if rail_t_dc is None:
            self.rail_t_dc = torch.zeros((3,), device=self.dev, dtype=torch.float32)
        else:
            self.rail_t_dc = torch.tensor(rail_t_dc, device=self.dev, dtype=torch.float32).view(3)

        # intrinsics (optional)
        self.rail_k0 = None
        self.rail_k1 = None
        if rail_k0 is not None:
            self.rail_k0 = torch.tensor(rail_k0, device=self.dev, dtype=torch.float32).view(3, 3)
        if rail_k1 is not None:
            self.rail_k1 = torch.tensor(rail_k1, device=self.dev, dtype=torch.float32).view(3, 3)

        self.rail_pair = None
        self.rail_enabled = (
            (self.rail_lambda > 0.0 or self.rail_lambda_dev > 0.0 or self.rail_lambda_entropy > 0.0 or self.rail_lambda_coverage > 0.0)
            and (rail_img0_path is not None and rail_img1_path is not None and len(rail_img0_path) > 0 and len(rail_img1_path) > 0)
        )
        # Phase-1: Essential-matrix rail loss requires calibrated intrinsics for BOTH images.
        if self.rail_enabled and (self.rail_lambda > 0.0 or self.rail_lambda_dev > 0.0):
            if self.rail_k0 is None or self.rail_k1 is None:
                raise ValueError(
                    "[Rail] rail loss enabled but intrinsics are missing. "
                    "Provide --rail_k0 and --rail_k1 (fx,fy,cx,cy in a 3x3 K) so we can normalize points with K^-1."
                )
        if self.rail_enabled:
            self.rail_pair = self._load_rail_pair(rail_img0_path, rail_img1_path)
            print(f"[Rail] Enabled self-supervision with pair: {rail_img0_path} | {rail_img1_path}")
            if (self.rail_k0 is None) != (self.rail_k1 is None):
                print("[Rail] Only one of rail_k0/rail_k1 was provided. Falling back to non-intrinsic normalization.")
        elif self.rail_lambda > 0:
            print("[Rail] rail_lambda > 0 but rail image paths are missing. Rail loss disabled.")

    def _load_rail_pair(self, img0_path, img1_path):
        import cv2
        w, h = self.training_res
        im0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
        im1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        if im0 is None or im1 is None:
            raise RuntimeError(f"[Rail] Could not read one of the rail images: {img0_path}, {img1_path}")
        h0, w0 = im0.shape[:2]
        h1, w1 = im1.shape[:2]
        im0 = cv2.resize(im0, (w, h))
        im1 = cv2.resize(im1, (w, h))

        # scale intrinsics to resized resolution once
        if self.rail_k0 is not None:
            sx0, sy0 = float(w) / float(w0), float(h) / float(h0)
            S0 = torch.tensor([[sx0, 0.0, 0.0],
                               [0.0, sy0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=self.rail_k0.dtype, device=self.dev)
            self.rail_k0 = S0 @ self.rail_k0
        if self.rail_k1 is not None:
            sx1, sy1 = float(w) / float(w1), float(h) / float(h1)
            S1 = torch.tensor([[sx1, 0.0, 0.0],
                               [0.0, sy1, 0.0],
                               [0.0, 0.0, 1.0]], dtype=self.rail_k1.dtype, device=self.dev)
            self.rail_k1 = S1 @ self.rail_k1

        t0 = torch.tensor(im0, dtype=torch.float32, device=self.dev).permute(2, 0, 1).unsqueeze(0) / 255.0
        t1 = torch.tensor(im1, dtype=torch.float32, device=self.dev).permute(2, 0, 1).unsqueeze(0) / 255.0

        # IMPORTANT: match original training behavior -> grayscale
        t0 = t0.mean(1, keepdim=True)
        t1 = t1.mean(1, keepdim=True)
        return t0, t1

    def train(self):

        self.net.train()
        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):

                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            d = next(self.data_iter)
                        except StopIteration:
                            print("End of DATASET!")
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)

                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                # Join megadepth & synthetic data
                with torch.inference_mode():
                    # RGB -> GRAY
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    # Cat two batches
                    if self.model_name in ('xfeat_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('xfeat_synthetic'):
                        p1 = p1s
                        p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                # Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                # Track some displayed metrics
                acc_coarse_0 = 0.0
                acc_coarse = 0.0
                acc_coords = 0.0
                acc_pos = 0.0
                nb_coarse = 0

                for b in range(len(positives_c)):
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                    m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

                    h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                    h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    # Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                    acc_pos = (acc_pos1 + acc_pos2) / 2.0

                    loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = utils.track_accuracy(m1, m2)
                    else:
                        acc_coarse = utils.track_accuracy(m1, m2)

                    nb_coarse = len(m1)

                loss = torch.cat(loss_items, -1).mean()

                # Rail loss (optional)
                rail_loss = torch.zeros((), device=self.dev)
                rail_dev = torch.zeros((), device=self.dev)
                rail_reg = torch.zeros((), device=self.dev)
                rail_phi = torch.zeros((), device=self.dev)
                rail_n_matches = 0

                if self.rail_enabled:
                    rail0, rail1 = self.rail_pair  # already grayscale
                    feats_r0, _, hmap_r0 = self.net(rail0)
                    feats_r1, _, hmap_r1 = self.net(rail1)

                    xr0, xr1, wr = extract_xfeat_matches(
                        feats_r0[0], feats_r1[0], hmap_r0[0, 0], hmap_r1[0, 0],
                        topk=self.rail_topk, min_cos=self.rail_min_cos
                    )

                    # Convert feature-map coordinates to image pixel coordinates.
                    sy0 = rail0.shape[-2] / feats_r0.shape[-2]
                    sx0 = rail0.shape[-1] / feats_r0.shape[-1]
                    sy1 = rail1.shape[-2] / feats_r1.shape[-2]
                    sx1 = rail1.shape[-1] / feats_r1.shape[-1]
                    xr0 = torch.stack([(xr0[:, 0] + 0.5) * sx0 - 0.5, (xr0[:, 1] + 0.5) * sy0 - 0.5], dim=1)
                    xr1 = torch.stack([(xr1[:, 0] + 0.5) * sx1 - 0.5, (xr1[:, 1] + 0.5) * sy1 - 0.5], dim=1)

                    rail_n_matches = int(len(wr))
                    if rail_n_matches >= self.rail_min_matches:
                        rail_loss, rail_dev, _, rail_reg, rail_phi, _ = rail_self_supervision_loss(
                            x1=xr0, x2=xr1, w=wr,
                            img_h=rail0.shape[-2], img_w=rail0.shape[-1],
                            phi_min=self.rail_phi_min, phi_max=self.rail_phi_max,
                            phi_grid_steps=self.rail_phi_grid_steps,
                            gn_steps=self.rail_gn_steps,
                            use_dev=(self.rail_lambda_dev > 0),
                            lambda_entropy=self.rail_lambda_entropy,
                            lambda_coverage=self.rail_lambda_coverage,
                            coverage_bins=self.rail_coverage_bins,
                            rail_radius=self.rail_radius,
                            r_dc=self.rail_r_dc,
                            t_dc=self.rail_t_dc,
                            k0=self.rail_k0,
                            k1=self.rail_k1
                        )
                        loss = loss + self.rail_lambda * rail_loss + self.rail_lambda_dev * rail_dev + rail_reg

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                # Save checkpoint
                if (i + 1) % self.save_ckpt_every == 0:
                    print('saving iter ', i + 1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i + 1}.pth')

                pbar.set_description(
                    'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} #matches_c: {:d} rail_loss {:.4f} rail_dev {:.4f} rail_nm {:d} phi {:.3f}'.format(
                        loss.item(), acc_coarse_0, acc_coarse, acc_coords, nb_coarse,
                        rail_loss.item() if self.rail_enabled else 0.0,
                        rail_dev.item() if self.rail_enabled else 0.0,
                        rail_n_matches,
                        rail_phi.item() if self.rail_enabled else 0.0
                    )
                )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)

                if self.rail_enabled:
                    self.writer.add_scalar('Rail/j_rail', rail_loss.item(), i)
                    self.writer.add_scalar('Rail/l_dev', rail_dev.item(), i)
                    self.writer.add_scalar('Rail/n_matches', rail_n_matches, i)
                    self.writer.add_scalar('Rail/phi', rail_phi.item(), i)

                if self.dry_run and i > 30:
                    print("Dry run finished.")
                    break


def main():
    args = parse_arguments()

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path,
        synthetic_root_path=args.synthetic_root_path,
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=(args.training_res[0], args.training_res[1]),
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,

        rail_img0_path=args.rail_img0_path,
        rail_img1_path=args.rail_img1_path,
        rail_lambda=args.rail_lambda,
        rail_lambda_dev=args.rail_lambda_dev,
        rail_min_matches=args.rail_min_matches,
        rail_topk=args.rail_topk,
        rail_min_cos=args.rail_min_cos,
        rail_phi_min=args.rail_phi_min,
        rail_phi_max=args.rail_phi_max,
        rail_phi_grid_steps=args.rail_phi_grid_steps,
        rail_gn_steps=args.rail_gn_steps,
        rail_lambda_entropy=args.rail_lambda_entropy,
        rail_lambda_coverage=args.rail_lambda_coverage,
        rail_coverage_bins=args.rail_coverage_bins,
        rail_radius=args.rail_radius,
        rail_r_dc=args.rail_r_dc,
        rail_t_dc=args.rail_t_dc,
        rail_k0=args.rail_k0,
        rail_k1=args.rail_k1,
    )

    trainer.train()


if __name__ == "__main__":
    main()