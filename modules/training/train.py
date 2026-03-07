"""
XFeat training script with optional rail-manifold self-supervision (circular OR linear).

Extrinsics expected as CAMERA->DEVICE:
  p_dev = R_cd * p_cam + t_cd

Rail modes:
- circular: 1D phi search, requires R_cd and t_cd
- linear: fixed orientation, sideways translation along +X_dev, requires R_cd (t_cd unused)
"""

import argparse
import os
import time
import glob
import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from modules.model import XFeatModel
from modules.dataset.augmentation import AugmentationPipe
from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from modules.training.utils import make_batch, get_corresponding_pts
from modules.training.losses import (
    dual_softmax_loss,
    coordinate_classification_loss,
    alike_distill_loss,
    keypoint_loss,
    extract_xfeat_matches,
    extract_xfeat_matches_soft,
    rail_self_supervision_loss,
)
from modules.training import utils


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
    p = argparse.ArgumentParser(description="XFeat training script.")

    p.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                   help='Path to the MegaDepth dataset root directory.')
    p.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                   help='Path to the synthetic dataset root directory.')
    p.add_argument('--ckpt_save_path', type=str, required=True,
                   help='Path to save the checkpoints.')
    p.add_argument('--training_type', type=str, default='xfeat_default',
                   choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                   help='Training recipe.')
    p.add_argument('--batch_size', type=int, default=10)
    p.add_argument('--n_steps', type=int, default=160_000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma_steplr', type=float, default=0.5)
    p.add_argument('--pretrained_path', type=str, default='',
                   help='Optional path to pretrained XFeat weights (.pth) to load before training.')

    # --- Fine-tuning control (train only last layers) ---
    p.add_argument('--finetune_last_layers', action='store_true',
                   help='If set, freeze the whole network and train only the modules listed in --finetune_modules.')
    p.add_argument('--finetune_modules', type=str,
                   default='block_fusion,heatmap_head,keypoint_head,fine_matcher',
                   help='Comma-separated list of XFeatModel attribute names to unfreeze when --finetune_last_layers is set.')
    p.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                   default=(800, 608), help='Training resolution as width,height.')
    p.add_argument('--device_num', type=str, default='0')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--save_ckpt_every', type=int, default=500)

    # --- Rail / manifold self-supervision ---
    p.add_argument('--rail_mode', type=str, default='circular',
                   choices=['circular', 'linear'],
                   help="Which manifold to use for self-supervision.")
    p.add_argument('--rail_img0_path', type=str, default='')
    p.add_argument('--rail_img1_path', type=str, default='')
    p.add_argument('--rail_lambda', type=float, default=0.0)
    p.add_argument('--rail_lambda_dev', type=float, default=0.0)
    p.add_argument('--rail_min_matches', type=int, default=64)
    p.add_argument('--rail_topk', type=int, default=1024)
    p.add_argument('--rail_min_cos', type=float, default=0.1)
    p.add_argument('--rail_tau', type=float, default=0.1,
                   help='Softmax temperature for Phase-2 soft assignment (lower = sharper).')
    p.add_argument('--rail_hard_matching', action='store_true',
                   help='Fall back to Phase-1 hard matching instead of Phase-2 soft assignment.')

    # circular params
    p.add_argument('--rail_phi_min', type=float, default=-0.35)
    p.add_argument('--rail_phi_max', type=float, default=0.35)
    p.add_argument('--rail_phi_grid_steps', type=int, default=41)
    p.add_argument('--rail_gn_steps', type=int, default=3)
    p.add_argument('--rail_radius', type=float, default=1.0)

    # linear params
    p.add_argument('--lin_allow_both_signs', action='store_true',
                   help="If set, linear mode tries both +X and -X and takes the best.")

    # regularizers
    p.add_argument('--rail_lambda_entropy', type=float, default=0.0)
    p.add_argument('--rail_lambda_coverage', type=float, default=0.0)
    p.add_argument('--rail_coverage_bins', type=int, default=8)

    # camera->device extrinsics
    p.add_argument('--rail_r_cd', type=_parse_mat3,
                   default=[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0],
                   help='Camera->device rotation R_cd (row-major 9 floats).')
    p.add_argument('--rail_t_cd', type=_parse_vec3,
                   default=[0.0,0.0,0.0],
                   help='Camera->device translation t_cd (3 floats).')

    # intrinsics (required for rail loss)
    p.add_argument('--rail_k0', type=_parse_mat3, default=None,
                   help='Intrinsics K0 (row-major 9 floats) for rail_img0.')
    p.add_argument('--rail_k1', type=_parse_mat3, default=None,
                   help='Intrinsics K1 (row-major 9 floats) for rail_img1.')

    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        if args.pretrained_path:
            ckpt = torch.load(args.pretrained_path, map_location=self.dev)
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            self.net.load_state_dict(state_dict, strict=False)
            print(f"[Init] Loaded pretrained weights from: {args.pretrained_path}")

        self.batch_size = args.batch_size
        self.steps = args.n_steps

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
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=args.gamma_steplr)

        # Synthetic
        self.augmentor = None
        if args.training_type in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                img_dir=args.synthetic_root_path,
                device=self.dev, load_dataset=True,
                batch_size=int(self.batch_size * 0.4 if args.training_type == 'xfeat_default' else self.batch_size),
                out_resolution=args.training_res,
                warp_resolution=args.training_res,
                sides_crop=0.1,
                max_num_imgs=3_000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4_000
            )

        # MegaDepth
        self.data_iter = None
        if args.training_type in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{args.megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{args.megadepth_root_path}/MegaDepth_v1"
            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset(
                [MegaDepthDataset(root_dir=TRAINVAL_DATA_SOURCE, npz_path=path)
                 for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")]
            )
            self.data_loader = DataLoader(
                data,
                batch_size=int(self.batch_size * 0.6 if args.training_type == 'xfeat_default' else self.batch_size),
                shuffle=True
            )
            self.data_iter = iter(self.data_loader)

        os.makedirs(args.ckpt_save_path, exist_ok=True)
        os.makedirs(args.ckpt_save_path + '/logdir', exist_ok=True)
        self.writer = SummaryWriter(
            args.ckpt_save_path + f'/logdir/{args.training_type}_' + time.strftime("%Y_%m_%d-%H_%M_%S")
        )

        # Rail setup
        self.rail_enabled = (
            (args.rail_lambda > 0.0 or args.rail_lambda_dev > 0.0 or
             args.rail_lambda_entropy > 0.0 or args.rail_lambda_coverage > 0.0)
            and (args.rail_img0_path != '' and args.rail_img1_path != '')
        )

        self.rail_pair = None
        self.rail_k0 = None
        self.rail_k1 = None

        if self.rail_enabled:
            if args.rail_k0 is None or args.rail_k1 is None:
                raise RuntimeError("[Rail] rail_k0 and rail_k1 are REQUIRED for rail self-supervision.")
            self.rail_k0 = torch.tensor(args.rail_k0, dtype=torch.float32, device=self.dev).view(3, 3)
            self.rail_k1 = torch.tensor(args.rail_k1, dtype=torch.float32, device=self.dev).view(3, 3)

            self.rail_r_cd = torch.tensor(args.rail_r_cd, dtype=torch.float32, device=self.dev).view(3, 3)
            self.rail_t_cd = torch.tensor(args.rail_t_cd, dtype=torch.float32, device=self.dev).view(3)

            self.rail_pair = self._load_rail_pair(args.rail_img0_path, args.rail_img1_path)

    def _load_rail_pair(self, img0_path, img1_path):
        """Load, resize to training_res, convert to grayscale. Also scale K to resized resolution."""
        import cv2
        w_new, h_new = self.args.training_res

        im0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
        im1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        if im0 is None or im1 is None:
            raise RuntimeError(f"[Rail] Could not read: {img0_path} or {img1_path}")

        h0, w0 = im0.shape[:2]
        h1, w1 = im1.shape[:2]

        im0 = cv2.resize(im0, (w_new, h_new))
        im1 = cv2.resize(im1, (w_new, h_new))

        # scale intrinsics once
        sx0, sy0 = float(w_new) / float(w0), float(h_new) / float(h0)
        sx1, sy1 = float(w_new) / float(w1), float(h_new) / float(h1)

        S0 = torch.tensor([[sx0, 0.0, 0.0], [0.0, sy0, 0.0], [0.0, 0.0, 1.0]],
                          dtype=torch.float32, device=self.dev)
        S1 = torch.tensor([[sx1, 0.0, 0.0], [0.0, sy1, 0.0], [0.0, 0.0, 1.0]],
                          dtype=torch.float32, device=self.dev)
        self.rail_k0 = S0 @ self.rail_k0
        self.rail_k1 = S1 @ self.rail_k1

        t0 = torch.tensor(im0, dtype=torch.float32, device=self.dev).permute(2, 0, 1).unsqueeze(0) / 255.0
        t1 = torch.tensor(im1, dtype=torch.float32, device=self.dev).permute(2, 0, 1).unsqueeze(0) / 255.0
        t0 = t0.mean(dim=1, keepdim=True)
        t1 = t1.mean(dim=1, keepdim=True)
        return t0, t1

    def train(self):
        self.net.train()
        pbar = tqdm.tqdm(total=self.steps)

        for i in range(self.steps):
            # Synthetic
            if self.augmentor is not None:
                p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty=0.10)
                p1s = p1s.mean(1, keepdim=True)
                p2s = p2s.mean(1, keepdim=True)
                h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                _, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)
            else:
                p1s = p2s = positives_s_coarse = None

            # MegaDepth
            if self.data_iter is not None:
                try:
                    data = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    data = next(self.data_iter)

                im1, im2, K1, K2, T_1to2, depth1, depth2 = data
                im1 = im1.to(self.dev).mean(1, keepdim=True)
                im2 = im2.to(self.dev).mean(1, keepdim=True)
                K1 = K1.to(self.dev)
                K2 = K2.to(self.dev)
                depth1 = depth1.to(self.dev)
                depth2 = depth2.to(self.dev)

                # NOTE: keep existing megadepth warper usage consistent with your repo;
                # this call signature may vary across forks.
                positives_md_coarse = megadepth_warper.spvs_coarse(
                    {"image0": im1, "image1": im2, "K0": K1, "K1": K2, "T_0to1": T_1to2,
                     "depth0": depth1, "depth1": depth2},
                    8
                )
            else:
                im1 = im2 = positives_md_coarse = None

            # Merge batches
            with torch.inference_mode():
                if self.args.training_type == 'xfeat_default':
                    p1 = torch.cat([p1s, im1], dim=0)
                    p2 = torch.cat([p2s, im2], dim=0)
                    positives_c = positives_s_coarse + positives_md_coarse
                elif self.args.training_type == 'xfeat_synthetic':
                    p1, p2, positives_c = p1s, p2s, positives_s_coarse
                else:
                    p1, p2, positives_c = im1, im2, positives_md_coarse

            if any(len(p) < 30 for p in positives_c):
                continue

            feats1, kpts1, hmap1 = self.net(p1)
            feats2, kpts2, hmap2 = self.net(p2)

            loss_items = []
            acc_coarse_0 = acc_coarse = acc_coords = acc_pos = 0.0
            nb_coarse = 0

            for b in range(len(positives_c)):
                pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

                h1s = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                h2s = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]
                coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                loss_ds, conf = dual_softmax_loss(m1, m2)
                loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)
                loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b], hmap1[b, 0], device=str(self.dev))
                loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b], hmap2[b, 0], device=str(self.dev))
                loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                acc_pos = float((acc_pos1 + acc_pos2) / 2.0)

                loss_kp = keypoint_loss(h1s, conf) + keypoint_loss(h2s, conf)

                loss_items += [
                    loss_ds.unsqueeze(0),
                    loss_coords.unsqueeze(0),
                    loss_kp.unsqueeze(0),
                    loss_kp_pos.unsqueeze(0)
                ]

                if b == 0:
                    acc_coarse_0 = utils.track_accuracy(m1, m2)
                else:
                    acc_coarse = utils.track_accuracy(m1, m2)

                nb_coarse = len(m1)

            loss = torch.cat(loss_items, -1).mean()

            # Rail loss
            rail_loss = torch.zeros((), device=self.dev)
            rail_dev = torch.zeros((), device=self.dev)
            rail_reg = torch.zeros((), device=self.dev)
            aux_param = torch.zeros((), device=self.dev)
            rail_n_matches = 0

            if self.rail_enabled:
                rail0, rail1 = self.rail_pair
                feats_r0, _, hmap_r0 = self.net(rail0)
                feats_r1, _, hmap_r1 = self.net(rail1)

                if self.args.rail_hard_matching:
                    # Phase-1: hard topk + MNN (no coord gradient)
                    x0_f, x1_f, w = extract_xfeat_matches(
                        feats_r0[0], feats_r1[0], hmap_r0[0, 0], hmap_r1[0, 0],
                        topk=self.args.rail_topk, min_cos=self.args.rail_min_cos
                    )
                else:
                    # Phase-2: dual-softmax soft assignment (coord + weight grads)
                    x0_f, x1_f, w = extract_xfeat_matches_soft(
                        feats_r0[0], feats_r1[0], hmap_r0[0, 0], hmap_r1[0, 0],
                        topk=self.args.rail_topk, tau=self.args.rail_tau
                    )

                # Feature-map -> pixel coords
                sy0 = rail0.shape[-2] / feats_r0.shape[-2]
                sx0 = rail0.shape[-1] / feats_r0.shape[-1]
                sy1 = rail1.shape[-2] / feats_r1.shape[-2]
                sx1 = rail1.shape[-1] / feats_r1.shape[-1]
                x0_px = torch.stack([(x0_f[:, 0] + 0.5) * sx0 - 0.5, (x0_f[:, 1] + 0.5) * sy0 - 0.5], dim=1)
                x1_px = torch.stack([(x1_f[:, 0] + 0.5) * sx1 - 0.5, (x1_f[:, 1] + 0.5) * sy1 - 0.5], dim=1)

                rail_n_matches = int(len(w))
                if rail_n_matches >= self.args.rail_min_matches:
                    rail_loss, rail_dev, _, rail_reg, aux_param, _ = rail_self_supervision_loss(
                        mode=self.args.rail_mode,
                        x1_px=x0_px, x2_px=x1_px, w=w,
                        k0=self.rail_k0, k1=self.rail_k1,
                        img_h=rail0.shape[-2], img_w=rail0.shape[-1],
                        rho_eps=1e-6,
                        use_dev=(self.args.rail_lambda_dev > 0.0),
                        lambda_entropy=self.args.rail_lambda_entropy,
                        lambda_coverage=self.args.rail_lambda_coverage,
                        coverage_bins=self.args.rail_coverage_bins,
                        phi_min=self.args.rail_phi_min,
                        phi_max=self.args.rail_phi_max,
                        phi_grid_steps=self.args.rail_phi_grid_steps,
                        gn_steps=self.args.rail_gn_steps,
                        rail_radius=self.args.rail_radius,
                        r_cd=self.rail_r_cd,
                        t_cd=self.rail_t_cd,
                        lin_allow_both_signs=self.args.lin_allow_both_signs,
                    )
                    loss = loss + self.args.rail_lambda * rail_loss + self.args.rail_lambda_dev * rail_dev + rail_reg

            # Optim step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()

            if (i + 1) % self.args.save_ckpt_every == 0:
                torch.save(self.net.state_dict(),
                           os.path.join(self.args.ckpt_save_path, f"{self.args.training_type}_{i+1}.pth"))

            pbar.set_description(
                f"Loss {loss.item():.4f} acc_c0 {acc_coarse_0:.3f} acc_c1 {acc_coarse:.3f} acc_f {acc_coords:.3f} "
                f"#c {nb_coarse:d} rail {rail_loss.item():.4f} dev {rail_dev.item():.4f} "
                f"nm {rail_n_matches:d} aux {aux_param.item():.3f}"
            )
            pbar.update(1)

            self.writer.add_scalar('Loss/total', loss.item(), i)
            self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
            self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
            self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
            self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
            if self.rail_enabled:
                self.writer.add_scalar('Rail/J', rail_loss.item(), i)
                self.writer.add_scalar('Rail/dev', rail_dev.item(), i)
                self.writer.add_scalar('Rail/n_matches', rail_n_matches, i)
                self.writer.add_scalar('Rail/aux', aux_param.item(), i)

            if self.args.dry_run and i > 30:
                break


def main():
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()


#     python -m modules.training.train \
#   --training_type xfeat_default \
#   --megadepth_root_path /path/to/MegaDepth \
#   --synthetic_root_path /path/to/coco_20k \
#   --ckpt_save_path /tmp/out \
#   --rail_mode circular \
#   --rail_img0_path /abs/rail0.png --rail_img1_path /abs/rail1.png \
#   --rail_lambda 0.2 \
#   --rail_r_cd "r11,r12,r13,r21,r22,r23,r31,r32,r33" \
#   --rail_t_cd "tx,ty,tz" \
#   --rail_k0 "fx,0,cx,0,fy,cy,0,0,1" \
#   --rail_k1 "fx,0,cx,0,fy,cy,0,0,1"


#phase 2
# ## python -m modules.training.train --rail_lambda 1.0 --rail_tau 0.1 ...


#linear
# python -m modules.training.train \
#   --training_type xfeat_default \
#   --megadepth_root_path /path/to/MegaDepth \
#   --synthetic_root_path /path/to/coco_20k \
#   --ckpt_save_path /tmp/out \
#   --rail_mode linear \
#   --lin_allow_both_signs \
#   --rail_img0_path /abs/lin0.png --rail_img1_path /abs/lin1.png \
#   --rail_lambda 0.2 \
#   --rail_r_cd "r11,r12,r13,r21,r22,r23,r31,r32,r33" \
#   --rail_t_cd "tx,ty,tz" \
#   --rail_k0 "fx,0,cx,0,fy,cy,0,0,1" \
#   --rail_k1 "fx,0,cx,0,fy,cy,0,0,1"
