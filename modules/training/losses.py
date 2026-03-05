import torch
import torch.nn.functional as F

from modules.dataset.megadepth import megadepth_warper
from modules.training import utils
from third_party.alike_wrapper import extract_alike_kpts

"""
PHASE-2 (Soft-Assignment):
- Dual-softmax soft assignment: match coordinates are differentiable expectations
  over the assignment matrix P, so gradients flow through BOTH weights AND coordinates.
- P = softmax(S/tau, dim=1) * softmax(S/tau, dim=0),  S = D1 @ D2^T
- Soft match coordinate:  x_tilde_j = sum_j P_ij * x_j  (weighted average)
- Inner solver (phi*) still stop-grad for stability.
- Intrinsics K are REQUIRED for Essential-based losses (calibrated coordinates).
- Supports two manifolds:
    * circular rail  : 1D parameter phi
    * linear rail    : fixed orientation, sideways translation along +X_dev (optionally try sign +/-)
- Extrinsics are CAMERA->DEVICE:  p_dev = R_cd p_cam + t_cd
"""


# =========================
# Original XFeat losses
# =========================

def dual_softmax_loss(X, Y, temp=0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) / temp
    conf_matrix = F.softmax(dist_mat, 1) * F.softmax(dist_mat, 0)

    conf = torch.ones_like(torch.diag(dist_mat))
    loss = F.nll_loss(
        torch.log(conf_matrix + 1e-8),
        torch.arange(conf_matrix.shape[0], device=conf_matrix.device),
        reduction='none'
    )

    conf = conf / (conf.sum() + 1e-8)
    loss = (loss * conf).sum()
    return loss * 2., conf_matrix


def smooth_l1_loss(x, y, beta=1.0):
    n = torch.abs(x - y)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


def fine_loss(coords1, coords2, margin=1.0, alpha=0.5):
    dist = torch.norm(coords1 - coords2, dim=1)
    loss = smooth_l1_loss(dist, torch.zeros_like(dist), beta=margin)
    loss = alpha * (1.0 - torch.exp(-loss))
    return loss.mean()


def alike_distill_loss(im, kp_map, scores, device='cuda'):
    kp_alike = extract_alike_kpts(im, device=device)

    if len(kp_alike) < 1:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    kp_map = kp_map.squeeze(0)

    scores_flat = scores.view(-1)
    idx = scores_flat.topk(k=300, dim=0).indices
    H, W = scores.shape
    x = (idx % W).float()
    y = (idx // W).float()
    x = (x / (W - 1)) * 2.0 - 1.0
    y = (y / (H - 1)) * 2.0 - 1.0
    kp_xfeat = torch.stack([x, y], dim=1)

    dist = torch.cdist(kp_xfeat, kp_alike)
    min_dist = dist.min(dim=1).values

    loss = min_dist.mean()
    acc = (min_dist < 0.05).float().mean()
    return loss, acc


def coordinate_classification_loss(coords_logits, pts1, pts2, conf, bins=8):
    # pts are coarse coords on feature map; offsets in [-4,4] (assuming stride 8 logic)
    off = (pts2 - pts1).clamp(-4, 4).long() + 4
    labels = off[:, 1] * bins + off[:, 0]   # y*bins + x

    coords_log = F.log_softmax(coords_logits, dim=-1)
    loss = F.nll_loss(coords_log, labels, reduction='none')

    conf = conf / (conf.sum() + 1e-8)
    loss = (loss * conf).sum()

    pred = coords_log.argmax(dim=-1)
    acc = (pred == labels).float().mean()
    return loss * 2., acc


def keypoint_loss(heatmap, target):
    return F.l1_loss(heatmap, target) * 3.0


def hard_triplet_loss(X, Y, margin=0.5):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100. * torch.eye(*dist_mat.size(), dtype=dist_mat.dtype, device=dist_mat.device)

    dist_neg = dist_neg + dist_neg.le(0.01).float() * 100.
    hard_neg = torch.min(dist_neg, 1)[0]
    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)
    return loss.mean()


# =========================
# Geometry helpers
# =========================

def _skew(t):
    t = t.view(3)
    tx, ty, tz = t[0], t[1], t[2]
    z = torch.zeros_like(tx)
    return torch.stack([
        torch.stack([z, -tz, ty]),
        torch.stack([tz, z, -tx]),
        torch.stack([-ty, tx, z]),
    ], dim=0)


def _rot_z(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([
        torch.stack([c, -s, z]),
        torch.stack([s,  c, z]),
        torch.stack([z,  z, o]),
    ], dim=0)


def _to_h(x):
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=1)


def normalize_points_with_K(x_px, K):
    """x_px (N,2) pixels -> calibrated homogeneous (N,3) via inv(K)."""
    K = K.to(device=x_px.device, dtype=x_px.dtype).view(3, 3)
    x_h = _to_h(x_px)
    x_n = (torch.linalg.inv(K) @ x_h.t()).t()
    return x_n


def sampson_error(x1n_h, x2n_h, E, eps=1e-8):
    Ex1 = (E @ x1n_h.t()).t()
    Etx2 = (E.t() @ x2n_h.t()).t()
    x2tEx1 = torch.sum(x2n_h * Ex1, dim=1)
    denom = Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2
    return (x2tEx1**2) / (denom + eps)


def robust_charbonnier(x, eps=1e-6):
    return torch.sqrt(x + eps*eps)


def weighted_8point_essential(x1n_h, x2n_h, w, eps=1e-8):
    """Weighted 8-point Essential estimate in calibrated coords. Autograd-safe."""
    N = x1n_h.shape[0]
    if N < 8:
        return torch.eye(3, device=x1n_h.device, dtype=x1n_h.dtype)

    w = torch.clamp(w, min=0.0)
    sw = torch.sqrt(w + eps).view(-1, 1)

    X1, Y1, Z1 = x1n_h[:, 0], x1n_h[:, 1], x1n_h[:, 2]
    X2, Y2, Z2 = x2n_h[:, 0], x2n_h[:, 1], x2n_h[:, 2]
    A = torch.stack([
        X2*X1, X2*Y1, X2*Z1,
        Y2*X1, Y2*Y1, Y2*Z1,
        Z2*X1, Z2*Y1, Z2*Z1
    ], dim=1)
    A = A * sw

    _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    E = Vh[-1, :].view(3, 3)

    # Essential projection: (s, s, 0)
    U, S, VhE = torch.linalg.svd(E)
    s = 0.5 * (S[0] + S[1])
    S_new = torch.stack([s, s, torch.zeros_like(s)])
    E = U @ torch.diag(S_new) @ VhE
    E = E / (E.norm() + eps)
    return E


# =========================
# Extrinsics conversion (CAM->DEV given)
# =========================

def _cam_to_dev_to_dev_to_cam(r_cd, t_cd):
    """
    Given camera->device: p_dev = R_cd p_cam + t_cd,
    compute device->camera: p_cam = R_dc p_dev + t_dc.
    """
    r_cd = r_cd.view(3, 3)
    t_cd = t_cd.view(3)
    r_dc = r_cd.t()
    t_dc = -r_dc @ t_cd
    return r_dc, t_dc


# =========================
# Circular rail manifold
# =========================

def essential_from_circular_phi(phi, rail_radius, r_cd, t_cd):
    """
    Circular rail in device frame, extrinsics are CAM->DEV (R_cd, t_cd).
    Internally uses DEV->CAM (R_dc, t_dc).

    E(phi) = [t_ji]_x R_ji with:
      R_ji = R_dc^T Rz(-phi) R_dc
      t_ji = R_dc^T( Rz(-phi)p0 - p0 + (Rz(-phi)-I)t_dc ), p0=[R,0,0]^T
    """
    device, dtype = phi.device, phi.dtype
    r_cd = r_cd.to(device=device, dtype=dtype).view(3, 3)
    t_cd = t_cd.to(device=device, dtype=dtype).view(3)
    r_dc, t_dc = _cam_to_dev_to_dev_to_cam(r_cd, t_cd)

    rz = _rot_z(-phi)
    r_ji = r_dc.t() @ rz @ r_dc

    p0 = torch.tensor([rail_radius, 0.0, 0.0], device=device, dtype=dtype)
    I = torch.eye(3, device=device, dtype=dtype)
    t_ji = r_dc.t() @ (rz @ p0 - p0 + (rz - I) @ t_dc)

    return _skew(t_ji) @ r_ji


def _circular_objective(phi, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd):
    E = essential_from_circular_phi(phi, rail_radius, r_cd, t_cd)
    r = sampson_error(x1n_h, x2n_h, E)
    return torch.sum(w * robust_charbonnier(r, eps=rho_eps))


@torch.no_grad()
def solve_phi_circular(x1n_h, x2n_h, w,
                       phi_min, phi_max,
                       coarse_steps=41, gn_steps=3,
                       rho_eps=1e-6,
                       rail_radius=1.0,
                       r_cd=None, t_cd=None):
    device, dtype = x1n_h.device, x1n_h.dtype
    grid = torch.linspace(phi_min, phi_max, steps=int(coarse_steps), device=device, dtype=dtype)
    costs = torch.stack([_circular_objective(phi, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd) for phi in grid])
    phi = grid[torch.argmin(costs)].clone()

    for _ in range(max(int(gn_steps), 0)):
        delta = torch.tensor(1e-3, device=device, dtype=dtype)
        c0 = _circular_objective(phi, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd)
        c1 = _circular_objective(phi + delta, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd)
        c2 = _circular_objective(phi - delta, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd)
        g = (c1 - c2) / (2.0 * delta)
        h = (c1 - 2.0*c0 + c2) / (delta*delta)
        step = g / (h + 1e-6)
        phi = torch.clamp(phi - step, min=float(phi_min), max=float(phi_max))

    E = essential_from_circular_phi(phi, rail_radius, r_cd, t_cd)
    J = _circular_objective(phi, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd)
    return phi, E, J


# =========================
# Linear rail manifold (fixed orientation, sideways along +X_dev)
# =========================

def essential_from_linear_sideways(r_cd, sign=+1.0):
    """
    Linear rail, fixed orientation:
      R_ji = I
      t_dev ∝ sign * [1,0,0]^T

    With CAM->DEV rotation R_cd:
      t_cam ∝ R_cd * t_dev   (since t_cam = R_dc^T t_dev and R_dc = R_cd^T => t_cam = R_cd t_dev)
      E = [t_cam]_x
    """
    device, dtype = r_cd.device, r_cd.dtype
    r_cd = r_cd.to(device=device, dtype=dtype).view(3, 3)
    t_dev = torch.tensor([sign, 0.0, 0.0], device=device, dtype=dtype)
    t_cam = r_cd @ t_dev
    return _skew(t_cam)


def _linear_objective(sign, x1n_h, x2n_h, w, rho_eps, r_cd):
    E = essential_from_linear_sideways(r_cd, sign=sign)
    r = sampson_error(x1n_h, x2n_h, E)
    return torch.sum(w * robust_charbonnier(r, eps=rho_eps))


@torch.no_grad()
def solve_linear_sign(x1n_h, x2n_h, w, rho_eps=1e-6, r_cd=None, allow_both=True):
    if not allow_both:
        sign = torch.tensor(1.0, device=x1n_h.device, dtype=x1n_h.dtype)
        E = essential_from_linear_sideways(r_cd, sign=1.0)
        J = _linear_objective(1.0, x1n_h, x2n_h, w, rho_eps, r_cd)
        return sign, E, J

    Jp = _linear_objective(1.0, x1n_h, x2n_h, w, rho_eps, r_cd)
    Jm = _linear_objective(-1.0, x1n_h, x2n_h, w, rho_eps, r_cd)
    if Jp <= Jm:
        sign = torch.tensor(1.0, device=x1n_h.device, dtype=x1n_h.dtype)
        E = essential_from_linear_sideways(r_cd, sign=1.0)
        return sign, E, Jp
    sign = torch.tensor(-1.0, device=x1n_h.device, dtype=x1n_h.dtype)
    E = essential_from_linear_sideways(r_cd, sign=-1.0)
    return sign, E, Jm


# =========================
# Match extraction — Phase 1 (hard coords, differentiable weights)  [KEPT FOR REFERENCE]
# =========================

def extract_xfeat_matches(f1, f2, h1, h2, topk=1024, min_cos=0.1):
    """
    Phase-1 hard extraction. Kept for compatibility / evaluation.
    f1,f2: [C,H,W]
    h1,h2: [H,W]
    Returns:
      x1, x2: [M,2] feature-map coords (float32) [x,y]
      w: [M] weights, normalized to sum 1 (NOT detached)
    """
    C, H, W = f1.shape
    n = H * W
    k = min(int(topk), n)
    if k <= 0:
        empty = torch.empty((0, 2), device=f1.device, dtype=torch.float32)
        return empty, empty, torch.empty((0,), device=f1.device, dtype=f1.dtype)

    d1 = F.normalize(f1.view(C, -1).t(), dim=-1)
    d2 = F.normalize(f2.view(C, -1).t(), dim=-1)
    r1 = h1.reshape(-1)
    r2 = h2.reshape(-1)

    i1 = torch.topk(r1, k=k, dim=0).indices
    i2 = torch.topk(r2, k=k, dim=0).indices
    d1k, d2k = d1[i1], d2[i2]
    r1k, r2k = r1[i1], r2[i2]

    S = d1k @ d2k.t()
    nn12 = torch.argmax(S, dim=1)
    nn21 = torch.argmax(S, dim=0)
    ids = torch.arange(k, device=f1.device)
    mutual = nn21[nn12] == ids
    if int(mutual.sum()) == 0:
        empty = torch.empty((0, 2), device=f1.device, dtype=torch.float32)
        return empty, empty, torch.empty((0,), device=f1.device, dtype=f1.dtype)

    src = ids[mutual]
    tgt = nn12[mutual]
    cos = S[src, tgt]
    keep = cos > float(min_cos)
    if int(keep.sum()) == 0:
        empty = torch.empty((0, 2), device=f1.device, dtype=torch.float32)
        return empty, empty, torch.empty((0,), device=f1.device, dtype=f1.dtype)

    src = src[keep]
    tgt = tgt[keep]
    cos = cos[keep]

    lin1 = i1[src]
    lin2 = i2[tgt]

    x1 = torch.stack([lin1 % W, lin1 // W], dim=1).to(torch.float32)
    x2 = torch.stack([lin2 % W, lin2 // W], dim=1).to(torch.float32)

    w_out = torch.clamp(cos, min=0.0) * torch.clamp(r1k[src], min=0.0) * torch.clamp(r2k[tgt], min=0.0)
    w_out = torch.clamp(w_out, min=0.0)
    w_out = w_out / (w_out.sum() + 1e-8)
    return x1, x2, w_out


# =========================
# Match extraction — Phase 2 (soft-assignment, fully differentiable)
# =========================

def extract_xfeat_matches_soft(f1, f2, h1, h2, topk=1024, tau=0.1, dust_bin=True):
    """
    Phase-2 soft-assignment extraction.  Gradients flow through BOTH
    match coordinates (via the soft expectation) AND match weights.

    f1,f2 : [C, H, W]  — dense descriptor feature maps
    h1,h2 : [H, W]     — heatmaps / reliability maps
    topk  : int         — number of source keypoints (selected by heatmap score)
    tau   : float       — softmax temperature (lower = sharper; 0.1 is a good start)
    dust_bin : bool     — if True, append a learned-free dustbin row/col so that poor
                          matches can be "explained away" instead of forced onto a target.

    Returns
    -------
    x1_soft : [M, 2]   soft source coords in feature-map space (x, y).  These are the
                        *hard* topk locations (no grad through them — they define the
                        query set).
    x2_soft : [M, 2]   soft target coords — differentiable expectations over all
                        candidate locations in image 2, weighted by the assignment row.
    w       : [M]      match weight per pair (normalized, has grad).
    """
    C, H, W = f1.shape
    n = H * W
    k = min(int(topk), n)
    if k <= 0:
        empty = torch.empty((0, 2), device=f1.device, dtype=torch.float32)
        return empty, empty, torch.empty((0,), device=f1.device, dtype=f1.dtype)

    # -- Dense descriptors (L2-normalised) & heatmap scores ---------------
    d1 = F.normalize(f1.view(C, -1).t(), dim=-1)   # [N1, C]
    d2 = F.normalize(f2.view(C, -1).t(), dim=-1)   # [N2, C]
    r1 = h1.reshape(-1)                              # [N1]
    r2 = h2.reshape(-1)                              # [N2]

    # -- Select top-k keypoints from heatmaps ----------------------------
    i1 = torch.topk(r1, k=k, dim=0).indices          # [k]
    i2 = torch.topk(r2, k=k, dim=0).indices          # [k]
    d1k = d1[i1]                                      # [k, C]
    d2k = d2[i2]                                      # [k, C]
    r1k = r1[i1]                                      # [k]
    r2k = r2[i2]                                      # [k]

    # -- Cosine similarity matrix ----------------------------------------
    S = d1k @ d2k.t()                                 # [k, k]

    # -- Dual softmax (transport-like) assignment -------------------------
    # P_ij = softmax(S/tau, dim=1)_ij  *  softmax(S/tau, dim=0)_ij
    logits = S / tau
    if dust_bin:
        # Append a dustbin column and row with score 0 (acts as "unmatched").
        # This lets the softmax assign low-confidence queries to the dustbin
        # instead of forcing them onto a real target.
        dust_col = torch.zeros(k, 1, device=S.device, dtype=S.dtype)
        dust_row = torch.zeros(1, k + 1, device=S.device, dtype=S.dtype)
        logits_aug = torch.cat([logits, dust_col], dim=1)      # [k, k+1]
        logits_aug = torch.cat([logits_aug, dust_row], dim=0)  # [k+1, k+1]
        P_aug = F.softmax(logits_aug, dim=1) * F.softmax(logits_aug, dim=0)
        P = P_aug[:k, :k]                                      # [k, k]
    else:
        P = F.softmax(logits, dim=1) * F.softmax(logits, dim=0)  # [k, k]

    # -- Candidate coordinates in feature-map space ----------------------
    # x2_cand[j] = (ix, iy) for the j-th selected keypoint in image 2
    x2_cand = torch.stack([i2 % W, i2 // W], dim=1).to(f1.dtype)  # [k, 2]

    # -- Soft target coordinates: expectation over assignment row ---------
    # x2_soft[i] = sum_j  P[i,j] * x2_cand[j]  /  sum_j P[i,j]
    row_sum = P.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [k, 1]
    P_norm = P / row_sum                                   # [k, k]
    x2_soft = P_norm @ x2_cand                             # [k, 2]  <-- differentiable!

    # -- Source coordinates (hard, defines query set) --------------------
    x1_hard = torch.stack([i1 % W, i1 // W], dim=1).to(torch.float32)  # [k, 2]

    # -- Match weights: assignment confidence * heatmap scores -----------
    # confidence = max assignment value per row (how peaked the assignment is)
    conf = P.max(dim=1).values                              # [k]
    w_out = torch.clamp(conf, min=0.0) * torch.clamp(r1k, min=0.0) * torch.clamp(r2k[P.argmax(dim=1)], min=0.0)
    w_out = torch.clamp(w_out, min=0.0)
    w_out = w_out / (w_out.sum() + 1e-8)

    return x1_hard, x2_soft, w_out


# =========================
# Regularizers (correct sign)
# =========================

def _weighted_entropy(w, eps=1e-8):
    w = torch.clamp(w, min=0.0)
    p = w / (w.sum() + eps)
    p = torch.clamp(p, min=eps)
    return -torch.sum(p * torch.log(p))


def _coverage_entropy(x_px, w, img_h, img_w, bins=8, eps=1e-8):
    if x_px.numel() == 0:
        return torch.zeros((), device=x_px.device, dtype=x_px.dtype)
    bins = max(int(bins), 2)
    xn = torch.clamp(x_px[:, 0] / max(img_w - 1, 1), 0.0, 1.0)
    yn = torch.clamp(x_px[:, 1] / max(img_h - 1, 1), 0.0, 1.0)
    ix = torch.clamp((xn * bins).long(), 0, bins-1)
    iy = torch.clamp((yn * bins).long(), 0, bins-1)
    idx = iy * bins + ix
    occ = torch.zeros((bins*bins,), device=x_px.device, dtype=x_px.dtype)
    occ = occ.index_add(0, idx, torch.clamp(w, min=0.0))
    occ = occ / (occ.sum() + eps)
    occ = torch.clamp(occ, min=eps)
    return -torch.sum(occ * torch.log(occ))


# =========================
# Unified rail loss API (mode: circular or linear)
# =========================

def rail_self_supervision_loss(
    *,
    mode: str,
    x1_px: torch.Tensor,
    x2_px: torch.Tensor,
    w: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    img_h: int,
    img_w: int,
    rho_eps: float = 1e-6,
    use_dev: bool = False,
    lambda_entropy: float = 0.0,
    lambda_coverage: float = 0.0,
    coverage_bins: int = 8,
    # circular params
    phi_min: float = -0.35,
    phi_max: float = 0.35,
    phi_grid_steps: int = 41,
    gn_steps: int = 3,
    rail_radius: float = 1.0,
    r_cd: torch.Tensor = None,
    t_cd: torch.Tensor = None,
    # linear params
    lin_allow_both_signs: bool = True,
):
    """
    Returns:
      J_manifold, L_dev, J_free, reg, aux_param, E_manifold

    aux_param:
      - circular: phi*
      - linear: sign (+1/-1)
    """
    if k0 is None or k1 is None:
        raise ValueError("[RailLoss] Intrinsics k0 and k1 are required for Essential-based rail losses.")

    x1n_h = normalize_points_with_K(x1_px, k0)
    x2n_h = normalize_points_with_K(x2_px, k1)

    mode_l = mode.lower()
    if mode_l in ("circular", "circle", "circ"):
        if r_cd is None or t_cd is None:
            raise ValueError("[RailLoss] circular mode requires r_cd and t_cd (camera->device extrinsics).")
        aux, E_m, J_m = solve_phi_circular(
            x1n_h=x1n_h, x2n_h=x2n_h, w=w,
            phi_min=phi_min, phi_max=phi_max,
            coarse_steps=phi_grid_steps, gn_steps=gn_steps,
            rho_eps=rho_eps, rail_radius=rail_radius,
            r_cd=r_cd, t_cd=t_cd
        )
    elif mode_l in ("linear", "lin"):
        if r_cd is None:
            raise ValueError("[RailLoss] linear mode requires r_cd (camera->device rotation).")
        aux, E_m, J_m = solve_linear_sign(
            x1n_h=x1n_h, x2n_h=x2n_h, w=w,
            rho_eps=rho_eps, r_cd=r_cd,
            allow_both=lin_allow_both_signs
        )
    else:
        raise ValueError(f"[RailLoss] Unknown mode: {mode}. Use 'circular' or 'linear'.")

    # Optional free fit (weighted 8-point)
    if use_dev:
        E_free = weighted_8point_essential(x1n_h, x2n_h, w)
        J_free = torch.sum(w * robust_charbonnier(sampson_error(x1n_h, x2n_h, E_free), eps=rho_eps))
        L_dev = torch.log(J_m + 1e-8) - torch.log(J_free + 1e-8)
    else:
        J_free = torch.zeros_like(J_m)
        L_dev = torch.zeros_like(J_m)

    # Regularizers (subtract entropy => encourages spread)
    reg = torch.zeros_like(J_m)
    if lambda_entropy > 0.0:
        reg = reg - float(lambda_entropy) * _weighted_entropy(w)
    if lambda_coverage > 0.0:
        reg = reg - float(lambda_coverage) * _coverage_entropy(
            x1_px, w, img_h=img_h, img_w=img_w, bins=coverage_bins
        )

    return J_m, L_dev, J_free, reg, aux, E_m