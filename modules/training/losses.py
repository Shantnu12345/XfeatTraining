import torch
import torch.nn.functional as F
import math

from modules.dataset.megadepth import megadepth_warper
from modules.training import utils
from third_party.alike_wrapper import extract_alike_kpts


# =========================
# Rail / Manifold utilities
# =========================

def _skew(v: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric matrix [v]_x such that [v]_x a = v x a."""
    z = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def _rot_z(phi: torch.Tensor, device, dtype) -> torch.Tensor:
    """Rotation about Z axis."""
    c = torch.cos(phi)
    s = torch.sin(phi)
    R = torch.zeros((3, 3), device=device, dtype=dtype)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    R[2, 2] = 1.0
    return R


def _to_h(x: torch.Tensor) -> torch.Tensor:
    """[N,2] -> [N,3] homogeneous."""
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=1)


def normalize_points_with_K(x_pix: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Normalize pixel coordinates using intrinsics:
      x_n = inv(K) * [u v 1]^T  -> return [x y] (inhomogeneous)
    x_pix: [N,2]
    K: [3,3]
    """
    xh = _to_h(x_pix)  # [N,3]
    invK = torch.linalg.inv(K)
    xn = (invK @ xh.t()).t()  # [N,3]
    return xn[:, :2] / xn[:, 2:3].clamp_min(1e-12)


def sampson_error(x1n: torch.Tensor, x2n: torch.Tensor, E: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Sampson distance for essential matrix E.
    x1n, x2n: [N,2] normalized camera coordinates
    E: [3,3]
    returns: [N]
    """
    x1h = _to_h(x1n)
    x2h = _to_h(x2n)
    Ex1 = (E @ x1h.t()).t()
    Etx2 = (E.t() @ x2h.t()).t()
    x2tEx1 = torch.sum(x2h * Ex1, dim=-1)
    denom = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2 + eps
    return (x2tEx1 ** 2) / denom


def robust_charbonnier(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier penalty: sqrt(r + eps^2)."""
    return torch.sqrt(r + eps * eps)


def essential_from_rail_phi(phi: torch.Tensor, rail_radius: float = 1.0, r_dc=None, t_dc=None) -> torch.Tensor:
    """
    Circular rail model with known device->camera extrinsics.
    E(da) = [t_ji(da)]_x R_ji(da), where:
      R_ji = R_dc^T Rz(-da) R_dc
      t_ji = R_dc^T( Rz(-da)p0 - p0 + (Rz(-da)-I)t_dc ), p0=[R,0,0]^T

    Convention:
      - device moves on circle in XY plane about +Z
      - device->camera rotation r_dc (3x3) and translation t_dc (3,)
      - relative motion from i to j uses -phi (consistent with earlier codex implementation)
    """
    device, dtype = phi.device, phi.dtype
    if r_dc is None:
        r_dc = torch.eye(3, device=device, dtype=dtype)
    else:
        r_dc = r_dc.to(device=device, dtype=dtype)
    if t_dc is None:
        t_dc = torch.zeros((3,), device=device, dtype=dtype)
    else:
        t_dc = t_dc.to(device=device, dtype=dtype).view(3)

    rz_neg = _rot_z(-phi, device, dtype)
    r_ji = r_dc.t() @ rz_neg @ r_dc

    p0 = torch.tensor([rail_radius, 0.0, 0.0], device=device, dtype=dtype)
    I = torch.eye(3, device=device, dtype=dtype)
    t_ji = r_dc.t() @ (rz_neg @ p0 - p0 + (rz_neg - I) @ t_dc)

    return _skew(t_ji) @ r_ji


def weighted_8point_essential(x1n: torch.Tensor, x2n: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Weighted 8-point estimate of an Essential/Fundamental-like matrix in normalized camera coordinates.

    Notes:
      • This is primarily used as a *reference* 'best unconstrained epipolar fit' (E_free).
      • We project to the Essential manifold by enforcing singular values (s, s, 0) without in-place ops
        (important for autograd stability).
    """
    if x1n.ndim != 2 or x2n.ndim != 2 or x1n.shape[1] != 2 or x2n.shape[1] != 2:
        raise ValueError("x1n and x2n must be [N,2] tensors.")
    if w.ndim != 1 or w.shape[0] != x1n.shape[0]:
        raise ValueError("w must be [N] and match x1n length.")

    x1, y1 = x1n[:, 0], x1n[:, 1]
    x2, y2 = x2n[:, 0], x2n[:, 1]

    # Design matrix A (N x 9)
    A = torch.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, torch.ones_like(x1)
    ], dim=1)

    # Normalize weights and apply sqrt for weighted least squares.
    w_norm = w / (w.sum() + eps)
    Aw = A * (w_norm.clamp_min(0.0).sqrt().unsqueeze(1))

    # Solve Aw e = 0 via SVD -> last right-singular vector.
    _, _, Vh = torch.linalg.svd(Aw, full_matrices=False)
    e = Vh[-1]
    E = e.view(3, 3)

    # Project to Essential manifold: singular values (s, s, 0)
    U, S, Vh_e = torch.linalg.svd(E)
    s = 0.5 * (S[0] + S[1])
    S_new = torch.stack([s, s, torch.zeros_like(s)])
    E_proj = U @ torch.diag(S_new) @ Vh_e
    return E_proj


def _rail_objective(phi: torch.Tensor,
                    x1n: torch.Tensor,
                    x2n: torch.Tensor,
                    w: torch.Tensor,
                    rho_eps: float,
                    rail_radius: float,
                    r_dc,
                    t_dc) -> torch.Tensor:
    E = essential_from_rail_phi(phi, rail_radius=rail_radius, r_dc=r_dc, t_dc=t_dc)
    r = sampson_error(x1n, x2n, E)
    return (w * robust_charbonnier(r, eps=rho_eps)).sum()


def solve_phi_rail(x1n: torch.Tensor,
                   x2n: torch.Tensor,
                   w: torch.Tensor,
                   phi_min: float = -0.35,
                   phi_max: float = 0.35,
                   coarse_steps: int = 41,
                   gn_steps: int = 3,
                   rho_eps: float = 1e-6,
                   rail_radius: float = 1.0,
                   r_dc=None,
                   t_dc=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Coarse 1D search + a few GN-like steps (finite-diff) to find phi*.
    For stability we STOP-GRAD through the solver by detaching phi each step.
    """
    device, dtype = x1n.device, x1n.dtype
    # Coarse search
    phis = torch.linspace(phi_min, phi_max, steps=coarse_steps, device=device, dtype=dtype)
    costs = []
    for p in phis:
        costs.append(_rail_objective(p, x1n, x2n, w, rho_eps, rail_radius, r_dc, t_dc))
    costs = torch.stack(costs)
    best_idx = torch.argmin(costs)
    phi = phis[best_idx].detach()

    # Small GN refinement (finite differences)
    for _ in range(max(0, int(gn_steps))):
        phi = phi.detach()
        h = torch.tensor(1e-3, device=device, dtype=dtype)
        f0 = _rail_objective(phi, x1n, x2n, w, rho_eps, rail_radius, r_dc, t_dc)
        f1 = _rail_objective(phi + h, x1n, x2n, w, rho_eps, rail_radius, r_dc, t_dc)
        f2 = _rail_objective(phi - h, x1n, x2n, w, rho_eps, rail_radius, r_dc, t_dc)

        # 2nd order approximation: f'(phi) ~ (f1 - f2)/(2h), f''(phi) ~ (f1 - 2f0 + f2)/h^2
        g = (f1 - f2) / (2.0 * h)
        H = (f1 - 2.0 * f0 + f2) / (h * h)
        step = g / (H.abs() + 1e-6)  # damped
        phi = (phi - step).clamp(min=phi_min, max=phi_max)

    E = essential_from_rail_phi(phi, rail_radius=rail_radius, r_dc=r_dc, t_dc=t_dc)
    j = _rail_objective(phi, x1n, x2n, w, rho_eps, rail_radius, r_dc, t_dc)
    return phi, E, j


def _weighted_entropy_regularizer(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Encourage non-degenerate distribution over matches."""
    p = (w / (w.sum() + eps)).clamp_min(eps)
    return -(p * torch.log(p)).sum()


def _weighted_coverage_regularizer(x1_pix: torch.Tensor,
                                  w: torch.Tensor,
                                  h: int,
                                  w_img: int,
                                  bins: int = 8,
                                  eps: float = 1e-8) -> torch.Tensor:
    """
    Encourage spatial coverage of matches by penalizing concentration into few bins.
    Uses a soft histogram over bins in image space (x1 only).
    """
    # Bin coords
    bx = torch.clamp((x1_pix[:, 0] / max(w_img, 1) * bins).long(), 0, bins - 1)
    by = torch.clamp((x1_pix[:, 1] / max(h, 1) * bins).long(), 0, bins - 1)
    idx = by * bins + bx
    nb = bins * bins
    hist = torch.zeros((nb,), device=x1_pix.device, dtype=x1_pix.dtype)
    hist.index_add_(0, idx, w)
    p = (hist / (hist.sum() + eps)).clamp_min(eps)
    return -(p * torch.log(p)).sum()


def extract_xfeat_matches(feats0: torch.Tensor,
                          feats1: torch.Tensor,
                          hmap0: torch.Tensor,
                          hmap1: torch.Tensor,
                          topk: int = 1024,
                          min_cos: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pseudo-matches from dense features:
      - take topk most reliable points in each image using heatmap hmap
      - mutual nearest neighbor in descriptor space
      - return coords in FEATURE MAP coordinates (not pixels): [N,2] (x,y) and weights [N]
    This is NOT fully differentiable (uses topk/argmax), which is OK for v1.
    """
    # hmap: [H,W]
    H0, W0 = hmap0.shape[-2], hmap0.shape[-1]
    H1, W1 = hmap1.shape[-2], hmap1.shape[-1]

    # pick topk locations by heatmap
    k0 = min(int(topk), H0 * W0)
    k1 = min(int(topk), H1 * W1)

    v0, idx0 = torch.topk(hmap0.reshape(-1), k=k0, largest=True)
    v1, idx1 = torch.topk(hmap1.reshape(-1), k=k1, largest=True)

    y0 = (idx0 // W0).long()
    x0 = (idx0 - y0 * W0).long()
    y1 = (idx1 // W1).long()
    x1 = (idx1 - y1 * W1).long()

    d0 = feats0[:, y0, x0].t()  # [k0, C]
    d1 = feats1[:, y1, x1].t()  # [k1, C]
    d0 = F.normalize(d0, dim=-1)
    d1 = F.normalize(d1, dim=-1)

    # Similarity and mutual NN
    sim = d0 @ d1.t()  # [k0,k1]
    nn01 = torch.argmax(sim, dim=1)  # [k0]
    nn10 = torch.argmax(sim, dim=0)  # [k1]
    ids0 = torch.arange(k0, device=feats0.device)
    mutual = ids0 == nn10[nn01]

    ids0 = ids0[mutual]
    ids1 = nn01[mutual]

    if ids0.numel() == 0:
        return (torch.zeros((0, 2), device=feats0.device, dtype=torch.float32),
                torch.zeros((0, 2), device=feats0.device, dtype=torch.float32),
                torch.zeros((0,), device=feats0.device, dtype=torch.float32))

    # cosine filter
    cos = sim[ids0, ids1]
    keep = cos >= float(min_cos)
    ids0 = ids0[keep]
    ids1 = ids1[keep]
    cos = cos[keep]

    x0m = x0[ids0].to(torch.float32)
    y0m = y0[ids0].to(torch.float32)
    x1m = x1[ids1].to(torch.float32)
    y1m = y1[ids1].to(torch.float32)

    # weights from similarity * reliability
    w0 = v0[ids0].clamp_min(0.0)
    w1v = v1[ids1].clamp_min(0.0)
    w = (cos.clamp_min(0.0) * (w0 * w1v).sqrt())
    w = w.clamp_min(0.0)
    w = w / (w.sum() + 1e-8)

    p0 = torch.stack([x0m, y0m], dim=1)
    p1 = torch.stack([x1m, y1m], dim=1)
    return p0, p1, w


def rail_self_supervision_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w: torch.Tensor,
    img_h: int,
    img_w: int,
    phi_min: float = -0.35,
    phi_max: float = 0.35,
    phi_grid_steps: int = 41,
    gn_steps: int = 3,
    use_dev: bool = False,
    rho_eps: float = 1e-6,
    lambda_entropy: float = 0.0,
    lambda_coverage: float = 0.0,
    coverage_bins: int = 8,
    rail_radius: float = 1.0,
    r_dc=None,
    t_dc=None,
    k0=None,
    k1=None,
    eps: float = 1e-8,
):
    """
    Returns:
      j_rail, l_dev, j_free, reg_w, phi_star, E_rail
    """
    # Normalize match weights
    w = torch.clamp(w, min=0.0)
    w = w / (w.sum() + eps)

    # Normalize points
    if k0 is not None and k1 is not None:
        k0 = k0.to(device=x1.device, dtype=x1.dtype)
        k1 = k1.to(device=x1.device, dtype=x1.dtype)
        x1n = normalize_points_with_K(x1, k0)
        x2n = normalize_points_with_K(x2, k1)
    else:
        raise ValueError('rail_self_supervision_loss requires calibrated intrinsics k0 and k1 (Essential matrix geometry). Provide K for both images.')

    phi_star, E_rail, j_rail = solve_phi_rail(
        x1n=x1n, x2n=x2n, w=w,
        phi_min=phi_min, phi_max=phi_max,
        coarse_steps=phi_grid_steps, gn_steps=gn_steps, rho_eps=rho_eps,
        rail_radius=rail_radius, r_dc=r_dc, t_dc=t_dc
    )

    # Optional "free vs rail" deviation
    if use_dev:
        E_free = weighted_8point_essential(x1n, x2n, w)
        j_free = (w * robust_charbonnier(sampson_error(x1n, x2n, E_free), eps=rho_eps)).sum()
        l_dev = torch.log(j_rail + eps) - torch.log(j_free + eps)
    else:
        E_free = None
        j_free = torch.zeros_like(j_rail)
        l_dev = torch.zeros_like(j_rail)

    # Regularizers on match weights / coverage
    reg_w = torch.zeros_like(j_rail)
    if lambda_entropy > 0:
        reg_w = reg_w - lambda_entropy * _weighted_entropy_regularizer(w)
    if lambda_coverage > 0:
        reg_w = reg_w - lambda_coverage * _weighted_coverage_regularizer(
            x1, w, h=int(img_h), w_img=int(img_w), bins=int(coverage_bins)
        )

    return j_rail, l_dev, j_free, reg_w, phi_star, E_rail


# =========================
# Original XFeat losses
# =========================

def dual_softmax_loss(X, Y, temp=0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
        conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device=X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf.detach()


def smooth_l1_loss(d, x, y, delta=1.0):
    # Smooth L1 Loss
    diff = d - (x - y)
    abs_diff = torch.abs(diff)
    loss = torch.where(abs_diff < delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    return loss.mean()


def fine_loss(p1, p2, fine_pred, delta=1.0):
    # Fine loss based on Smooth L1 for both x and y
    loss_x = smooth_l1_loss(fine_pred[:, 0], p2[:, 0], p1[:, 0], delta)
    loss_y = smooth_l1_loss(fine_pred[:, 1], p2[:, 1], p1[:, 1], delta)
    return (loss_x + loss_y) / 2


def alike_distill_loss(kpts, img, kp_th=0.1, top_k=2000):
    """Distillation loss using ALIKE keypoint detector."""
    with torch.inference_mode():
        b = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
        pts = extract_alike_kpts(b, top_k=top_k, kp_th=kp_th)  # Nx2
        if pts is None or len(pts) == 0:
            return torch.zeros((), device=img.device), 0.0
        pts = torch.tensor(pts, device=img.device, dtype=torch.long)
        pts = pts.clamp(min=0, max=min(kpts.shape[-2], kpts.shape[-1]) - 1)

    # kpts: [1, H, W]
    pred = kpts[0, pts[:, 1], pts[:, 0]]
    loss = (1.0 - pred).mean()
    acc = (pred > 0.5).float().mean().item()
    return loss, acc


def keypoint_position_loss(pred_kpts, gt_kpts):
    # pred_kpts: [1,H,W], gt_kpts: [N,2] (x,y)
    gt_vals = pred_kpts[0, gt_kpts[:, 1].long(), gt_kpts[:, 0].long()]
    return (1.0 - gt_vals).mean()


def coordinate_classification_loss(coords, pts1, pts2, conf, bins=8):
    """
    coords: predicted logits for offset bins
    pts1/pts2: [N,2] coarse coords
    conf: [N] confidence
    """
    # offsets in [-4,4] range (because stride=8)
    off = (pts2 - pts1).clamp(-4, 4).long() + 4
    # bin index
    tgt = off[:, 1] * bins + off[:, 0]  # y * bins + x
    loss = F.cross_entropy(coords, tgt, reduction='none')
    loss = (loss * conf).mean()

    pred = torch.argmax(coords, dim=1)
    acc = (pred == tgt).float().mean().item()
    return loss, acc


def keypoint_loss(h, conf):
    # reliability loss encourages high heatmap at correspondence points
    return ((1.0 - h) * conf).mean()


def hard_triplet_loss(x, y, margin=1.0):
    # Triplet loss (unused by default training)
    dist_pos = 1.0 - (x * y).sum(-1)
    dist_neg = 1.0 - (x @ y.t()).max(dim=1)[0]
    return F.relu(dist_pos - dist_neg + margin).mean()