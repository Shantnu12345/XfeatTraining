import torch
import torch.nn.functional as F
import numpy as _np
import sys as _sys
import os as _os
from modules.dataset.megadepth import megadepth_warper
from modules.trainz̄ing import utils
# from third_party.alike_wrapper import extract_alike_kpts
"""
Rail self-supervision for circular / linear rail manifolds.
Extrinsics are CAMERA->DEVICE: p_dev = R_cd p_cam + t_cd.
"""
# Ensure pycameramodel package (local copy) is on the path.
_pycam_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', 'pycameramodel'))
if _pycam_root not in _sys.path:
    _sys.path.insert(0, _pycam_root)
try:
    from pycameramodel import device as _pycam_device_module
    _PYCAM_AVAILABLE = True
except ImportError:
    _PYCAM_AVAILABLE = False
# =========================
# Rail default hyper-parameters (keeps train.py argparse clean)
# =========================
RAIL_DEFAULTS = {
    # matching
    "topk": 1024,
    "min_cos": 0.1,
    "tau": 0.1,
    "hard_matching": False,
    "min_matches": 64,
    # circular solver
    "phi_min": -0.35,
    "phi_max": 0.35,
    "phi_grid_steps": 41,
    "gn_steps": 3,
    "radius": 1.0,
    # linear solver
    "lin_allow_both_signs": True,
    # regularizers
    "lambda_dev": 0.0,
    "lambda_entropy": 0.0,
    "lambda_coverage": 0.0,
    "coverage_bins": 8,
    # robust kernel
    "rho_eps": 1e-6,
}
# =========================
# Original XFeat losses
# =========================
def dual_softmax_loss(X, Y, temp=0.2):
    """
    Coarse descriptor matching loss using dual softmax (symmetric Sinkhorn-like).
    Computes a soft assignment matrix P = softmax(S/temp, dim=1) * softmax(S/temp, dim=0)
    where S = X @ Y^T is the similarity matrix between descriptor sets X and Y.
    The loss is the negative log-likelihood that diagonal entries (ground-truth
    correspondences) are assigned high probability.
    """
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
    """
    Element-wise Smooth-L1 (Huber) loss.
    Quadratic for |x-y| < beta, linear beyond. More robust to outliers than MSE.
    """
    n = torch.abs(x - y)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss
def fine_loss(coords1, coords2, margin=1.0, alpha=0.5):
    """
    Fine-level coordinate regression loss.
    Computes per-pair Euclidean distance, applies Smooth-L1 with `margin`,
    then maps through  alpha * (1 - exp(-loss))  to bound the contribution
    of large errors (saturating loss).
    """
    dist = torch.norm(coords1 - coords2, dim=1)
    loss = smooth_l1_loss(dist, torch.zeros_like(dist), beta=margin)
    loss = alpha * (1.0 - torch.exp(-loss))
    return loss.mean()
def alike_distill_loss(im, kp_map, scores, device='cuda'):
    """
    Keypoint distillation loss from the ALIKE detector.
    Extracts ALIKE keypoints from the raw image, selects the top-300 XFeat
    keypoints by score, and computes the mean distance from each XFeat keypoint
    to its nearest ALIKE keypoint (in normalised [-1,1] image coordinates).
    This encourages XFeat to predict keypoints near where ALIKE would.
    """
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
    """
    Fine coordinate-offset classification loss.
    Quantises the (pts2 - pts1) offset into an 8x8 bin grid and trains the
    network to predict the correct bin via cross-entropy. The confidence
    vector `conf` weights each sample (normalised to sum 1).
    """
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
    """
    Keypoint reliability loss — L1 distance between the predicted heatmap and
    a target heatmap (e.g. generated from ALIKE detections). Scaled by 3.0.
    """
    return F.l1_loss(heatmap, target) * 3.0
def hard_triplet_loss(X, Y, margin=0.5):
    """
    Hard-negative triplet loss for descriptor learning.
    For each positive pair (X[i], Y[i]), the hardest negative is the closest
    non-matching descriptor in Y. Loss = max(0, margin + d_pos - d_neg).
    """
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
    """
    Skew-symmetric (cross-product) matrix from a 3-vector.
    Given t = [tx, ty, tz], returns the 3x3 matrix [t]_x such that
    [t]_x v = t x v for any vector v.
    Used to build the Essential matrix E = [t]_x R.
    """
    t = t.view(3)
    tx, ty, tz = t[0], t[1], t[2]
    z = torch.zeros_like(tx)
    return torch.stack([
        torch.stack([z, -tz, ty]),
        torch.stack([tz, z, -tx]),
        torch.stack([-ty, tx, z]),
    ], dim=0)
def _rot_z(theta):
    """
    3x3 rotation matrix about the Z-axis by angle `theta` (radians).
    Rz(theta) = [[cos, -sin, 0],
                 [sin,  cos, 0],
                 [0,    0,   1]]
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([
        torch.stack([c, -s, z]),
        torch.stack([s,  c, z]),
        torch.stack([z,  z, o]),
    ], dim=0)
def _rot_x(theta):
    """
    3x3 rotation matrix about the X-axis by angle `theta` (radians).
    Rx(theta) = [[1,    0,     0  ],
                 [0,  cos, -sin  ],
                 [0,  sin,  cos  ]]
    Used for the circular-rail manifold when the device frame has
    X=up, Y=left, Z=inward and the rail lies in the ZY plane
    (so rotation is about the X / "up" axis).
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([
        torch.stack([o,  z,  z]),
        torch.stack([z,  c, -s]),
        torch.stack([z,  s,  c]),
    ], dim=0)
def _to_h(x):
    """
    Convert 2D points to homogeneous coordinates by appending a column of ones.
    """
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=1)
def normalize_points_with_K(x_px, K):
    """x_px (N,2) pixels -> calibrated homogeneous (N,3) via inv(K)."""
    K = K.to(device=x_px.device, dtype=x_px.dtype).view(3, 3)
    x_h = _to_h(x_px)
    x_n = (torch.linalg.inv(K) @ x_h.t()).t()
    return x_n
class _PycamUndistortFn(torch.autograd.Function):
    """
    Differentiable wrapper around pycameramodel's try_unproject_from_pixel.
    Forward : calls cam.try_unproject_from_pixel(pt) for each pixel in x_px
              and returns calibrated homogeneous 3-D rays [N, 3].
              Handles any distortion model (LINEAR, RADIAL, FISHEYE_*,
              CHEBYSHEV_*, FISHEYE_RATIONAL, etc.).
    Backward: finite-difference (central-difference) Jacobian.
              4 numpy calls per point (Δx+/Δx−/Δy+/Δy−) — fully model-agnostic.
              Gradient flows through x_px (e.g. soft-assignment x2_soft) and
              is zero for invalid pixels (outside the undistortion limit).
    """
    @staticmethod
    def forward(ctx, x_px, cam):
        """
        Parameters
        ----------
        x_px : Tensor [N, 2]  float32/float64 pixel coordinates
        cam  : pycameramodel.device.Camera  (not a tensor)
        Returns
        -------
        Tensor [N, 3]  calibrated homogeneous coords (ideal + appended 1)
        """
        pts = x_px.detach().cpu().numpy()           # [N, 2] float64
        N = pts.shape[0]
        out = _np.zeros((N, 3), dtype=_np.float32)
        valid = _np.zeros(N, dtype=bool)
        for i in range(N):
            result = cam.try_unproject_from_pixel(pts[i])
            if result is not None:
                out[i] = result.astype(_np.float32)   # [3]: [x_ideal, y_ideal, 1]
                valid[i] = True
            else:
                # Outside undistortion range — map to neutral optical-axis ray.
                # Weight w for such points is typically ~0 so they cannot harm loss.
                out[i] = _np.array([0.0, 0.0, 1.0], dtype=_np.float32)
        ctx.save_for_backward(x_px)
        ctx.cam = cam
        ctx.valid = valid
        return torch.from_numpy(out).to(device=x_px.device, dtype=x_px.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output : Tensor [N, 3]
        Returns grad w.r.t. x_px [N, 2] and None for cam.
        """
        x_px, = ctx.saved_tensors
        cam   = ctx.cam
        valid = ctx.valid
        pts   = x_px.detach().cpu().numpy()           # [N, 2]
        grad_np = grad_output.detach().cpu().numpy()  # [N, 3]
        N = pts.shape[0]
        # 0.05 px is small enough for sub-pixel accuracy yet avoids
        # numerical noise from the iterative undistortion solver.
        eps = 5e-2
        grad_in = _np.zeros_like(pts, dtype=_np.float32)
        for i in range(N):
            if not valid[i]:
                continue
            # 3x2 Jacobian: d(ideal_3d[c]) / d(pixel[k])
            J = _np.zeros((3, 2), dtype=_np.float32)
            for k in range(2):
                pt_p = pts[i].copy(); pt_p[k] += eps
                pt_m = pts[i].copy(); pt_m[k] -= eps
                rp = cam.try_unproject_from_pixel(pt_p)
                rm = cam.try_unproject_from_pixel(pt_m)
                if rp is not None and rm is not None:
                    J[:, k] = ((rp - rm) / (2.0 * eps)).astype(_np.float32)
            # chain rule: grad_in[i] = J^T @ grad_output[i], shape [2]
            grad_in[i] = J.T @ grad_np[i]
        return (
            torch.from_numpy(grad_in).to(device=x_px.device, dtype=x_px.dtype),
            None,   # cam is not a Tensor — no gradient
        )
def normalize_points_with_cam(x_px, cam):
    """
    Fisheye-aware replacement for normalize_points_with_K.
    Converts pixel coordinates [N, 2] → calibrated homogeneous 3-D rays [N, 3]
    using the full distortion model stored in a pycameramodel Camera object.
    Supports any model: LINEAR, RADIAL_*, FISHEYE_*, CHEBYSHEV_*,
    FISHEYE_RATIONAL, etc.
    Differentiable: gradients w.r.t. x_px flow through a finite-difference
    Jacobian (central differences, Δpx = 0.05 px) — model-agnostic.
    Invalid pixels (outside the undistortion range) are mapped to [0, 0, 1]
    with zero gradient, so they neither contribute to the Sampson cost nor
    corrupt the gradient.
    """
    if not _PYCAM_AVAILABLE:
        raise RuntimeError(
            "[normalize_points_with_cam] pycameramodel is not importable. "
            "Install it with: pip install -e modules/pycameramodel"
        )
    return _PycamUndistortFn.apply(x_px, cam)
def sampson_error(x1n_h, x2n_h, E, eps=1e-8):
    """
    Sampson distance (first-order approximation to geometric/reprojection error).
    For calibrated homogeneous points x1, x2 and Essential matrix E:
        d_Sampson = (x2^T E x1)^2 / ( (Ex1)[0]^2 + (Ex1)[1]^2 + (E^Tx2)[0]^2 + (E^Tx2)[1]^2 )
    A point pair lying exactly on its epipolar line gives d=0.
    This is scale-invariant in E (numerator and denominator both scale as a^2).
    """
    Ex1 = (E @ x1n_h.t()).t()
    Etx2 = (E.t() @ x2n_h.t()).t()
    x2tEx1 = torch.sum(x2n_h * Ex1, dim=1)
    denom = Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2
    return (x2tEx1**2) / (denom + eps)
def robust_charbonnier(x, eps=1e-6):
    """
    Charbonnier robust kernel: rho(x) = sqrt(x + eps^2).
    A smooth, differentiable approximation to sqrt(x) that avoids the singularity
    at x=0. Sub-linear growth makes it robust to outlier residuals — large
    errors are down-weighted relative to MSE.
    """
    return torch.sqrt(x + eps*eps)
def weighted_8point_essential(x1n_h, x2n_h, w, eps=1e-8):
    """
    Weighted 8-point algorithm for Essential matrix estimation.
    Constructs the Kronecker-product data matrix A from calibrated point
    pairs, weights each row by sqrt(w), then solves for E via SVD.
    The result is projected onto the Essential manifold by enforcing
    singular values (s, s, 0).
    This is an "unconstrained" estimate (no manifold prior) used by the
    deviation loss to compare against the rail-constrained E.
    Fully autograd-safe: gradients flow through w and point coordinates.
    """
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
# Extrinsics helpers (CAM->DEV given / loaded from XML)
# =========================
def _rodrigues_to_matrix_np(rvec):
    """
    Convert a Rodrigues (angle-axis) rotation vector to a 3x3 rotation matrix
    using the standard Rodrigues formula.  Pure NumPy — no OpenCV required.
    Parameters
    ----------
    rvec : array-like, shape (3,)
        Angle-axis rotation vector. The direction is the axis and the
        magnitude ||rvec|| is the angle in radians.
    Returns
    -------
    R : np.ndarray, shape (3, 3)
    """
    import numpy as np
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    theta = np.linalg.norm(rvec)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = rvec / theta
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
def load_rail_extrinsics_from_calib(device_calib_path, imu_name=None):
    """
    Load camera-to-device extrinsics (R_cd, t_cd) from a device_calibration.xml.
    The ``<SFConfig><Stateinit>`` block stores the IMU/body-to-camera extrinsics:
        ombc  - Rodrigues angle-axis rotation vector, BODY→CAMERA
                p_cam = R_bc @ p_body + t_bc   where R_bc = rodrigues(ombc)
        tbc   - translation, BODY origin → CAMERA origin, expressed in BODY frame
    Device frame convention (as per hardware spec):
        X = up,  Y = left,  Z = inward (into the scene)
    Rail plane is the ZY plane.
    This function converts the BODY→CAMERA convention to CAMERA→DEVICE:
        R_cd = R_bc^T
        t_cd = -R_bc^T @ t_bc
    Parameters
    ----------
    device_calib_path : str
        Path to device_calibration.xml.
    imu_name : str or None
        Key of the IMU config entry (e.g. ``"imu_config_None"``).  When None,
        the parser's default ``"imu_config"`` entry (primary IMU) is used.
    Returns
    -------
    R_cd : np.ndarray, shape (3, 3), dtype float32
        Camera-to-device rotation matrix.
    t_cd : np.ndarray, shape (3,), dtype float32
        Camera-to-device translation vector (in device/body frame units).
    Raises
    ------
    RuntimeError
        If no SFConfig / IMU config is found in the XML.
    """
    import numpy as np
    import os as _os, sys as _sys
    _pycam_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', 'pycameramodel'))
    if _pycam_root not in _sys.path:
        _sys.path.insert(0, _pycam_root)
    from pycameramodel.utils import parse_device_calibration
    params = parse_device_calibration(device_calib_path)
    key = imu_name if imu_name else 'imu_config'
    if key not in params:
        raise RuntimeError(
            f"[load_rail_extrinsics_from_calib] No IMU config key '{key}' found in "
            f"{device_calib_path}.  Available keys: {list(params.keys())}"
        )
    imu = params[key]
    ombc = np.array(imu['ombc'], dtype=np.float64).ravel()   # Rodrigues rotation, BODY->CAM
    tbc  = np.array(imu['tbc'],  dtype=np.float64).ravel()   # translation, BODY->CAM in BODY frame
    R_bc = _rodrigues_to_matrix_np(ombc)   # body->camera rotation (3x3)
    # Camera->device (body):
    R_cd = R_bc.T
    t_cd = -(R_bc.T @ tbc)
    print(f"[Rail] Loaded extrinsics from '{device_calib_path}' (key='{key}')")
    print(f"  ombc  = {ombc}")
    print(f"  tbc   = {tbc}")
    print(f"  R_cd  =\n{R_cd}")
    print(f"  t_cd  = {t_cd}")
    return R_cd.astype(np.float32), t_cd.astype(np.float32)
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
    Circular rail in the device ZY plane (device frame: X=up, Y=left, Z=inward).
    Extrinsics are CAM->DEV (R_cd, t_cd).  Internally uses DEV->CAM (R_dc, t_dc).
    The device rotates about the X axis ("up" direction) because the rail
    lies in the ZY plane.  The reference position in device coords is:
        p0 = [0, 0, R]^T   (along +Z = inward, at angle phi=0)
    E(phi) = [t_ji]_x R_ji  with:
      R_ji = R_dc^T  Rx(-phi)  R_dc
      t_ji = R_dc^T ( Rx(-phi)p0 - p0 + (Rx(-phi)-I) t_dc )
    """
    device, dtype = phi.device, phi.dtype
    r_cd = r_cd.to(device=device, dtype=dtype).view(3, 3)
    t_cd = t_cd.to(device=device, dtype=dtype).view(3)
    r_dc, t_dc = _cam_to_dev_to_dev_to_cam(r_cd, t_cd)
    # Rotate about X axis (rail in ZY plane, X=up is the rotation axis)
    rx = _rot_x(-phi)
    r_ji = r_dc.t() @ rx @ r_dc
    # Reference position in device frame: on the +Z axis at distance rail_radius
    p0 = torch.tensor([0.0, 0.0, rail_radius], device=device, dtype=dtype)
    I = torch.eye(3, device=device, dtype=dtype)
    t_ji = r_dc.t() @ (rx @ p0 - p0 + (rx - I) @ t_dc)
    return _skew(t_ji) @ r_ji
def _circular_objective(phi, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd):
    """
    Evaluate the weighted robust Sampson cost at a given angle phi on the circular rail.
    Builds E(phi) from the circular-rail geometry, computes per-match Sampson
    errors, applies the Charbonnier kernel, and returns the weighted sum:
        J(phi) = sum_m  w_m * rho( d_Sampson(x1_m, x2_m, E(phi)) )
    """
    E = essential_from_circular_phi(phi, rail_radius, r_cd, t_cd)
    r = sampson_error(x1n_h, x2n_h, E)
    return torch.sum(w * robust_charbonnier(r, eps=rho_eps))
def solve_phi_circular(x1n_h, x2n_h, w,
                         phi_min, phi_max,
                         coarse_steps=41, gn_steps=3,
                         rho_eps=1e-6,
                         rail_radius=1.0,
                         r_cd=None, t_cd=None):
     """
     Solve for the optimal rail angle phi* on the circular manifold.
     Two-stage optimisation:
        1. Coarse grid search: evaluate J(phi) on a uniform grid of
            `coarse_steps` values in [phi_min, phi_max] and pick the minimum.
        2. Gauss-Newton refinement: `gn_steps` iterations of finite-difference
            Newton updates  phi <- phi - g/H  (clamped to [phi_min, phi_max]).
     The search for phi* is under no_grad, but the final cost computation is outside.
     """
     device, dtype = x1n_h.device, x1n_h.dtype
     with torch.no_grad():
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
     # Final cost computation outside no_grad for gradient flow
     J = _circular_objective(phi, x1n_h, x2n_h, w, rho_eps, rail_radius, r_cd, t_cd)
     return phi, E, J
# =========================
# Linear rail manifold (fixed orientation, sideways along +X_dev)
# =========================
def essential_from_linear_sideways(r_cd, sign=+1.0):
    """
    Linear rail in the device ZY plane (device frame: X=up, Y=left, Z=inward).
    The device translates sideways along the device +Y axis (Y=left/right).
    Orientation is fixed: R_ji = I.
      t_dev ∝ sign * [0, 1, 0]^T   (along Y = left)
    With CAM->DEV rotation R_cd:
      t_cam = R_cd @ t_dev   (since R_dc = R_cd^T, R_dc^T = R_cd)
      E = [t_cam]_x
    """
    device, dtype = r_cd.device, r_cd.dtype
    r_cd = r_cd.to(device=device, dtype=dtype).view(3, 3)
    # Translation is along Y (left/right) in the ZY rail plane
    t_dev = torch.tensor([0.0, sign, 0.0], device=device, dtype=dtype)
    t_cam = r_cd @ t_dev
    return _skew(t_cam)
def _linear_objective(sign, x1n_h, x2n_h, w, rho_eps, r_cd):
    """
    Evaluate the weighted robust Sampson cost for the linear rail at a given sign.
    The linear rail has no rotation (R_ji = I) and translation direction
    t_dev = sign * [1,0,0]^T. Only two possible Essential matrices exist
    (sign=+1 or sign=-1).
    """
    E = essential_from_linear_sideways(r_cd, sign=sign)
    r = sampson_error(x1n_h, x2n_h, E)
    return torch.sum(w * robust_charbonnier(r, eps=rho_eps))
def solve_linear_sign(x1n_h, x2n_h, w, rho_eps=1e-6, r_cd=None, allow_both=True):
    """
    Solve for the optimal translation sign on the linear rail.
    Since the linear rail has only two possible Essential matrices (sign=+1
    and sign=-1), this is a discrete exhaustive search (no optimisation loop).
    The search for sign is under no_grad, but the final cost computation is outside.
    """
    if not allow_both:
        with torch.no_grad():
            sign = torch.tensor(1.0, device=x1n_h.device, dtype=x1n_h.dtype)
            E = essential_from_linear_sideways(r_cd, sign=1.0)
        # Final cost computation outside no_grad
        J = _linear_objective(sign.item(), x1n_h, x2n_h, w, rho_eps, r_cd)
        return sign, E, J
    with torch.no_grad():
        Jp = _linear_objective(1.0, x1n_h, x2n_h, w, rho_eps, r_cd)
        Jm = _linear_objective(-1.0, x1n_h, x2n_h, w, rho_eps, r_cd)
        if Jp <= Jm:
            sign = torch.tensor(1.0, device=x1n_h.device, dtype=x1n_h.dtype)
            E = essential_from_linear_sideways(r_cd, sign=1.0)
        else:
            sign = torch.tensor(-1.0, device=x1n_h.device, dtype=x1n_h.dtype)
            E = essential_from_linear_sideways(r_cd, sign=-1.0)
    # Final cost computation outside no_grad
    J = _linear_objective(sign.item(), x1n_h, x2n_h, w, rho_eps, r_cd)
    return sign, E, J
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
    Phase-2 soft-assignment extraction. Gradients flow through BOTH
    match coordinates (via the soft expectation) AND match weights.
    f1,f2 : [C, H, W]  — dense descriptor feature maps
    h1,h2 : [H, W]     — heatmaps / reliability maps
    topk  : int         — number of source keypoints (selected by heatmap score)
    tau   : float       — softmax temperature (lower = sharper; 0.1 is a good start)
    dust_bin : bool     — if True, append a dustbin row/col so that poor
                          matches can be "explained away" instead of forced onto a target.
    Returns x1_hard [M,2], x2_soft [M,2], w [M].
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
    """
    Shannon entropy of the normalised weight distribution.
    H(w) = - sum_m  p_m log(p_m),   where p_m = w_m / sum(w)
    Higher entropy means weights are more uniformly spread across matches.
    Used as a regulariser: maximising entropy discourages the network from
    collapsing all weight onto a few matches.
    """
    w = torch.clamp(w, min=0.0)
    p = w / (w.sum() + eps)
    p = torch.clamp(p, min=eps)
    return -torch.sum(p * torch.log(p))
def _coverage_entropy(x_px, w, img_h, img_w, bins=8, eps=1e-8):
    """
    Spatial coverage entropy of match locations.
    Divides the image into a bins x bins grid. For each cell, accumulates the
    total match weight landing in that cell, normalises to a probability
    distribution, and computes its Shannon entropy.
    Higher entropy means matches are spread across the image rather than
    clustered in one region. Used as a regulariser (subtracted from loss
    -> maximise entropy -> encourage spatial spread).
    """
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
    k0: torch.Tensor = None,
    k1: torch.Tensor = None,
    cam=None,
    img_h: int,
    img_w: int,
    r_cd: torch.Tensor = None,
    t_cd: torch.Tensor = None,
):
    """
    Unified rail self-supervision loss (supports circular and linear manifolds).
    This is the main entry point called from the training loop. Given matched
    pixel coordinates and weights (from either Phase-1 hard or Phase-2 soft
    extraction), it:
      1. Converts pixel coords to calibrated homogeneous coords.
         - If `cam` (a pycameramodel Camera) is provided, uses the full
           fisheye/distortion model via normalize_points_with_cam — supports
           any model: LINEAR, RADIAL_*, FISHEYE_*, CHEBYSHEV_*, FISHEYE_RATIONAL.
           Differentiable via a finite-difference Jacobian.
         - Otherwise falls back to the pinhole normalize_points_with_K using
           `k0` / `k1` intrinsic matrices.
      2. Solves for the optimal manifold parameter (phi* or sign*) under
         stop-grad so the solver itself doesn't inject gradients.
      3. Computes the manifold cost J_manifold = sum w_m rho(d_Sampson(...)).
      4. Optionally computes the deviation loss: L_dev = log(J_manifold) - log(J_free)
         where J_free uses an unconstrained weighted 8-point Essential estimate.
      5. Optionally adds entropy + spatial-coverage regularisers.
    Gradients flow through w (and through x2_px in Phase-2 soft mode) back
    into the descriptor and heatmap branches of the network.
    Uses RAIL_DEFAULTS for solver/regularizer hyper-parameters.
    Returns (J_manifold, L_dev, J_free, reg, aux_param, E_manifold).
    """
    rc = RAIL_DEFAULTS
    if cam is not None:
        # Fisheye-aware undistortion via pycameramodel (any distortion model).
        x1n_h = normalize_points_with_cam(x1_px, cam)
        x2n_h = normalize_points_with_cam(x2_px, cam)
    elif k0 is not None and k1 is not None:
        # Pinhole fallback.
        x1n_h = normalize_points_with_K(x1_px, k0)
        x2n_h = normalize_points_with_K(x2_px, k1)
    else:
        raise ValueError(
            "[RailLoss] Provide either `cam` (pycameramodel Camera) for fisheye undistortion "
            "or `k0`/`k1` intrinsic tensors for pinhole undistortion."
        )
    mode_l = mode.lower()
    if mode_l in ("circular", "circle", "circ"):
        if r_cd is None or t_cd is None:
            raise ValueError("[RailLoss] circular mode requires r_cd and t_cd.")
        aux, E_m, J_m = solve_phi_circular(
            x1n_h=x1n_h, x2n_h=x2n_h, w=w,
            phi_min=rc["phi_min"], phi_max=rc["phi_max"],
            coarse_steps=rc["phi_grid_steps"], gn_steps=rc["gn_steps"],
            rho_eps=rc["rho_eps"], rail_radius=rc["radius"],
            r_cd=r_cd, t_cd=t_cd
        )
    elif mode_l in ("linear", "lin"):
        if r_cd is None:
            raise ValueError("[RailLoss] linear mode requires r_cd.")
        aux, E_m, J_m = solve_linear_sign(
            x1n_h=x1n_h, x2n_h=x2n_h, w=w,
            rho_eps=rc["rho_eps"], r_cd=r_cd,
            allow_both=rc["lin_allow_both_signs"]
        )
    else:
        raise ValueError(f"[RailLoss] Unknown mode: {mode}. Use 'circular' or 'linear'.")
    # Optional deviation loss (free 8-point vs manifold)
    if rc["lambda_dev"] > 0.0:
        E_free = weighted_8point_essential(x1n_h, x2n_h, w)
        J_free = torch.sum(w * robust_charbonnier(sampson_error(x1n_h, x2n_h, E_free), eps=rc["rho_eps"]))
        L_dev = torch.log(J_m + 1e-8) - torch.log(J_free + 1e-8)
    else:
        J_free = torch.zeros_like(J_m)
        L_dev = torch.zeros_like(J_m)
    # Regularizers
    reg = torch.zeros_like(J_m)
    if rc["lambda_entropy"] > 0.0:
        reg = reg - rc["lambda_entropy"] * _weighted_entropy(w)
    if rc["lambda_coverage"] > 0.0:
        reg = reg - rc["lambda_coverage"] * _coverage_entropy(
            x1_px, w, img_h=img_h, img_w=img_w, bins=rc["coverage_bins"]
        )
    return J_m, L_dev, J_free, reg, aux, E_m