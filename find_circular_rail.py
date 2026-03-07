"""
Determine the circular rail rotation axis & visualise the setup.

Given: ombc is CAM→BODY, so Rbc = rodrigues(ombc) = R_cd (camera→device).

Circular rail geometry (from losses.py):
    - Device moves along a circle of radius R in a plane.
    - The rotation axis is perpendicular to the rail plane.
    - Reference position: p0 = [0, 0, R] (at phi=0).
    - R_ji = R_dc^T  Rx(-phi)  R_dc   (for rotation about X axis)
    - t_ji = R_dc^T ( Rx(-phi)*p0 - p0 + (Rx(-phi)-I) * t_dc )

This script tests all 3 possible rotation axes (X, Y, Z) and both rotation
conventions (R_cd vs R_cd^T) against SIFT matches to find which combination
gives the lowest Sampson error across multiple angle candidates.

After finding the answer, opens an interactive 3D plotly plot showing:
  - Device coordinate frame (X=up, Y=left, Z=inward)
  - Camera coordinate frame
  - The circular rail arc (in the plane perp. to the winning axis)
  - Multiple device positions along the arc
  - The rotation axis

Run on the machine with data:
    python find_circular_rail.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import cv2
import numpy as np
import torch

# === CONFIGURE THESE PATHS ===
CIRCULAR_DATA_PATH = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Circular_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/Camera2_train'
DEVICE_CALIB_PATH  = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Circular_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/device_calibration.xml'
RAIL_CAM_NAME = 'trackingA'
IMU_NAME = ''
TRAINING_RES = (800, 608)
NUM_PAIRS = 30
RAIL_RADIUS = 1.0           # metres (match losses.py default)
PHI_MIN = -0.15             # ~8.6 deg
PHI_MAX =  0.15
COARSE_STEPS = 41
# ==============================


# ─── geometry helpers ──────────────────────────────────────────────────────────

def _skew(t):
    t = t.view(3)
    tx, ty, tz = t[0], t[1], t[2]
    z = torch.zeros_like(tx)
    return torch.stack([
        torch.stack([z, -tz, ty]),
        torch.stack([tz, z, -tx]),
        torch.stack([-ty, tx, z]),
    ], dim=0)


def _rot_about(axis_idx, phi):
    """Rotation matrix about axis 0=X, 1=Y, 2=Z by angle phi (radians)."""
    c, s = torch.cos(phi), torch.sin(phi)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    if axis_idx == 0:  # X
        return torch.stack([
            torch.stack([o, z, z]),
            torch.stack([z, c, -s]),
            torch.stack([z, s, c]),
        ], dim=0)
    elif axis_idx == 1:  # Y
        return torch.stack([
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ], dim=0)
    else:  # Z
        return torch.stack([
            torch.stack([c, -s, z]),
            torch.stack([s, c, z]),
            torch.stack([z, z, o]),
        ], dim=0)


def _reference_pos(axis_idx, radius):
    """Reference position p0 on the circle: perpendicular to the rotation axis."""
    # For X-axis rotation: circle in ZY plane, p0 = [0, 0, R]
    # For Y-axis rotation: circle in XZ plane, p0 = [0, 0, R]
    # For Z-axis rotation: circle in XY plane, p0 = [R, 0, 0]
    if axis_idx == 0:  # X
        return torch.tensor([0., 0., radius])
    elif axis_idx == 1:  # Y
        return torch.tensor([radius, 0., 0.])
    else:  # Z
        return torch.tensor([radius, 0., 0.])


def sampson_error(x1h, x2h, E, eps=1e-8):
    Ex1 = (E @ x1h.t()).t()
    Etx2 = (E.t() @ x2h.t()).t()
    x2tEx1 = torch.sum(x2h * Ex1, dim=1)
    denom = Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2
    return (x2tEx1**2) / (denom + eps)


def robust_charbonnier(x, eps=1e-6):
    return torch.sqrt(x + eps)


def essential_from_circular(phi, radius, axis_idx, r_cd, t_cd, use_transpose):
    """
    Build Essential matrix for a circular rail with given rotation axis.

    Args:
        phi:           rotation angle (radians)
        radius:        rail radius (metres)
        axis_idx:      0=X, 1=Y, 2=Z  (device-frame rotation axis)
        r_cd:          cam→dev rotation  (3×3)
        t_cd:          cam→dev translation (3,)
        use_transpose: if True, use R_dc = R_cd (instead of R_cd^T)
                       — this tests the alternate convention
    Returns:
        E  (3×3 Essential matrix)
    """
    device, dtype = phi.device, phi.dtype
    r_cd = r_cd.to(device=device, dtype=dtype).view(3, 3)
    t_cd = t_cd.to(device=device, dtype=dtype).view(3)

    if use_transpose:
        r_dc = r_cd        # alternate convention test
        t_dc = -r_dc @ t_cd
    else:
        r_dc = r_cd.t()    # standard: dev→cam = R_cd^T
        t_dc = -r_dc @ t_cd

    rx = _rot_about(axis_idx, -phi)
    r_ji = r_dc.t() @ rx @ r_dc
    p0 = _reference_pos(axis_idx, radius).to(device=device, dtype=dtype)
    I = torch.eye(3, device=device, dtype=dtype)
    t_ji = r_dc.t() @ (rx @ p0 - p0 + (rx - I) @ t_dc)
    return _skew(t_ji) @ r_ji


def circular_objective(phi, x1h, x2h, radius, axis_idx, r_cd, t_cd, use_transpose):
    E = essential_from_circular(phi, radius, axis_idx, r_cd, t_cd, use_transpose)
    r = sampson_error(x1h, x2h, E)
    return robust_charbonnier(r).median().item()


# ─── 3D visualisation ─────────────────────────────────────────────────────────

def _arrow_trace(origin, direction, length, color, name, width=5, dash=None):
    import plotly.graph_objects as go
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12) * length
    tip = o + d
    line_style = dict(color=color, width=width)
    if dash:
        line_style['dash'] = dash
    line = go.Scatter3d(
        x=[o[0], tip[0]], y=[o[1], tip[1]], z=[o[2], tip[2]],
        mode='lines', line=line_style, name=name, showlegend=True,
    )
    cone = go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[d[0]], v=[d[1]], w=[d[2]],
        sizemode='absolute', sizeref=length * 0.12,
        colorscale=[[0, color], [1, color]], showscale=False,
        showlegend=False,
    )
    return [line, cone]


def _frame_traces(origin, R_cols, axis_length, prefix,
                  colors=('red', 'green', 'blue'),
                  labels=('X', 'Y', 'Z'), dash=None):
    traces = []
    for i in range(3):
        name = f'{prefix} {labels[i]}'
        traces += _arrow_trace(origin, R_cols[:, i], axis_length,
                               colors[i], name, width=4, dash=dash)
    return traces


def plot_circular_rail(R_cd_np, tbc_np, best_axis_idx, best_axis_name,
                       best_use_transpose, best_phi, radius):
    """
    Interactive 3D plot of device frame, camera frame, circular rail arc,
    multiple device positions along the arc, and the rotation axis.
    Everything is in the DEVICE coordinate system.
    """
    import plotly.graph_objects as go

    R_cd = R_cd_np.astype(np.float64)
    t_cd = tbc_np.astype(np.float64)
    cam_origin = t_cd  # camera origin in device frame

    ax_len = 0.15

    traces = []

    # ── Device frame at origin ──
    traces += _frame_traces(np.zeros(3), np.eye(3), ax_len, 'Dev',
                            colors=('red', 'green', 'blue'),
                            labels=('X↑', 'Y←', 'Z↗'))

    # ── Camera frame at camera position ──
    traces += _frame_traces(cam_origin, R_cd, ax_len, 'Cam',
                            colors=('salmon', 'lightgreen', 'lightskyblue'),
                            labels=('Xc', 'Yc', 'Zc(optical)'), dash='dot')

    # ── Optical axis ──
    optical = R_cd[:, 2]
    traces += _arrow_trace(cam_origin, optical, ax_len * 3,
                           'cyan', 'Optical axis', width=2, dash='dash')

    # ── Rotation axis (through origin) ──
    axis_vec = np.zeros(3)
    axis_vec[best_axis_idx] = 1.0
    traces += _arrow_trace(-axis_vec * 0.3, axis_vec, 0.6,
                           'gold', f'Rot axis ({best_axis_name})', width=6)

    # ── Reference position p0 ──
    p0_np = np.zeros(3)
    if best_axis_idx == 0:
        p0_np[2] = radius
    elif best_axis_idx == 1:
        p0_np[0] = radius
    else:
        p0_np[0] = radius
    traces.append(go.Scatter3d(
        x=[p0_np[0]], y=[p0_np[1]], z=[p0_np[2]],
        mode='markers+text', marker=dict(size=5, color='purple'),
        text=[f'p0 (R={radius:.1f}m)'], textposition='top center',
        name='Reference pos p0',
    ))

    # ── Circular arc (many phi values) ──
    n_arc = 100
    phis = np.linspace(-np.pi / 6, np.pi / 6, n_arc)  # ±30 deg for visualisation
    arc_pts = []
    for phi_val in phis:
        Rx = _rot_about_np(best_axis_idx, -phi_val)
        arc_pts.append(Rx @ p0_np)
    arc_pts = np.array(arc_pts)
    traces.append(go.Scatter3d(
        x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2],
        mode='lines', line=dict(color='magenta', width=4),
        name='Circular rail arc',
    ))

    # ── Device positions at several phi values ──
    sample_phis = np.linspace(-np.pi / 12, np.pi / 12, 5)  # ±15 deg, 5 pts
    for i, phi_val in enumerate(sample_phis):
        Rx = _rot_about_np(best_axis_idx, -phi_val)
        pos = Rx @ p0_np
        angle_deg = np.degrees(phi_val)
        traces.append(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers+text',
            marker=dict(size=4, color='darkorange',
                        symbol='diamond'),
            text=[f'{angle_deg:+.1f}°'],
            textposition='top center',
            name=f'Dev @ {angle_deg:+.1f}°',
            showlegend=(i == 0),
        ))

    # ── Markers ──
    traces.append(go.Scatter3d(
        x=[cam_origin[0]], y=[cam_origin[1]], z=[cam_origin[2]],
        mode='markers+text', marker=dict(size=6, color='orange'),
        text=['Camera'], textposition='bottom center',
        name='Camera origin',
    ))
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text', marker=dict(size=6, color='black'),
        text=['Device/IMU'], textposition='bottom center',
        name='Device origin',
    ))

    # ── Rail plane (semi-transparent surface) ──
    # Draw a thin mesh in the plane perpendicular to the rotation axis
    _draw_rail_plane(traces, best_axis_idx, radius, go)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Circular Rail Setup  (Device Frame)',
        scene=dict(
            xaxis_title='Dev X (up)',
            yaxis_title='Dev Y (left)',
            zaxis_title='Dev Z (inward)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99),
        width=1000, height=800,
    )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'circular_rail_setup.html')
    fig.write_html(out, auto_open=True)
    print(f"\n3D plot saved → {out}")


def _rot_about_np(axis_idx, phi):
    """Numpy rotation matrix about axis 0=X, 1=Y, 2=Z."""
    c, s = np.cos(phi), np.sin(phi)
    if axis_idx == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif axis_idx == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    else:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _draw_rail_plane(traces, axis_idx, radius, go):
    """Add a semi-transparent disc in the rail plane."""
    n = 60
    angles = np.linspace(0, 2 * np.pi, n)
    # The rail plane is perpendicular to axis_idx
    xs, ys, zs = [], [], []
    for a in angles:
        r = radius * 1.2
        if axis_idx == 0:   # plane is YZ
            xs.append(0.0); ys.append(r * np.cos(a)); zs.append(r * np.sin(a))
        elif axis_idx == 1: # plane is XZ
            xs.append(r * np.cos(a)); ys.append(0.0); zs.append(r * np.sin(a))
        else:               # plane is XY
            xs.append(r * np.cos(a)); ys.append(r * np.sin(a)); zs.append(0.0)
    # Add center point
    cx = cy = cz = 0.0
    xs.append(cx); ys.append(cy); zs.append(cz)
    # Triangulation: fan from center
    import numpy as np
    center_idx = len(xs) - 1
    i_arr, j_arr, k_arr = [], [], []
    for idx in range(n - 1):
        i_arr.append(center_idx)
        j_arr.append(idx)
        k_arr.append(idx + 1)
    traces.append(go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_arr, j=j_arr, k=k_arr,
        color='magenta', opacity=0.08,
        name='Rail plane',
        showlegend=True,
    ))


# ─── main logic ───────────────────────────────────────────────────────────────

def find_circular_rail():
    from modules.training.losses import _rodrigues_to_matrix_np

    _pycam_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'modules', 'pycameramodel'))
    if _pycam_root not in sys.path:
        sys.path.insert(0, _pycam_root)
    from pycameramodel.utils import parse_device_calibration
    from pycameramodel import device as pycam

    # --- Load extrinsics (ombc = cam→body, so Rbc = R_cd) ---
    params = parse_device_calibration(DEVICE_CALIB_PATH)
    key = IMU_NAME if IMU_NAME else 'imu_config'
    imu = params[key]
    ombc = np.array(imu['ombc'], dtype=np.float64).ravel()
    tbc  = np.array(imu['tbc'],  dtype=np.float64).ravel()
    Rbc = _rodrigues_to_matrix_np(ombc)
    r_cd = torch.from_numpy(Rbc.astype(np.float32))  # cam→device
    t_cd = torch.from_numpy(tbc.astype(np.float32))

    print(f"ombc = {ombc}")
    print(f"tbc  = {tbc}")
    print(f"R_cd (=Rbc, cam→dev) =\n{Rbc}")
    print()

    # --- Load camera ---
    dev = pycam.Device(DEVICE_CALIB_PATH)
    cam_keys = list(dev.cameras.keys())
    cam_key = RAIL_CAM_NAME if RAIL_CAM_NAME in dev.cameras else cam_keys[0]
    cam = dev.cameras[cam_key]
    print(f"Camera: {cam_key}\n")

    # --- Load image pairs & get SIFT matches ---
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    paths = sorted([os.path.join(CIRCULAR_DATA_PATH, f)
                    for f in os.listdir(CIRCULAR_DATA_PATH)
                    if os.path.splitext(f)[1].lower() in exts])
    import random
    random.seed(42)

    sift_pairs = []
    attempts = 0
    while len(sift_pairs) < NUM_PAIRS and attempts < NUM_PAIRS * 10:
        i0, i1 = random.sample(range(len(paths)), 2)
        im0 = cv2.resize(cv2.imread(paths[i0], cv2.IMREAD_GRAYSCALE),
                          TRAINING_RES)
        im1 = cv2.resize(cv2.imread(paths[i1], cv2.IMREAD_GRAYSCALE),
                          TRAINING_RES)
        sift = cv2.SIFT_create()
        kp0, des0 = sift.detectAndCompute(im0, None)
        kp1, des1 = sift.detectAndCompute(im1, None)
        if des0 is None or des1 is None:
            attempts += 1; continue
        matches = cv2.BFMatcher().knnMatch(des0, des1, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            attempts += 1; continue
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

        # Undistort via pycameramodel
        x0h, x1h = [], []
        for pt in pts0:
            r = cam.try_unproject_from_pixel(pt)
            x0h.append(r if r is not None else np.array([0., 0., 1.]))
        for pt in pts1:
            r = cam.try_unproject_from_pixel(pt)
            x1h.append(r if r is not None else np.array([0., 0., 1.]))
        x0h = torch.from_numpy(np.array(x0h, dtype=np.float32))
        x1h = torch.from_numpy(np.array(x1h, dtype=np.float32))
        sift_pairs.append((x0h, x1h))
        attempts += 1

    print(f"Collected {len(sift_pairs)} SIFT-matched pairs\n")

    # --- Test all combos ---
    axis_names = {0: 'X (up)', 1: 'Y (left)', 2: 'Z (inward)'}
    conv_names = {False: 'R_dc = R_cd^T (standard)', True: 'R_dc = R_cd (alternate)'}

    results = {}
    for axis_idx in [0, 1, 2]:
        for use_transpose in [False, True]:
            medians = []
            for x0h, x1h in sift_pairs:
                # Sweep phi and take best
                phis = torch.linspace(PHI_MIN, PHI_MAX, COARSE_STEPS)
                best = float('inf')
                for phi in phis:
                    val = circular_objective(phi, x0h, x1h,
                                             RAIL_RADIUS, axis_idx,
                                             r_cd, t_cd, use_transpose)
                    if val < best:
                        best = val
                medians.append(best)
            avg = np.mean(medians)
            results[(axis_idx, use_transpose)] = avg

    # --- OpenCV reference ---
    ref_medians = []
    for x0h, x1h in sift_pairs:
        E_cv, _ = cv2.findEssentialMat(
            x0h[:, :2].numpy(), x1h[:, :2].numpy(),
            np.eye(3), method=cv2.RANSAC, threshold=0.005)
        if E_cv is not None:
            serr = sampson_error(
                x0h, x1h,
                torch.from_numpy(E_cv[:3].astype(np.float32)))
            ref_medians.append(robust_charbonnier(serr).median().item())
    ref_avg = np.mean(ref_medians) if ref_medians else float('inf')

    # --- Print results ---
    print("=" * 85)
    print(f"{'Avg Median Robust Sampson':>26}  |  Rot axis          |  Convention")
    print("=" * 85)
    for (aidx, ut), err in sorted(results.items(), key=lambda x: x[1]):
        marker = "  <-- BEST" if err == min(results.values()) else ""
        print(f"  {err:24.8f}  |  {axis_names[aidx]:<17} |  "
              f"{conv_names[ut]}{marker}")
    print("-" * 85)
    print(f"  {ref_avg:24.8f}  |  OpenCV 5pt RANSAC (reference)")
    print("=" * 85)
    print()

    best_combo = min(results, key=results.get)
    best_axis_idx, best_use_transpose = best_combo
    print(f">>> ANSWER: Rotation axis = {axis_names[best_axis_idx]}, "
          f"Convention = {conv_names[best_use_transpose]}")
    print()

    # Find best phi for the winning combo (just for the plot annotation)
    best_phi = 0.0
    best_cost = float('inf')
    x0h, x1h = sift_pairs[0]
    phis = torch.linspace(PHI_MIN, PHI_MAX, COARSE_STEPS)
    for phi in phis:
        val = circular_objective(phi, x0h, x1h, RAIL_RADIUS,
                                 best_axis_idx, r_cd, t_cd,
                                 best_use_transpose)
        if val < best_cost:
            best_cost = val
            best_phi = phi.item()

    # --- 3D interactive plot ---
    print("\nGenerating interactive 3D plot …")
    plot_circular_rail(Rbc, tbc, best_axis_idx,
                       axis_names[best_axis_idx],
                       best_use_transpose, best_phi, RAIL_RADIUS)


if __name__ == '__main__':
    find_circular_rail()
