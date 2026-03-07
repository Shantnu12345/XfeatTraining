"""
Determine the linear rail translation axis in device frame & visualise the setup.

Given: ombc is CAM→BODY, so Rbc = rodrigues(ombc) = R_cd (camera→device).

Tests all 3 device-frame axes (X, Y, Z) × 2 rotation directions (r_cd@t vs r_cd^T@t)
against SIFT matches to find which combination gives the lowest Sampson error.

After finding the answer, opens an interactive 3D plotly plot (in browser) showing:
  - Device coordinate frame (X=up, Y=left, Z=inward)
  - Camera coordinate frame (transformed by extrinsics)
  - Camera optical axis (Z_cam direction)
  - The winning linear rail axis & direction

Run on the machine with data:
    python find_rail_axis.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import cv2
import numpy as np
import torch

# === CONFIGURE THESE PATHS ===
RAIL_DATA_PATH = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Linear_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/Camera2_train'
DEVICE_CALIB_PATH = '/local/mnt/workspace/v3dof/data/C_Building_Zumba_Room_Center/Linear_Rail/Foreseer/Capture_2/forseer_8220f229_2024-04-22-15-23-38/device_calibration.xml'
RAIL_CAM_NAME = 'trackingA'
IMU_NAME = ''
NUM_PAIRS = 30
RAIL_VIS_LENGTH = 0.5       # length of rail line in visualisation (metres)
# ==============================


# ─── geometry helpers ──────────────────────────────────────────────────────────

def skew(t):
    t = t.view(3)
    tx, ty, tz = t[0], t[1], t[2]
    z = torch.zeros_like(tx)
    return torch.stack([
        torch.stack([z, -tz, ty]),
        torch.stack([tz, z, -tx]),
        torch.stack([-ty, tx, z]),
    ], dim=0)


def sampson_error(x1h, x2h, E, eps=1e-8):
    Ex1 = (E @ x1h.t()).t()
    Etx2 = (E.t() @ x2h.t()).t()
    x2tEx1 = torch.sum(x2h * Ex1, dim=1)
    denom = Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2
    return (x2tEx1**2) / (denom + eps)


# ─── 3D visualisation helpers ─────────────────────────────────────────────────

def _arrow_trace(origin, direction, length, color, name, width=5, dash=None):
    """Plotly Scatter3d line + Cone tip for a 3D arrow."""
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
    """Three arrows for a coordinate frame (columns of R as axis dirs)."""
    traces = []
    for i in range(3):
        name = f'{prefix} {labels[i]}'
        traces += _arrow_trace(origin, R_cols[:, i], axis_length,
                               colors[i], name, width=4, dash=dash)
    return traces


def plot_linear_rail(R_cd_np, tbc_np, best_axis_vec, best_axis_name):
    """
    Interactive 3D plot of device frame, camera frame, and the winning linear
    rail axis.  Everything is drawn in the DEVICE coordinate system.

    Device origin = (0,0,0), X=up(red), Y=left(green), Z=inward(blue).
    Camera origin = tbc (camera origin in device frame).
    Camera axes   = columns of R_cd (cam→dev).
    """
    import plotly.graph_objects as go

    R_cd = R_cd_np.astype(np.float64)       # cam→dev rotation
    t_cd = tbc_np.astype(np.float64)         # cam→dev translation

    # Camera origin in device frame (when p_cam=0: p_dev = t_cd)
    cam_origin = t_cd

    ax = 0.15  # axis length (15 cm)

    traces = []

    # ── Device frame at origin ──
    traces += _frame_traces(np.zeros(3), np.eye(3), ax, 'Dev',
                            colors=('red', 'green', 'blue'),
                            labels=('X↑', 'Y←', 'Z↗'))

    # ── Camera frame at camera position ──
    # Columns of R_cd = device-frame expression of cam X, Y, Z axes
    traces += _frame_traces(cam_origin, R_cd, ax, 'Cam',
                            colors=('salmon', 'lightgreen', 'lightskyblue'),
                            labels=('Xc', 'Yc', 'Zc(optical)'), dash='dot')

    # ── Extended optical axis ──
    optical = R_cd[:, 2]
    traces += _arrow_trace(cam_origin, optical, ax * 3,
                           'cyan', 'Optical axis', width=2, dash='dash')

    # ── Linear rail ──
    rail_dir = np.asarray(best_axis_vec, dtype=np.float64)
    half = RAIL_VIS_LENGTH / 2
    p0 = -rail_dir * half
    p1 = rail_dir * half
    traces.append(go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode='lines', line=dict(color='magenta', width=8),
        name=f'Rail ({best_axis_name})',
    ))
    traces += _arrow_trace(np.zeros(3), rail_dir, RAIL_VIS_LENGTH * 0.6,
                           'magenta', 'Rail +dir', width=6)

    # ── Markers ──
    traces.append(go.Scatter3d(
        x=[cam_origin[0]], y=[cam_origin[1]], z=[cam_origin[2]],
        mode='markers+text', marker=dict(size=6, color='orange'),
        text=['Camera'], textposition='top center',
        name='Camera origin',
    ))
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text', marker=dict(size=6, color='black'),
        text=['Device/IMU'], textposition='bottom center',
        name='Device origin',
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Linear Rail Setup  (Device Frame)',
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
                       'linear_rail_setup.html')
    fig.write_html(out, auto_open=True)
    print(f"\n3D plot saved → {out}")


# ─── main logic ───────────────────────────────────────────────────────────────

def find_rail_axis():
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
    paths = sorted([os.path.join(RAIL_DATA_PATH, f)
                    for f in os.listdir(RAIL_DATA_PATH)
                    if os.path.splitext(f)[1].lower() in exts])
    import random
    random.seed(42)

    sift_pairs = []
    attempts = 0
    while len(sift_pairs) < NUM_PAIRS and attempts < NUM_PAIRS * 10:
        i0, i1 = random.sample(range(len(paths)), 2)
        im0 = cv2.imread(paths[i0], cv2.IMREAD_GRAYSCALE)
        im1 = cv2.imread(paths[i1], cv2.IMREAD_GRAYSCALE)
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
    axes = {
        'X (up)':      torch.tensor([1., 0., 0.]),
        'Y (left)':    torch.tensor([0., 1., 0.]),
        'Z (inward)':  torch.tensor([0., 0., 1.]),
    }
    rot_fns = {
        't_cam = R_cd @ t_dev':   lambda r, t: r @ t,
        't_cam = R_cd^T @ t_dev': lambda r, t: r.t() @ t,
    }

    results = {}
    for aname, axis in axes.items():
        for rname, rot_fn in rot_fns.items():
            medians = []
            for x0h, x1h in sift_pairs:
                best = float('inf')
                for sign in [+1., -1.]:
                    t_dev = sign * axis
                    t_cam = rot_fn(r_cd, t_dev)
                    E = skew(t_cam)
                    serr = sampson_error(x0h, x1h, E)
                    best = min(best, serr.median().item())
                medians.append(best)
            avg = np.mean(medians)
            results[(aname, rname)] = avg

    # --- OpenCV 5-point reference ---
    ref_medians = []
    for x0h, x1h in sift_pairs:
        E_cv, _ = cv2.findEssentialMat(
            x0h[:, :2].numpy(), x1h[:, :2].numpy(),
            np.eye(3), method=cv2.RANSAC, threshold=0.005)
        if E_cv is not None:
            serr = sampson_error(
                x0h, x1h,
                torch.from_numpy(E_cv[:3].astype(np.float32)))
            ref_medians.append(serr.median().item())
    ref_avg = np.mean(ref_medians) if ref_medians else float('inf')

    # --- Print results ---
    print("=" * 75)
    print(f"{'Avg Median Sampson':>20}  |  Axis             |  Rotation")
    print("=" * 75)
    for (aname, rname), err in sorted(results.items(), key=lambda x: x[1]):
        marker = "  <-- BEST" if err == min(results.values()) else ""
        print(f"  {err:18.8f}  |  {aname:<16} |  {rname}{marker}")
    print("-" * 75)
    print(f"  {ref_avg:18.8f}  |  OpenCV 5pt RANSAC (reference)")
    print("=" * 75)
    print()

    best_combo = min(results, key=results.get)
    best_axis_name, best_rot_name = best_combo
    best_axis_vec = axes[best_axis_name].numpy()
    print(f">>> ANSWER: Rail axis = {best_axis_name}, rotation = {best_rot_name}")
    print()
    print("Use this in losses.py essential_from_linear_sideways():")
    print(f"    t_dev = torch.tensor({best_axis_vec.tolist()}, ...)")
    if 'R_cd^T' in best_rot_name:
        print(f"    t_cam = r_cd.t() @ t_dev")
    else:
        print(f"    t_cam = r_cd @ t_dev")

    # --- 3D interactive plot ---
    print("\nGenerating interactive 3D plot …")
    plot_linear_rail(Rbc, tbc, best_axis_vec, best_axis_name)


if __name__ == '__main__':
    find_rail_axis()
