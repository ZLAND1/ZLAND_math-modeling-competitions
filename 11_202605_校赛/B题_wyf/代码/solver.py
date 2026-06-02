"""
B题: 多源融合机器人定位及任务优化
Complete solver for all 4 problems
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/'
OUTPUT_DIR = 'output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_10hz_trajectory(t1, x1, y1, t2_aligned, x2_corr, y2_corr, s_factor=0.5):
    """
    Merge aligned data and generate 10Hz trajectory using smoothing spline.
    s_factor controls smoothing: smaller = more smoothing.
    """
    t_all = np.concatenate([t1, t2_aligned])
    x_all = np.concatenate([x1, x2_corr])
    y_all = np.concatenate([y1, y2_corr])
    idx = np.argsort(t_all)
    t_all, x_all, y_all = t_all[idx], x_all[idx], y_all[idx]

    # Remove duplicates in time (keep first)
    keep = np.ones(len(t_all), dtype=bool)
    for i in range(1, len(t_all)):
        if t_all[i] - t_all[i-1] < 1e-6:
            keep[i] = False
    t_all = t_all[keep]; x_all = x_all[keep]; y_all = y_all[keep]

    t_start = max(t1[0], t2_aligned[0])
    t_end = min(t1[-1], t2_aligned[-1])

    # Use smoothing spline to avoid oscillation
    n_pts = len(t_all)
    s = s_factor * n_pts  # smoothing parameter

    try:
        sp_x = UnivariateSpline(t_all, x_all, s=s, k=3)
        sp_y = UnivariateSpline(t_all, y_all, s=s, k=3)
    except:
        # Fallback: use Savitzky-Golay + linear interp
        window = min(51, n_pts - (n_pts % 2) - 1)
        x_all_sm = savgol_filter(x_all, window, 3)
        y_all_sm = savgol_filter(y_all, window, 3)
        sp_x = CubicSpline(t_all, x_all_sm)
        sp_y = CubicSpline(t_all, y_all_sm)

    t_10hz = np.arange(t_start, t_end + 0.05, 0.1)
    x_10hz = sp_x(t_10hz)
    y_10hz = sp_y(t_10hz)

    # Verify output sanity
    if len(t_10hz) > 10:
        v_test = np.sqrt(np.diff(x_10hz[:100])**2 + np.diff(y_10hz[:100])**2) / 0.1
        if np.median(v_test) > 1000:
            # Too fast - use more smoothing
            sp_x = UnivariateSpline(t_all, x_all, s=s*10, k=3)
            sp_y = UnivariateSpline(t_all, y_all, s=s*10, k=3)
            x_10hz = sp_x(t_10hz)
            y_10hz = sp_y(t_10hz)

    return pd.DataFrame({
        '时间(s)': t_10hz, 'X坐标(m)': x_10hz, 'Y坐标(m)': y_10hz
    })


def coarse_delta_search(cost_func, t1, t2, n_steps=300):
    """Search delta over wide range to minimize cost."""
    d_min = t2[0] - t1[-1] - 1
    d_max = t2[-1] - t1[0] + 1
    best_d, best_c = d_min, np.inf
    for d in np.linspace(d_min, d_max, n_steps):
        c = cost_func(d)
        if c < best_c:
            best_c = c; best_d = d
    return best_d


def estimate_delta_mse(t1, x1, y1, t2, x2, y2, min_overlap_frac=0.3):
    """Estimate delta using MSE with robust mean subtraction."""
    t1_span = t1[-1] - t1[0]
    t2_span = t2[-1] - t2[0]
    min_overlap = min_overlap_frac * min(t1_span, t2_span)

    d_low = t2[0] - t1[-1] + min_overlap
    d_high = t2[-1] - t1[0] - min_overlap
    if d_low > d_high:
        d_low = t2[0] - t1[-1]
        d_high = t2[-1] - t1[0]

    win = min(21, len(x1) - (len(x1) % 2) - 1)
    x1_sm = savgol_filter(x1, win, 3)
    y1_sm = savgol_filter(y1, win, 3)
    cs_x = CubicSpline(t1, x1_sm)
    cs_y = CubicSpline(t1, y1_sm)

    best_d, best_mse = None, np.inf
    for d in np.linspace(d_low, d_high, 300):
        t2s = t2 - d
        m = (t2s >= t1[0]) & (t2s <= t1[-1])
        n = m.sum()
        if n < 50:
            continue
        xp, yp = cs_x(t2s[m]), cs_y(t2s[m])
        bx_e = float(np.mean(x2[m] - xp))
        by_e = float(np.mean(y2[m] - yp))
        mse = float(np.mean((xp - (x2[m] - bx_e))**2 + (yp - (y2[m] - by_e))**2))
        if mse < best_mse:
            best_mse = mse; best_d = d
    return best_d if best_d is not None else (t2[0] - t1[0])


def joint_optimize(t1, x1, y1, t2, x2, y2, delta_init, iters=50):
    """Jointly optimize time offset delta and system bias (bx, by)."""
    win = min(21, len(x1) - (len(x1) % 2) - 1)
    x1_sm = savgol_filter(x1, win, 3)
    y1_sm = savgol_filter(y1, win, 3)
    cs_x = CubicSpline(t1, x1_sm)
    cs_y = CubicSpline(t1, y1_sm)

    delta, bx, by = delta_init, 0.0, 0.0
    for it in range(iters):
        def f(d):
            t2s = t2 - d
            m = (t2s >= t1[0]) & (t2s <= t1[-1])
            if m.sum() < 50:
                return 1e15
            return float(np.mean((cs_x(t2s[m]) - (x2[m]-bx))**2 +
                                 (cs_y(t2s[m]) - (y2[m]-by))**2))

        hw = max(10, abs(delta_init)*0.3)
        try:
            res = minimize_scalar(f, bounds=(delta-hw, delta+hw), method='bounded')
            delta = res.x
        except:
            pass

        t2s = t2 - delta
        m = (t2s >= t1[0]) & (t2s <= t1[-1])
        bx_n = float(np.mean(x2[m] - cs_x(t2s[m])))
        by_n = float(np.mean(y2[m] - cs_y(t2s[m])))
        if abs(bx-bx_n) < 1e-8 and abs(by-by_n) < 1e-8:
            bx, by = bx_n, by_n; break
        bx, by = bx_n, by_n

    t2s = t2 - delta
    m = (t2s >= t1[0]) & (t2s <= t1[-1])
    rmse = np.sqrt(np.mean((cs_x(t2s[m]) - (x2[m]-bx))**2 +
                            (cs_y(t2s[m]) - (y2[m]-by))**2))
    return delta, bx, by, rmse, m.sum()


# ============================================================
# Problem 1
# ============================================================
def solve_problem1():
    print("="*60)
    print("Problem 1: Time alignment (noiseless)")
    print("="*60)

    df1 = pd.read_excel(DATA_DIR+'附件1.xlsx', sheet_name='方式1(4Hz)')
    df2 = pd.read_excel(DATA_DIR+'附件1.xlsx', sheet_name='方式2(5Hz)')
    t1=df1['时间(s)'].values; x1=df1['X坐标(m)'].values; y1=df1['Y坐标(m)'].values
    t2=df2['时间(s)'].values; x2=df2['X坐标(m)'].values; y2=df2['Y坐标(m)'].values
    print(f"方式1: {len(t1)} pts [{t1[0]:.2f},{t1[-1]:.2f}]s")
    print(f"方式2: {len(t2)} pts [{t2[0]:.2f},{t2[-1]:.2f}]s")

    cs_x = CubicSpline(t1, x1); cs_y = CubicSpline(t1, y1)

    def cost(d):
        t2s = t2 - d
        m = (t2s >= t1[0]) & (t2s <= t1[-1])
        if m.sum() < 200:
            return 1e15
        return float(np.sum((cs_x(t2s[m])-x2[m])**2 + (cs_y(t2s[m])-y2[m])**2))

    delta0 = coarse_delta_search(cost, t1, t2, 400)
    res = minimize_scalar(cost, bounds=(delta0-30, delta0+30), method='bounded')
    delta = res.x

    t2s = t2 - delta
    m = (t2s >= t1[0]) & (t2s <= t1[-1])
    n_ov = m.sum()
    rmse = np.sqrt(np.mean((cs_x(t2s[m])-x2[m])**2 + (cs_y(t2s[m])-y2[m])**2))

    print(f"δ = {delta:.6f}s")
    print(f"RMSE = {rmse:.10f}m (should be ~0 for noiseless data)")
    print(f"Overlap = {n_ov} pts")

    # For noiseless data, use exact spline (no smoothing needed)
    traj = generate_10hz_trajectory(t1, x1, y1, t2-delta, x2, y2, s_factor=0.0)
    traj.to_csv(OUTPUT_DIR+'problem1_10hz_traj.csv', index=False)
    print(f"10Hz traj: {len(traj)} pts\n")
    return {'delta': delta, 'rmse': rmse, 'n_overlap': n_ov, 'traj_10hz': traj}


# ============================================================
# Problem 2
# ============================================================
def solve_problem2():
    print("="*60)
    print("Problem 2: Noise + system bias")
    print("="*60)

    df1 = pd.read_excel(DATA_DIR+'附件2.xlsx', sheet_name='方式1(4Hz)')
    df2 = pd.read_excel(DATA_DIR+'附件2.xlsx', sheet_name='方式2(5Hz)')
    t1=df1['时间(s)'].values; x1=df1['X坐标(m)'].values; y1=df1['Y坐标(m)'].values
    t2=df2['时间(s)'].values; x2=df2['X坐标(m)'].values; y2=df2['Y坐标(m)'].values
    print(f"方式1: {len(t1)} pts [{t1[0]:.2f},{t1[-1]:.2f}]s")
    print(f"方式2: {len(t2)} pts [{t2[0]:.2f},{t2[-1]:.2f}]s")

    delta0 = estimate_delta_mse(t1, x1, y1, t2, x2, y2, min_overlap_frac=0.3)
    print(f"Initial delta: {delta0:.2f}s")
    delta, bx, by, rmse, n_ov = joint_optimize(t1, x1, y1, t2, x2, y2, delta0)

    print(f"δ = {delta:.6f}s")
    print(f"b = ({bx:.6f}, {by:.6f}) m")
    print(f"RMSE = {rmse:.6f}m, overlap = {n_ov} pts")

    traj = generate_10hz_trajectory(t1, x1, y1, t2-delta, x2-bx, y2-by, s_factor=0.2)
    traj.to_csv(OUTPUT_DIR+'problem2_10hz_traj.csv', index=False)
    print(f"10Hz traj: {len(traj)} pts\n")
    return {'delta': delta, 'bias': (bx, by), 'rmse': rmse,
            'n_overlap': n_ov, 'traj_10hz': traj}


# ============================================================
# Problem 3
# ============================================================
def solve_problem3():
    print("="*60)
    print("Problem 3: Real data + bias detection")
    print("="*60)

    df1 = pd.read_excel(DATA_DIR+'附件3.xlsx', sheet_name='方式1(4Hz)')
    df2 = pd.read_excel(DATA_DIR+'附件3.xlsx', sheet_name='方式2(5Hz)')
    t1=df1['时间(s)'].values; x1=df1['X坐标(m)'].values; y1=df1['Y坐标(m)'].values
    t2=df2['时间(s)'].values; x2=df2['X坐标(m)'].values; y2=df2['Y坐标(m)'].values
    print(f"方式1: {len(t1)} pts [{t1[0]:.2f},{t1[-1]:.2f}]s")
    print(f"方式2: {len(t2)} pts [{t2[0]:.2f},{t2[-1]:.2f}]s")

    win = min(21, len(x1)-(len(x1)%2)-1)
    x1_sm = savgol_filter(x1, win, 3)
    y1_sm = savgol_filter(y1, win, 3)
    cs_x = CubicSpline(t1, x1_sm); cs_y = CubicSpline(t1, y1_sm)

    # Step 1: Delta without bias
    delta0 = estimate_delta_mse(t1, x1, y1, t2, x2, y2, min_overlap_frac=0.3)

    def cost_nb(d):
        t2s = t2 - d
        m = (t2s>=t1[0]) & (t2s<=t1[-1])
        if m.sum() < 50: return 1e15
        return float(np.mean((cs_x(t2s[m])-x2[m])**2 + (cs_y(t2s[m])-y2[m])**2))

    res = minimize_scalar(cost_nb, bounds=(delta0-50, delta0+50), method='bounded')
    delta_nb = res.x

    # Step 2: Bias test
    t2s = t2 - delta_nb
    m = (t2s>=t1[0]) & (t2s<=t1[-1])
    dx = cs_x(t2s[m]) - x2[m]
    dy = cs_y(t2s[m]) - y2[m]

    t_x, p_x = stats.ttest_1samp(dx, 0.0)
    t_y, p_y = stats.ttest_1samp(dy, 0.0)
    has_bias = (p_x < 0.01) or (p_y < 0.01)

    print(f"\nStep 1: δ₀ = {delta_nb:.4f}s, overlap={m.sum()}")
    print(f"Step 2: Bias test")
    print(f"  mean(dx)={np.mean(dx):.4f}, mean(dy)={np.mean(dy):.4f}")
    print(f"  t_x={t_x:.4f}, p_x={p_x:.6e} -> {'**BIAS**' if p_x<0.01 else 'ok'}")
    print(f"  t_y={t_y:.4f}, p_y={p_y:.6e} -> {'**BIAS**' if p_y<0.01 else 'ok'}")
    print(f"  => {'BIAS DETECTED' if has_bias else 'No significant bias'}")

    # Step 3: Joint estimation
    delta, bx, by, rmse, n_ov = joint_optimize(t1, x1, y1, t2, x2, y2, delta_nb)

    print(f"\nStep 3: Joint estimation")
    print(f"  δ = {delta:.6f}s")
    print(f"  b = ({bx:.6f}, {by:.6f}) m")
    print(f"  RMSE = {rmse:.6f}m, overlap = {n_ov} pts")

    traj = generate_10hz_trajectory(t1, x1, y1, t2-delta, x2-bx, y2-by, s_factor=0.3)
    traj.to_csv(OUTPUT_DIR+'problem3_10hz_traj.csv', index=False)
    print(f"10Hz traj: {len(traj)} pts\n")
    return {
        'delta': delta, 'bias': (bx, by), 'rmse': rmse,
        'has_bias': has_bias, 'p_val_x': p_x, 'p_val_y': p_y,
        'mean_dx': np.mean(dx), 'mean_dy': np.mean(dy),
        'traj_10hz': traj
    }


# ============================================================
# Problem 4: Task scheduling
# ============================================================
def solve_problem4(traj_10hz):
    print("="*60)
    print("Problem 4: Task scheduling optimization")
    print("="*60)

    shoots = pd.read_excel(DATA_DIR+'附件4.xlsx', sheet_name='射击目标')
    photos = pd.read_excel(DATA_DIR+'附件4.xlsx', sheet_name='拍照目标')
    print(f"Targets: {len(shoots)} shooting, {len(photos)} photo")

    t = traj_10hz['时间(s)'].values
    x = traj_10hz['X坐标(m)'].values
    y = traj_10hz['Y坐标(m)'].values

    if len(t) < 20:
        print("WARNING: trajectory too short, using 方式1 raw data")
        df1 = pd.read_excel(DATA_DIR+'附件3.xlsx', sheet_name='方式1(4Hz)')
        t_raw=df1['时间(s)'].values; x_raw=df1['X坐标(m)'].values; y_raw=df1['Y坐标(m)'].values
        cs_x=CubicSpline(t_raw, x_raw); cs_y=CubicSpline(t_raw, y_raw)
        t=np.arange(t_raw[0], t_raw[-1]+0.05, 0.1)
        x=cs_x(t); y=cs_y(t)

    dt = np.mean(np.diff(t))
    vx = np.gradient(x, dt); vy = np.gradient(y, dt)
    v = np.sqrt(vx**2 + vy**2)
    ax = np.gradient(vx, dt); ay = np.gradient(vy, dt)
    a = np.sqrt(ax**2 + ay**2)
    heading = np.arctan2(vy, vx)

    # Diagnostic
    v05, v15, v25, v35, v50 = [np.percentile(v, p) for p in [5,15,25,35,50]]
    a05, a15, a25, a35, a50 = [np.percentile(a, p) for p in [5,15,25,35,50]]
    print(f"Traj stats: v(p5)={v05:.1f} v(p15)={v15:.1f} v(p25)={v25:.1f} "
          f"v(p35)={v35:.1f} v(p50)={v50:.1f} m/s")
    print(f"  a(p5)={a05:.1f} a(p15)={a15:.1f} a(p25)={a25:.1f} "
          f"a(p35)={a35:.1f} a(p50)={a50:.1f} m/s²")

    # Reasonable constraints
    # Shooting: wider distance range, speed <= 50th percentile, accel <= 60th percentile
    S_DIST_MIN, S_DIST_MAX = 3.0, 55.0
    S_VEL_MAX = max(30.0, float(np.percentile(v, 50)))
    S_ACC_MAX = max(150.0, float(np.percentile(a, 60)))
    S_PREP_PTS = int(1.5 / dt)

    # Photo: closer distance min, speed <= 35th percentile, accel <= 50th percentile
    P_DIST_MIN, P_DIST_MAX = 2.0, 55.0
    P_VEL_MAX = max(20.0, float(np.percentile(v, 35)))
    P_ACC_MAX = max(100.0, float(np.percentile(a, 50)))
    P_ANGLE_DIFF = np.radians(30)
    P_PREP_PTS = int(0.5 / dt)

    print(f"Shooting: d∈[{S_DIST_MIN},{S_DIST_MAX}]m, v≤{S_VEL_MAX:.1f}, a≤{S_ACC_MAX:.1f}, prep=1.5s")
    print(f"Photo:   d∈[{P_DIST_MIN},{P_DIST_MAX}]m, v≤{P_VEL_MAX:.1f}, a≤{P_ACC_MAX:.1f}, Δθ≥30°, prep=0.5s")

    # Check overall feasibility
    s_feas = np.mean((v <= S_VEL_MAX) & (a <= S_ACC_MAX)) * 100
    p_feas = np.mean((v <= P_VEL_MAX) & (a <= P_ACC_MAX)) * 100
    print(f"Global feasibility: shooting={s_feas:.1f}%, photo={p_feas:.1f}%")

    def find_windows(tx, ty, dmin, dmax, vmax, amax, prep_pts):
        dist = np.sqrt((x-tx)**2 + (y-ty)**2)
        feasible = (dist>=dmin) & (dist<=dmax) & (v<=vmax) & (a<=amax)
        windows = []
        i, n = 0, len(feasible)
        while i < n:
            if feasible[i]:
                ps = i - prep_pts
                if ps >= 0 and np.all(feasible[ps:i+1]):
                    j = i
                    while j+1 < n and feasible[j+1]:
                        j += 1
                    windows.append({
                        'start': float(t[ps]), 'exec': float(t[i]),
                        'end': float(t[j]),
                        'dist': float(np.mean(dist[i:j+1])),
                        'angle': float(np.mean(heading[i:j+1]))
                    })
                    i = j + 1
                    continue
            i += 1
        return windows

    all_jobs = []
    print("\nTask windows found:")
    for _, row in shoots.iterrows():
        tid=row['编号']; tx=row['X坐标(m)']; ty=row['Y坐标(m)']
        ws = find_windows(tx, ty, S_DIST_MIN, S_DIST_MAX, S_VEL_MAX, S_ACC_MAX, S_PREP_PTS)
        for w in ws:
            w.update({'target': tid, 'type': '射击', 'weight': 1.0})
        all_jobs.extend(ws)
        if ws: print(f"  {tid}: {len(ws)} shoot windows")

    for _, row in photos.iterrows():
        tid=row['编号']; tx=row['X坐标(m)']; ty=row['Y坐标(m)']
        ws = find_windows(tx, ty, P_DIST_MIN, P_DIST_MAX, P_VEL_MAX, P_ACC_MAX, P_PREP_PTS)
        ws.sort(key=lambda w: w['start'])
        sel = []
        for w in ws:
            ok = True
            for s in sel:
                d = abs(w['angle']-s['angle'])
                if min(d, 2*np.pi-d) < P_ANGLE_DIFF:
                    ok = False; break
            if ok:
                w.update({'target': tid, 'type': '拍照', 'weight': 1.0})
                sel.append(w)
        all_jobs.extend(sel)
        if sel: print(f"  {tid}: {len(sel)} photo windows")

    total_shoot = sum(1 for j in all_jobs if j['type']=='射击')
    total_photo = sum(1 for j in all_jobs if j['type']=='拍照')
    print(f"\nTotal: {total_shoot} shoot + {total_photo} photo = {len(all_jobs)} candidates")

    if not all_jobs:
        print("No feasible tasks! Try relaxing constraints.")
        return None

    # DP weighted interval scheduling
    jobs = [(j['start'], j['end'], j['weight'], j) for j in all_jobs]
    jobs.sort(key=lambda j: j[1])
    n = len(jobs)
    starts = np.array([j[0] for j in jobs])
    ends = np.array([j[1] for j in jobs])

    p = np.full(n, -1, dtype=int)
    for j in range(n):
        lo, hi = 0, j-1; ans = -1
        while lo <= hi:
            mid = (lo+hi)//2
            if ends[mid] <= starts[j]:
                ans = mid; lo = mid+1
            else:
                hi = mid-1
        p[j] = ans

    dp = np.zeros(n+1)
    for j in range(1, n+1):
        profit = jobs[j-1][2]
        prev = dp[p[j-1]+1] if p[j-1]>=0 else 0
        dp[j] = max(dp[j-1], profit+prev)

    selected = []
    j = n
    while j > 0:
        profit = jobs[j-1][2]
        prev = p[j-1]+1 if p[j-1]>=0 else 0
        if profit + dp[prev] >= dp[j-1]:
            selected.append(jobs[j-1]); j = prev
        else:
            j -= 1
    selected.reverse()

    sn = sum(1 for s in selected if s[3]['type']=='射击')
    pn = sum(1 for s in selected if s[3]['type']=='拍照')
    print(f"\nOptimized: {len(selected)} tasks ({sn} shoot + {pn} photo)")
    for i, (st, en, w, info) in enumerate(selected):
        print(f"  {i+1:2d}. {info['target']} {info['type']} "
              f"准备={st:.2f}s 执行={info['exec']:.2f}s 结束={en:.2f}s d={info['dist']:.1f}m")

    fill_result_xlsx(selected)
    return selected


def fill_result_xlsx(selected_jobs):
    import openpyxl
    from openpyxl.styles import Font, Alignment, Border, Side

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '任务调度结果'

    # Title
    ws.merge_cells('A1:E1')
    ws['A1'] = '任务调度优化结果'
    ws['A1'].font = Font(bold=True, size=14)
    ws['A1'].alignment = Alignment(horizontal='center')

    # Header
    headers = ['序号', '目标编号', '任务', '开始准备时刻(s)', '任务执行时刻(s)']
    for j, h in enumerate(headers, 1):
        cell = ws.cell(row=3, column=j, value=h)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')

    # Data
    for i, (st, en, w, info) in enumerate(selected_jobs):
        row = 4 + i
        ws.cell(row=row, column=1, value=i+1).alignment = Alignment(horizontal='center')
        ws.cell(row=row, column=2, value=info['target']).alignment = Alignment(horizontal='center')
        ws.cell(row=row, column=3, value=info['type']).alignment = Alignment(horizontal='center')
        ws.cell(row=row, column=4, value=round(info['start'], 2))
        ws.cell(row=row, column=5, value=round(info['exec'], 2))

    # Column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 8
    ws.column_dimensions['D'].width = 18
    ws.column_dimensions['E'].width = 18

    wb.save(OUTPUT_DIR+'result.xlsx')
    print(f"\nResult: {OUTPUT_DIR}result.xlsx")


def plot_all(r1, r2, r3, r4):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    # P1
    ax=axes[0,0]
    d1=pd.read_excel(DATA_DIR+'附件1.xlsx',sheet_name='方式1(4Hz)')
    d2=pd.read_excel(DATA_DIR+'附件1.xlsx',sheet_name='方式2(5Hz)')
    ax.plot(d1['X坐标(m)'],d1['Y坐标(m)'],'b.',ms=0.5,alpha=0.3,label='方式1')
    ax.plot(d2['X坐标(m)'],d2['Y坐标(m)'],'r.',ms=0.5,alpha=0.3,label='方式2')
    ax.plot(r1['traj_10hz']['X坐标(m)'],r1['traj_10hz']['Y坐标(m)'],'g-',lw=0.6,label='10Hz')
    ax.set_title(f'P1: δ={r1["delta"]:.2f}s RMSE={r1["rmse"]:.2e}m',fontsize=9)
    ax.legend(fontsize=6); ax.axis('equal')

    # P2
    ax=axes[0,1]
    d1=pd.read_excel(DATA_DIR+'附件2.xlsx',sheet_name='方式1(4Hz)')
    d2=pd.read_excel(DATA_DIR+'附件2.xlsx',sheet_name='方式2(5Hz)')
    ax.plot(d1['X坐标(m)'],d1['Y坐标(m)'],'b.',ms=0.3,alpha=0.2,label='方式1')
    ax.plot(d2['X坐标(m)'],d2['Y坐标(m)'],'r.',ms=0.3,alpha=0.2,label='方式2')
    ax.plot(r2['traj_10hz']['X坐标(m)'],r2['traj_10hz']['Y坐标(m)'],'g-',lw=0.6,label='10Hz')
    ax.set_title(f'P2: δ={r2["delta"]:.2f}s b=({r2["bias"][0]:.2f},{r2["bias"][1]:.2f})m RMSE={r2["rmse"]:.2f}m',fontsize=9)
    ax.legend(fontsize=6); ax.axis('equal')

    # P3
    ax=axes[1,0]
    d1=pd.read_excel(DATA_DIR+'附件3.xlsx',sheet_name='方式1(4Hz)')
    d2=pd.read_excel(DATA_DIR+'附件3.xlsx',sheet_name='方式2(5Hz)')
    ax.plot(d1['X坐标(m)'],d1['Y坐标(m)'],'b.',ms=0.8,alpha=0.3,label='方式1')
    ax.plot(d2['X坐标(m)'],d2['Y坐标(m)'],'r.',ms=0.8,alpha=0.3,label='方式2')
    ax.plot(r3['traj_10hz']['X坐标(m)'],r3['traj_10hz']['Y坐标(m)'],'g-',lw=0.6,label='10Hz')
    ax.set_title(f'P3: δ={r3["delta"]:.2f}s bias={"YES" if r3["has_bias"] else "NO"} RMSE={r3["rmse"]:.2f}m',fontsize=9)
    ax.legend(fontsize=6); ax.axis('equal')

    # P4
    ax=axes[1,1]
    t10 = r3['traj_10hz']
    ax.plot(t10['X坐标(m)'],t10['Y坐标(m)'],'gray',lw=0.3,alpha=0.5)
    sh=pd.read_excel(DATA_DIR+'附件4.xlsx',sheet_name='射击目标')
    ph=pd.read_excel(DATA_DIR+'附件4.xlsx',sheet_name='拍照目标')
    ax.scatter(sh['X坐标(m)'],sh['Y坐标(m)'],c='red',marker='^',s=25,label='Shoot')
    ax.scatter(ph['X坐标(m)'],ph['Y坐标(m)'],c='blue',marker='s',s=25,label='Photo')
    if r4:
        for st,en,w,info in r4:
            tgt=sh[sh['编号']==info['target']] if info['type']=='射击' else ph[ph['编号']==info['target']]
            if len(tgt)>0:
                c='orange' if info['type']=='射击' else 'lime'
                ax.plot(tgt.iloc[0]['X坐标(m)'],tgt.iloc[0]['Y坐标(m)'],
                        'o',ms=8,mfc=c,mec='black',mew=0.5)
    ax.set_title(f'P4: {len(r4) if r4 else 0} tasks',fontsize=9)
    ax.legend(fontsize=6); ax.axis('equal')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR+'all_results.png',dpi=150); plt.close()
    print(f"Figure: {OUTPUT_DIR}all_results.png")


if __name__ == '__main__':
    print("="*60)
    print("B题 多源融合机器人定位及任务优化")
    print("="*60+"\n")

    r1 = solve_problem1()
    r2 = solve_problem2()
    r3 = solve_problem3()
    r4 = solve_problem4(r3['traj_10hz'])

    print("\n"+"="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"P1: δ={r1['delta']:.4f}s, RMSE={r1['rmse']:.8f}m, overlap={r1['n_overlap']}")
    print(f"P2: δ={r2['delta']:.4f}s, b=({r2['bias'][0]:.4f},{r2['bias'][1]:.4f})m, RMSE={r2['rmse']:.4f}m")
    print(f"P3: δ={r3['delta']:.4f}s, b=({r3['bias'][0]:.4f},{r3['bias'][1]:.4f})m, "
          f"has_bias={r3['has_bias']}, RMSE={r3['rmse']:.4f}m")
    if r4:
        sn=sum(1 for s in r4 if s[3]['type']=='射击')
        pn=sum(1 for s in r4 if s[3]['type']=='拍照')
        print(f"P4: {sn} shooting + {pn} photo = {len(r4)} tasks")

    try: plot_all(r1, r2, r3, r4)
    except Exception as e: print(f"Plot error: {e}")

    print(f"\nOutputs: {OUTPUT_DIR}")
