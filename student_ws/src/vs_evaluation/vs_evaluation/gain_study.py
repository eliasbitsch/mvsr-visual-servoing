"""Gain study: IBVS gain lambda vs convergence time and overshoot.

    $VS_PYTHON -m vs_evaluation.gain_study --campaign /root/vs_data/eval/final

Reads the gain_* blocks of the campaign (run_eval --gain) and main_dino_sam2 as the default gain;
the gain comes from `ibvs_gain` in each block's config.json, for an older main_dino_sam2 without
it from servo.yaml. Writes gain_study.csv (one row per run),
gain_study_summary.csv (one row per gain and target) and gain_study.png into the campaign folder.

Overshoot is taken from the ground truth, per axis: how far the camera goes past the reference
pose, against the sign of the initial error, relative to that initial error. Axes with an initial
error below MIN_AXIS_ERR are skipped (their "overshoot" would be noise divided by almost zero).
For blue_box the yaw is included the same way (feature error e_angle, rad). The relative value
inflates on axes that start just above MIN_AXIS_ERR, so the plot shows the absolute overshoot in
mm (largest over x, y, z).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_GAIN = 0.6           # servo.yaml
DEFAULT_BLOCK = 'main_dino_sam2'
MIN_AXIS_ERR = 0.005         # m
MIN_YAW_ERR = np.radians(2)  # rad
HOLD_TIME = 1.0              # servo.yaml converged_hold_time
AXES = ['gt_ex', 'gt_ey', 'gt_ez']


def axis_overshoot(series, min_initial):
    """Fraction past zero against the initial sign, nan if the axis starts (almost) at zero."""
    x = series.dropna().to_numpy()
    if len(x) < 2 or abs(x[0]) < min_initial:
        return np.nan
    return max(0.0, float(np.max(-np.sign(x[0]) * x))) / abs(x[0])


def axis_overshoot_abs(series, min_initial):
    """Distance past zero against the initial sign, in the series' unit. Same lower bound as the
    relative value: on an axis that starts at ~0 any sideways drift would count as overshoot."""
    x = series.dropna().to_numpy()
    if len(x) < 2 or abs(x[0]) < min_initial:
        return np.nan
    return max(0.0, float(np.max(-np.sign(x[0]) * x)))


def run_metrics(csv_path, with_yaw):
    s = pd.read_csv(csv_path)
    per_axis = {a: axis_overshoot(s[a], MIN_AXIS_ERR) for a in AXES}
    if with_yaw:
        per_axis['yaw'] = axis_overshoot(s['e_angle'], MIN_YAW_ERR)
    vals = [v for v in per_axis.values() if not np.isnan(v)]
    # Relative overshoot inflates on an axis that starts just above MIN_AXIS_ERR (4 mm past
    # a 6 mm start reads 64 %); the absolute value in mm is the one to compare across gains
    abs_mm = [1000 * v for v in (axis_overshoot_abs(s[a], MIN_AXIS_ERR) for a in AXES)
              if not np.isnan(v)]
    # Peak camera speed: shows where the speed limit (max_linear_speed) takes over from lambda
    v = s[['v_x', 'v_y', 'v_z']].dropna().to_numpy()
    return {'overshoot': max(vals) if vals else np.nan,
            'overshoot_mm': max(abs_mm) if abs_mm else np.nan,
            **{f'os_{k.replace("gt_e", "")}': v_ for k, v_ in per_axis.items()},
            'initial_err_m': float(s['gt_et'].dropna().iloc[0]) if s['gt_et'].notna().any() else np.nan,
            'peak_speed': float(np.linalg.norm(v, axis=1).max()) if len(v) else np.nan}


def block_gain(block_dir):
    cfg_path = block_dir / 'config.json'
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    if cfg.get('ibvs_gain') is not None:
        return float(cfg['ibvs_gain'])
    if cfg.get('gain') is not None:
        return float(cfg['gain'])
    return DEFAULT_GAIN if block_dir.name == DEFAULT_BLOCK else None


def collect(campaign):
    rows = []
    # Only the gain blocks and the main block: every other block changes more than the gain
    # (detector, occlusion, latency, ...), even though its config.json records the gain too
    blocks = [p for p in Path(campaign).iterdir()
              if p.is_dir() and (p.name.startswith('gain_') or p.name == DEFAULT_BLOCK)]
    for block_dir in sorted(blocks):
        gain = block_gain(block_dir)
        if gain is None or not (block_dir / 'summary.csv').exists():
            continue
        summary = pd.read_csv(block_dir / 'summary.csv')
        # run_eval before 779e806 counted the controller's converged_hold_time (1 s) into the
        # convergence time; those summaries lack conv_time_sim_s. Same definition for all blocks:
        hold = HOLD_TIME if 'conv_time_sim_s' not in summary.columns else 0.0
        for _, r in summary.iterrows():
            run_csv = block_dir / 'runs' / f'{r["target"]}_S{r["start"]}.csv'
            row = {'block': block_dir.name, 'gain': gain, 'target': r['target'],
                   'start': r['start'], 'outcome': r['outcome'], 'success': r['success'],
                   # clip keeps nan (not converged) as nan; max(0.0, nan) would give 0.0
                   'conv_time_s': np.clip(r['conv_time_s'] - hold, 0.0, None),
                   'final_pos_err_m': r.get('final_pos_err_m')}
            if run_csv.exists() and r['outcome'] not in ('no reference', 'not detected'):
                row.update(run_metrics(run_csv, with_yaw=r['target'] == 'blue_box'))
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # main_dino_sam2 covers all targets; compare only those the gain blocks ran on
    gain_targets = set(df.loc[df['block'] != DEFAULT_BLOCK, 'target'])
    return df[df['target'].isin(gain_targets)] if gain_targets else df


def summarise(df):
    def q(p):
        return lambda x: x.quantile(p)
    g = df.groupby(['gain', 'target'])
    return g.agg(runs=('start', 'size'), success=('success', 'mean'),
                 conv_median=('conv_time_s', 'median'), conv_q25=('conv_time_s', q(0.25)),
                 conv_q75=('conv_time_s', q(0.75)),
                 overshoot_median=('overshoot', 'median'), overshoot_max=('overshoot', 'max'),
                 overshoot_mm_median=('overshoot_mm', 'median'),
                 overshoot_mm_max=('overshoot_mm', 'max'),
                 final_err_mm_median=('final_pos_err_m', lambda x: 1000 * x.median()),
                 peak_speed_median=('peak_speed', 'median')).reset_index()


def plot(summary, path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for target, d in summary.groupby('target'):
        d = d.sort_values('gain')
        ax1.errorbar(d['gain'], d['conv_median'],
                     yerr=[d['conv_median'] - d['conv_q25'], d['conv_q75'] - d['conv_median']],
                     marker='o', capsize=3, label=target)
        ax2.plot(d['gain'], d['overshoot_mm_median'], marker='o', label=f'{target} median')
        ax2.plot(d['gain'], d['overshoot_mm_max'], marker='x', ls='--', label=f'{target} max')
    ax1.set(xlabel='gain λ', ylabel='convergence time [s]', title='Convergence time (median, IQR)')
    ax2.set(xlabel='gain λ', ylabel='overshoot [mm]', title='Overshoot past the reference (ground truth)')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, help='campaign folder, e.g. data/eval/final')
    args = ap.parse_args()
    out = Path(args.campaign)
    df = collect(out)
    if df.empty:
        raise SystemExit(f'no gain blocks with results in {out}')
    df.to_csv(out / 'gain_study.csv', index=False)
    summary = summarise(df)
    summary.to_csv(out / 'gain_study_summary.csv', index=False)
    with pd.option_context('display.width', 160, 'display.precision', 3):
        print(summary.to_string(index=False))
    import matplotlib
    matplotlib.use('Agg')  # CLI only: the notebook keeps its inline backend
    plot(summary, out / 'gain_study.png')
    print(f'written to {out}/gain_study*.csv, gain_study.png')


if __name__ == '__main__':
    main()
