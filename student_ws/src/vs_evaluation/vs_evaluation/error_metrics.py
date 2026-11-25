"""Error metrics that do not depend on the controller's stop criterion.

    $VS_PYTHON -m vs_evaluation.error_metrics --campaign /root/vs_data/eval/final

`success` in summary.csv mostly measures where the controller stops: it ends a run once the
feature error stays below converged_threshold (0.01) for converged_hold_time, which is about
4-5 mm of position error at the default gain. A run that would have gone on to 1 mm and one that
stalls at 4.9 mm both count the same, and the final error clusters just under 5 mm.

These metrics come from the ground-truth position error over time (runs/<target>_S<n>.csv):
  t_<N>mm     first time the error falls below N mm (10, 5, 2), nan if never
  min_err_mm  smallest error reached during the run
  iae_mm_s    integral of the error over the first HORIZON seconds (mm*s). After a converged run
              the arm stands still, so its last error is held until HORIZON; this way a fast run
              is not rewarded just for ending early
Lost-target tests and runs without a reference or detection are left out.

Writes error_metrics.csv (one row per run) and error_metrics_summary.csv (one row per block and
target) into the campaign folder.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS_MM = (10, 5, 2)
HORIZON = 15.0  # s, longer than every converged run of the main configuration
SKIP_OUTCOMES = ('no reference', 'not detected', 'lost test')


def first_below(t, err, threshold):
    """First time err < threshold, nan if never."""
    idx = np.flatnonzero(err < threshold)
    return float(t[idx[0]]) if len(idx) else np.nan


def iae(t, err, horizon, hold_last):
    """Integral of err over [t0, t0 + horizon]. hold_last: extend the last value to the horizon
    (the arm stands after a converged run), else integrate only the recorded part."""
    t = np.asarray(t, dtype=float) - t[0]
    err = np.asarray(err, dtype=float)
    if t[-1] > horizon:  # cut exactly at the horizon, the value there interpolated
        end = np.interp(horizon, t, err)
        keep = t < horizon
        t, err = np.append(t[keep], horizon), np.append(err[keep], end)
    elif hold_last and t[-1] < horizon:
        t, err = np.append(t, horizon), np.append(err, err[-1])
    return float(np.sum(np.diff(t) * (err[1:] + err[:-1]) / 2))


def series_metrics(series, converged):
    """Metrics of one run from its time series (columns t, gt_et in m)."""
    s = series[['t', 'gt_et']].dropna()
    if len(s) < 2:
        return {}
    t, err_mm = s['t'].to_numpy(), 1000 * s['gt_et'].to_numpy()
    out = {f't_{n}mm': first_below(t - t[0], err_mm, n) for n in THRESHOLDS_MM}
    out['min_err_mm'] = float(err_mm.min())
    out['iae_mm_s'] = iae(t, err_mm, HORIZON, hold_last=bool(converged))
    return out


def collect(campaign):
    rows = []
    for block_dir in sorted(p for p in Path(campaign).iterdir() if p.is_dir()):
        summary_path = block_dir / 'summary.csv'
        if not summary_path.exists():
            continue
        for _, r in pd.read_csv(summary_path).iterrows():
            if r['outcome'] in SKIP_OUTCOMES:
                continue
            run_csv = block_dir / 'runs' / f'{r["target"]}_S{r["start"]}.csv'
            if not run_csv.exists():
                continue
            m = series_metrics(pd.read_csv(run_csv), r['converged'] == 1)
            if m:
                rows.append({'block': block_dir.name, 'target': r['target'], 'start': r['start'],
                             'outcome': r['outcome'], 'success': r['success'],
                             'final_err_mm': 1000 * r['final_pos_err_m'], **m})
    return pd.DataFrame(rows)


def summarise(df):
    agg = {'runs': ('start', 'size'), 'success': ('success', 'mean'),
           'final_err_mm_median': ('final_err_mm', 'median'),
           'min_err_mm_median': ('min_err_mm', 'median'),
           'iae_mm_s_median': ('iae_mm_s', 'median')}
    for n in THRESHOLDS_MM:
        # Share of runs that got below n mm at all, and the median time for those that did
        agg[f'reached_{n}mm'] = (f't_{n}mm', lambda x: x.notna().mean())
        agg[f't_{n}mm_median'] = (f't_{n}mm', 'median')
    return df.groupby(['block', 'target']).agg(**agg).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, help='campaign folder, e.g. data/eval/final')
    args = ap.parse_args()
    out = Path(args.campaign)
    df = collect(out)
    if df.empty:
        raise SystemExit(f'no runs with time series in {out}')
    df.to_csv(out / 'error_metrics.csv', index=False)
    summary = summarise(df)
    summary.to_csv(out / 'error_metrics_summary.csv', index=False)
    with pd.option_context('display.width', 200, 'display.precision', 2,
                           'display.max_rows', 200):
        print(summary.to_string(index=False))
    print(f'written to {out}/error_metrics*.csv')


if __name__ == '__main__':
    main()
