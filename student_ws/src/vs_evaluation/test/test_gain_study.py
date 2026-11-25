import json

import numpy as np
import pytest

pd = pytest.importorskip('pandas')  # in the venv ($VS_PYTHON), not in the system python

from vs_evaluation import gain_study as gs  # noqa: E402


def test_overshoot_skips_axes_that_start_near_zero():
    assert np.isnan(gs.axis_overshoot(pd.Series([0.004, -0.003, 0.0]), 0.005))
    assert np.isnan(gs.axis_overshoot_abs(pd.Series([0.004, -0.003, 0.0]), 0.005))


def test_overshoot_relative_and_absolute():
    x = pd.Series([0.010, np.nan, 0.004, -0.002, -0.001])
    assert np.isclose(gs.axis_overshoot(x, 0.005), 0.2)
    assert np.isclose(gs.axis_overshoot_abs(x, 0.005), 0.002)
    # Negative start: the overshoot is the positive excursion
    assert np.isclose(gs.axis_overshoot_abs(-x, 0.005), 0.002)


def test_no_overshoot_on_a_monotonic_approach():
    assert gs.axis_overshoot(pd.Series([0.02, 0.01, 0.001]), 0.005) == 0.0


def write_block(campaign, name, config, summary, series):
    d = campaign / name
    (d / 'runs').mkdir(parents=True)
    (d / 'config.json').write_text(json.dumps(config))
    pd.DataFrame(summary).to_csv(d / 'summary.csv', index=False)
    for r in summary:
        series.to_csv(d / 'runs' / f'{r["target"]}_S{r["start"]}.csv', index=False)


def run_series():
    n = 5
    return pd.DataFrame({'gt_ex': [0.02, 0.01, 0.0, -0.001, 0.0], 'gt_ey': [0.001] * n,
                         'gt_ez': [0.0] * n, 'e_angle': [0.0] * n,
                         'gt_et': [0.02, 0.01, 0.001, 0.001, 0.0],
                         'v_x': [0.1, 0.05, 0, 0, 0], 'v_y': [0.0] * n, 'v_z': [0.0] * n})


def row(start, conv, outcome='converged', **extra):
    return {'target': 'red_can', 'start': start, 'outcome': outcome,
            'success': int(outcome == 'converged'), 'conv_time_s': conv,
            'final_pos_err_m': 0.004, **extra}


def test_collect(tmp_path):
    s = run_series()
    # Old main block: no conv_time_sim_s column, so the 1 s hold is taken off
    write_block(tmp_path, 'main_dino_sam2', {}, [row(1, 5.0), row(2, np.nan, 'timeout')], s)
    write_block(tmp_path, 'gain_1p0', {'ibvs_gain': 1.0}, [row(1, 3.0, conv_time_sim_s=3.0)], s)
    write_block(tmp_path, 'gainlc_1p0',
                {'ibvs_gain': 1.0, 'ibvs_params': {'gain': 1.0, 'latency_compensation': 0.1}},
                [row(1, 2.5, conv_time_sim_s=2.5)], s)
    write_block(tmp_path, 'det_owlv2', {'ibvs_gain': 0.6}, [row(1, 4.0)], s)  # not a gain block

    df = gs.collect(tmp_path).set_index(['block', 'start'])
    assert set(df.index.get_level_values('block')) == {'main_dino_sam2', 'gain_1p0', 'gainlc_1p0'}
    assert df.loc[('main_dino_sam2', 1), 'gain'] == gs.DEFAULT_GAIN
    assert df.loc[('main_dino_sam2', 1), 'conv_time_s'] == 4.0
    assert np.isnan(df.loc[('main_dino_sam2', 2), 'conv_time_s'])   # nan stays nan, not 0
    assert df.loc[('gain_1p0', 1), 'conv_time_s'] == 3.0
    assert df.loc[('gain_1p0', 1), 'latency_comp'] == 0.0
    assert df.loc[('gainlc_1p0', 1), 'latency_comp'] == 0.1
    # x starts at 20 mm and goes 1 mm past; y starts at 1 mm and is skipped
    assert np.isclose(df.loc[('gain_1p0', 1), 'overshoot_mm'], 1.0)
    assert np.isnan(df.loc[('gain_1p0', 1), 'os_y'])

    summary = gs.summarise(gs.collect(tmp_path))
    assert len(summary) == 3   # gain 0.6, 1.0 without and 1.0 with compensation
