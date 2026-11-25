import numpy as np
import pytest

pd = pytest.importorskip('pandas')  # in the venv ($VS_PYTHON), not in the system python

from vs_evaluation import error_metrics as em  # noqa: E402


def test_first_below():
    t = np.array([0.0, 1.0, 2.0])
    assert em.first_below(t, np.array([12.0, 8.0, 4.0]), 10) == 1.0
    assert np.isnan(em.first_below(t, np.array([12.0, 11.0, 10.5]), 10))


def test_iae_holds_the_last_value_only_for_converged_runs():
    t, err = np.array([0.0, 2.0]), np.array([10.0, 2.0])
    assert np.isclose(em.iae(t, err, horizon=4.0, hold_last=False), 12.0)
    assert np.isclose(em.iae(t, err, horizon=4.0, hold_last=True), 16.0)   # + 2 mm * 2 s
    # Longer runs are cut exactly at the horizon (value there interpolated: 6 mm at t = 4 s)
    t, err = np.array([0.0, 2.0, 6.0]), np.array([10.0, 2.0, 10.0])
    for hold_last in (False, True):
        assert np.isclose(em.iae(t, err, horizon=4.0, hold_last=hold_last), 12.0 + 8.0)


def test_series_metrics_from_a_decay():
    t = np.arange(0, 5.01, 0.05) + 100.0   # wall time does not start at 0
    series = pd.DataFrame({'t': t, 'gt_et': 0.02 * np.exp(-(t - 100.0))})
    m = em.series_metrics(series, converged=True)
    assert np.isclose(m['t_10mm'], np.log(2), atol=0.05)
    assert np.isclose(m['t_2mm'], np.log(10), atol=0.05)
    assert m['min_err_mm'] < 0.2
    assert em.series_metrics(series.iloc[:1], converged=True) == {}


def test_collect_skips_runs_without_servoing(tmp_path):
    block = tmp_path / 'main'
    (block / 'runs').mkdir(parents=True)
    rows = [{'target': 'red_can', 'start': s, 'outcome': o, 'success': int(o == 'converged'),
             'converged': int(o == 'converged'), 'final_pos_err_m': 0.004}
            for s, o in [(1, 'converged'), (2, 'not detected'), (3, 'lost test')]]
    pd.DataFrame(rows).to_csv(block / 'summary.csv', index=False)
    series = pd.DataFrame({'t': [0.0, 1.0, 2.0], 'gt_et': [0.012, 0.006, 0.004]})
    for s in (1, 3):
        series.to_csv(block / 'runs' / f'red_can_S{s}.csv', index=False)
    df = em.collect(tmp_path)
    assert list(df['start']) == [1]
    summary = em.summarise(df)
    assert summary.loc[0, 'reached_5mm'] == 1.0 and summary.loc[0, 'reached_2mm'] == 0.0
    assert np.isclose(summary.loc[0, 't_5mm_median'], 2.0)
