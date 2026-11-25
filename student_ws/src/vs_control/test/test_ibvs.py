import numpy as np

from vs_control import ibvs

K = (600.0, 600.0, 320.0, 240.0)


def features(u, v, area, angle):
    return ibvs.normalised_features(u, v, area, angle, K, desired_area=9000.0, desired_depth=0.3)


def test_prediction_without_dead_time_or_motion_is_identity():
    s = features(400, 200, 7000, 0.3)
    assert np.allclose(ibvs.predict_features(s, np.ones(6), 0.0), s)
    assert np.allclose(ibvs.predict_features(s, np.zeros(6), 0.1), s)


def test_prediction_follows_the_commanded_decay():
    # With v = -gain L^+ e the error decays as de/dt = -gain e, so dt ahead it is (1 - gain dt) e
    s, s_star = features(400, 200, 7000, 0.3), features(320, 240, 9000, 0.0)
    for control_yaw in (False, True):
        twist, e = ibvs.control_law(s.copy(), s_star, gain=1.5, control_yaw=control_yaw)
        e_pred = ibvs.predict_features(s, twist, 0.1) - s_star
        n = 4 if control_yaw else 3
        assert np.allclose(e_pred[:n], (1 - 1.5 * 0.1) * e[:n])


def test_prediction_ignores_wx_wy():
    # The control law does not servo wx, wy from the image; the optical-axis alignment adds them
    s = features(400, 200, 7000, 0.3)
    assert np.allclose(ibvs.predict_features(s, [0, 0, 0, 0.5, -0.5, 0], 0.1), s)


def test_control_law_wraps_the_angle():
    s_star = features(320, 240, 9000, 0.0)
    s = features(320, 240, 9000, np.pi - 0.1)   # principal axis is only defined modulo pi
    twist, e = ibvs.control_law(s.copy(), s_star, gain=1.0, control_yaw=True)
    assert np.isclose(e[3], -0.1)
    assert np.isclose(twist[5], -0.1)           # turns the short way, not by almost pi
