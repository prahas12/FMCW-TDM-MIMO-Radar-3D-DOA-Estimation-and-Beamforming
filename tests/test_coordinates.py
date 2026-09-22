
import numpy as np
from radar_pipeline.localization.coordinates import spherical_to_cartesian, cartesian_to_spherical

def test_spherical_to_cartesian():
    # Broadside (r, az, el) -> (10, 0, 0)
    x, y, z = spherical_to_cartesian(10, 0, 0)
    assert np.isclose(x, 0)
    assert np.isclose(y, 0)
    assert np.isclose(z, 10)

def test_cartesian_to_spherical():
    r, az, el = cartesian_to_spherical(0, 0, 10)
    assert np.isclose(r, 10)
    assert np.isclose(az, 0)
    assert np.isclose(el, 0)
