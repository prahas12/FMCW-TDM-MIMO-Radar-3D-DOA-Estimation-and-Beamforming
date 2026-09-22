
import numpy as np

def spherical_to_cartesian(r, az_deg, el_deg):
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    
    # Radar mapping: X=Azimuth(right), Y=Elevation(up), Z=Broadside(depth)
    x = r * np.sin(az_rad) * np.cos(el_rad)
    y = r * np.sin(el_rad)
    z = r * np.cos(az_rad) * np.cos(el_rad)
    
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(y / r)
    az = np.arcsin(x / (r * np.cos(el)))
    
    return r, np.rad2deg(az), np.rad2deg(el)
