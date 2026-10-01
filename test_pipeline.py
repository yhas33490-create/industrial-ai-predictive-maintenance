import pytest
import numpy as np

def test_power_calculation():
    rpm = 1500
    torque = 40
    power_kw = ((rpm * 2 * np.pi / 60) * torque) / 1000.0
    assert round(power_kw, 2) == 6.28

def test_temp_difference():
    air_temp = 300
    proc_temp = 310
    assert (proc_temp - air_temp) == 10
