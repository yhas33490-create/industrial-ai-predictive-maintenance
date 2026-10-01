import pytest
import numpy as np


def test_power_calculation():
    """Verify raw mechanical power calculation (RPM * Torque)."""
    rpm = 1500
    torque = 40
    power = rpm * torque
    assert power == 60000


def test_temp_difference():
    """Verify temperature differential calculation."""
    air_temp = 300.0
    proc_temp = 310.0
    temp_diff = proc_temp - air_temp
    assert temp_diff == 10.0


def test_power_kw_conversion():
    """Verify power conversion formula to kilowatt (kW)."""
    rpm = 1500
    torque = 40
    power_kw = ((rpm * 2 * np.pi / 60) * torque) / 1000.0
    assert round(power_kw, 2) == 6.28


def test_osf_metric_calculation():
    """Verify Overstrain Failure (OSF) metric computation."""
    tool_wear = 200
    torque = 60
    osf_metric = tool_wear * torque
    assert osf_metric == 12000
