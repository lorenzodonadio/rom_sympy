from numba import jit
import sympy as sm
import numpy as np


@jit(nopython=True)
def thrust_force(R, v, a=None, C_T=None, rho=1.225):
    """
    Compute the thrust force on a horizontal-axis wind turbine.

    Parameters:
    ----------
    R : float
        Rotor radius [m].
    v : float
        Free-stream wind speed [m/s].
    a : float, optional
        Axial induction factor. If given, C_T is computed as 4a(1 - a).
    C_T : float, optional
        Thrust coefficient. If given, overrides 'a'.
    rho : float, optional
        Air density [kg/m^3].

    Returns:
    -------
    T : float
        Thrust force [N].
    """

    A = np.pi * R**2

    if C_T is None:
        if a is None:
            # Default to Betz optimal value
            a = 1 / 3
        C_T = 4 * a * (1 - a)

    return 0.5 * rho * A * v**2 * C_T


@jit(nopython=True)
def thrust_force_jit(R, v, a=1 / 3, rho=1.225):
    """
    Compute the thrust force on a horizontal-axis wind turbine.

    Parameters:
    ----------
    R : float
        Rotor radius [m].
    v : float
        Free-stream wind speed [m/s].
    a : float, optional
        Axial induction factor. If given, C_T is computed as 4a(1 - a).
    rho : float, optional
        Air density [kg/m^3].

    Returns:
    -------
    T : float
        Thrust force [N].
    """

    A = np.pi * R**2
    C_T = 4 * a * (1 - a)
    return 0.5 * rho * A * v**2 * C_T


def rotor_torque(R, v, omega, C_P=None, a=1 / 3, lambda_tsr=None, rho=1.225):
    """
    Compute the torque exerted on a horizontal-axis wind turbine rotor.

    Parameters:
    ----------
    R : float
        Rotor radius [m].
    v : float
        Free-stream wind speed [m/s].
    omega : float
        Rotor angular velocity [rad/s].
    C_P : float, optional
        Power coefficient. If not provided, approximated from a.
    a : float, optional
        Axial induction factor, used to approximate C_P if needed.
    lambda_tsr : float, optional
        Tip-speed ratio. If not provided, computed as omega * R / v.
    rho : float, optional
        Air density [kg/m^3].

    Returns:
    -------
    Q : float
        Rotor torque [Nm].
    """

    A = np.pi * R**2

    if lambda_tsr is None:
        lambda_tsr = omega * R / v + 1e-3

    if C_P is None:
        if a is None:
            a = 1 / 3  # Betz optimal
        C_P = 4 * a * (1 - a) ** 2  # Approximate

    return 0.5 * rho * A * v**3 * (C_P / lambda_tsr)


def rotor_power(R, v, C_P=None, a=1 / 3, rho=1.225):
    A = np.pi * R**2

    if C_P is None:
        if a is None:
            a = 1 / 3  # Betz optimal
        C_P = 4 * a * (1 - a) ** 2  # Approximate

    return 0.5 * rho * A * v**3 * C_P


def rotor_torque_jit(R, v, omega, a=1 / 3, rho=1.225):

    A = np.pi * R**2
    lambda_tsr = omega * R / v + 1e-3
    C_P = 4 * a * (1 - a) ** 2  # Approximate
    return 0.5 * rho * A * v**2 * (C_P / lambda_tsr)


def generator_torque_control(Q, omega, omega_max, k_p=1.0):
    """
    Simple generator torque control to limit rotor speed.

    Parameters:
    ----------
    Q : float
        Rotor aerodynamic torque [Nm].
    omega : float
        Rotor speed [rad/s].
    omega_max : float
        Maximum allowed rotor speed [rad/s].
    k_p : float
        Proportional gain for torque control. Default: 1.0.

    Returns:
    -------
    Q_g : float
        Generator torque [Nm].
    """

    if omega < omega_max:
        # Soft control: generator torque ramps up
        Q_g = k_p * (omega / omega_max) * Q
        # Never exceed rotor torque
        Q_g = min(Q_g, Q)
    else:
        # Full braking torque to avoid overspeed
        Q_g = 1.00001 * Q

    return Q_g
