import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class IdealHAWT:
    """
    Class for calculating performance parameters of an ideal 3-bladed horizontal axis wind turbine
    with wake rotation and tip-loss correction.
    """

    def __init__(self, rotor_radius, air_density=1.225):
        """
        Initialize the wind turbine parameters.

        Parameters:
        -----------
        rotor_radius : float
            Rotor radius in meters
        air_density : float
            Air density in kg/m³ (default: 1.225 kg/m³ at sea level)
        """
        self.R = rotor_radius
        self.rho = air_density
        self.A = np.pi * rotor_radius**2  # Rotor swept area

    def tsr(self, wind_speed, angular_velocity):
        """
        Calculate the tip speed ratio.

        Parameters:
        -----------
        wind_speed : float
            Free stream wind speed in m/s
        angular_velocity : float
            Rotor angular velocity in rad/s

        Returns:
        --------
        float : Tip speed ratio
        """
        if abs(wind_speed) < 1e-2:
            return 0

        return abs((angular_velocity * self.R) / wind_speed)

    def local_speed_ratio(self, tsr, radial_pos):
        """
        Calculate the local speed ratio at a given radial position.

        Parameters:
        -----------
        tsr : float
        radial_pos : float Radial position (0 at hub, R at tip)

        Returns:
        --------
        float : Local speed ratio
        """
        return tsr * (radial_pos / self.R)

    def prandtl_tip_loss_factor(self, radial_pos, a, tsr):
        """
        Calculate Prandtl tip loss correction factor for 3 blades.

        Parameters:
        -----------
        radial_pos : float
            Radial position
        a : float
            Axial induction factor
        tsr : float
            Tip speed ratio

        Returns:
        --------
        float : Tip loss factor F
        """
        if radial_pos >= self.R:
            return 0.0

        # Local speed ratio at current radial position
        lambda_r = self.local_speed_ratio(tsr, radial_pos)

        # Avoid division by zero
        if lambda_r == 0 or radial_pos == 0:
            return 0.0

        # Calculate flow angle phi
        # phi = np.arctan(1 / ((1 - a) * lambda_r))
        # Calculate flow angle phi
        phi = np.arctan((1 - a) * (4 * a - 1) / (a * lambda_r))

        # Calculate the argument for the exponential - tip loss
        f_tip = (3 / 2) * ((self.R - radial_pos) / (radial_pos * np.sin(phi)))
        exp_tip = np.exp(-np.clip(f_tip, -100, 100))
        F_tip = (2 / np.pi) * np.arccos(np.clip(exp_tip, -1.0, 1.0))

        # Hub loss factor
        hub_radius = 0.1 * self.R  # Assume hub at 10% of radius
        if radial_pos <= hub_radius:
            # f_hub = (3 / 2) * ((radial_pos - hub_radius) / (radial_pos * np.sin(phi)))
            # exp_hub = np.exp(-np.clip(f_hub, -100, 100))
            # F_hub = (2 / np.pi) * np.arccos(np.clip(exp_hub, -1.0, 1.0))
            F_hub = 0.0
        else:
            f_hub = (3 / 2) * ((radial_pos - hub_radius) / (radial_pos * np.sin(phi)))
            exp_hub = np.exp(-np.clip(f_hub, -100, 100))
            F_hub = (2 / np.pi) * np.arccos(np.clip(exp_hub, -1.0, 1.0))

        return F_tip * F_hub

    def axial_induction_from_lambda_r(self, lambda_r, radial_pos, tsr):
        """
        Calculate axial induction factor for maximum power at given local speed ratio.

        Parameters:
        -----------
        lambda_r : float
            Local speed ratio
        radial_pos : float
            Radial position for tip loss calculation
        tsr : float
            Tip speed ratio for tip loss calculation

        Returns:
        --------
        float : Axial induction factor a
        """
        if lambda_r == 0:
            return 0.25

        # Define the equation to solve
        def equation(a):
            return (1 - a) * (4 * a - 1) ** 2 - lambda_r**2 * (1 - 3 * a)

        # Solve for a in the range [0.25, 0.333]
        a_initial = 0.3
        try:
            a_solution = fsolve(equation, a_initial)[0]
            a_solution = np.clip(a_solution, 0.25, 1 / 3)
        except:
            a_solution = 1 / 3

        # Apply tip loss factor (always applied)
        # F = self.prandtl_tip_loss_factor(radial_pos, a_solution, tsr)

        # For modern turbines operating at optimal TSR (7-10), axial induction stays around 0.33
        # No Glauert correction needed as a < 0.4 for optimal operation
        return np.clip(a_solution, 0.0, 1.0)

    def angular_induction_factor(self, a, radial_pos, tsr):
        """
        Calculate angular induction factor from axial induction factor.

        Parameters:
        -----------
        a : float
            Axial induction factor a
        radial_pos : float
            Radial position for tip loss calculation
        tsr : float
            Tip speed ratio for tip loss calculation

        Returns:
        --------
        float : Angular induction factor a'
        """
        if a <= 0.25 or abs(4 * a - 1) < 1e-10:
            return 0.0

        # a_prime = (1 - 3 * a) / (4 * a - 1)
        local_tsr = self.local_speed_ratio(tsr, radial_pos)
        a_prime = -0.5 + 0.5 * np.sqrt(1 + 4 * a * (1 - a) / (local_tsr**2))
        # # Apply tip loss correction to angular induction factor (always applied)
        # F = self.prandtl_tip_loss_factor(radial_pos, axial_induction, tsr)
        # if F > 1e-10:  # Avoid division by zero
        #     a_prime = a_prime / F

        return max(a_prime, 0.0)

    def power_coefficient(self, tsr, n_points=100):
        """
        Calculate the power coefficient by integrating over the rotor.

        Parameters:
        -----------
        tsr : float
            Tip speed ratio
        n_points : int
            Number of integration points

        Returns:
        --------
        float : Power coefficient C_P
        """
        if tsr <= 1:
            return 0.05 * tsr

        radial_pos = np.linspace(0.1 * self.R, self.R, n_points)
        d_lambda_r = tsr / n_points

        integral = 0.0

        for r in radial_pos:
            lambda_r = self.local_speed_ratio(tsr, r)

            if lambda_r <= 0:
                continue

            a = self.axial_induction_from_lambda_r(lambda_r, r, tsr)
            a_prime = self.angular_induction_factor(a, r, tsr)

            # Get tip loss factor (always applied)
            F = self.prandtl_tip_loss_factor(r, a, tsr)

            integral += F * a_prime * (1 - a) * lambda_r**3 * d_lambda_r

        cp = (8 / tsr**2) * integral
        return min(cp, 0.5926)

    def annular_torque(self, wind_speed, angular_velocity, radial_pos, dr):
        """
        Calculate torque on an annular element at radius r with thickness dr.

        Parameters:
        -----------
        wind_speed : float
            Free stream wind speed in m/s
        angular_velocity : float
            Rotor angular velocity in rad/s
        radial_pos : float
            Radial position of the annular element
        dr : float
            Thickness of the annular element

        Returns:
        --------
        float : Torque on annular element in N·m
        """
        lambda_tsr = self.tsr(wind_speed, angular_velocity)
        lambda_r = self.local_speed_ratio(lambda_tsr, radial_pos)

        a = self.axial_induction_from_lambda_r(lambda_r, radial_pos, lambda_tsr)
        a_prime = self.angular_induction_factor(a, radial_pos, lambda_tsr)

        # Calculate tip loss factor (always applied)
        F = self.prandtl_tip_loss_factor(radial_pos, a, lambda_tsr)

        # Calculate annular area and torque
        dA = 2 * np.pi * radial_pos * dr
        Cq = 4 * F * a_prime * (1 - a)
        dQ = 0.5 * Cq * self.rho * wind_speed * angular_velocity * radial_pos**2 * dA
        return dQ

    def total_torque(self, wind_speed, angular_velocity, n_annuli=100):
        """
        Calculate total torque on the rotor by integrating annular torques.

        Parameters:
        -----------
        wind_speed : float
            Free stream wind speed in m/s
        angular_velocity : float
            Rotor angular velocity in rad/s
        n_annuli : int
            Number of annular elements for integration

        Returns:
        --------
        float : Total torque in N·m
        """
        dr = self.R / n_annuli
        total_torque = 0.0
        tsr = self.tsr(wind_speed, angular_velocity)

        if tsr < 0:
            return total_torque
        elif tsr < 2:
            Cq = 0.04
            total_torque = 0.5 * Cq * self.rho * self.R * self.A * wind_speed**2

        for i in range(n_annuli):
            r = (i + 0.5) * dr
            if r < 0.1 * self.R:  # Skip very small radii near hub
                continue
            dQ = self.annular_torque(wind_speed, angular_velocity, r, dr)
            if np.isfinite(dQ):  # Only add finite values
                total_torque += dQ

        return total_torque

    def thrust_coefficient(self, wind_speed, angular_velocity, n_points=100):
        tsr = self.tsr(wind_speed, angular_velocity)
        den = 0.5 * self.rho * self.A * (self.R * tsr) ** 2
        return self.total_thrust(wind_speed, angular_velocity, n_points) / den

    def total_thrust(self, wind_speed, angular_velocity, n_points=100):
        """
        Calculate thrust coefficient C_T.

        Parameters:
        -----------
        tsr : float
            Tip speed ratio
        n_points : int
            Number of integration points

        Returns:
        --------
        float : Thrust coefficient C_T
        """

        tsr = self.tsr(wind_speed, angular_velocity)
        thrust = 0.0

        if tsr < 0:
            return thrust
        elif tsr < 2:
            Ct_parked = 0.18
            thrust = 0.5 * Ct_parked * self.rho * self.A * (wind_speed) ** 2
            # print(thrust)
        radial_poss = np.linspace(0.1 * self.R, 0.95 * self.R, n_points)

        dr = self.R / n_points

        for r in radial_poss:
            dA = 2 * np.pi * r * dr
            lambda_r = self.local_speed_ratio(tsr, r)

            if lambda_r <= 0:
                continue

            a = self.axial_induction_from_lambda_r(lambda_r, r, tsr)
            F = self.prandtl_tip_loss_factor(r, a, tsr)
            Ct = 4.0 * a * (1.0 - a)
            # Annular thrust - using standard momentum theory (no Glauert correction)
            # For optimal TSR operation, a < 0.4 so C_T = 4a(1-a) is valid
            dT = 0.5 * Ct * F * self.rho * (wind_speed) ** 2 * dA

            if np.isfinite(dT):
                thrust += dT

        return thrust

    def power_output(self, wind_speed, angular_velocity, mechanical_efficiency=0.95):
        """
        Calculate mechanical power output.

        Parameters:
        -----------
        wind_speed : float
            Free stream wind speed in m/s
        angular_velocity : float
            Rotor angular velocity in rad/s
        mechanical_efficiency : float
            Mechanical/electrical efficiency (default: 0.95)

        Returns:
        --------
        float : Power output in Watts
        """
        lambda_tsr = self.tsr(wind_speed, angular_velocity)
        cp = self.power_coefficient(lambda_tsr)
        if not np.isfinite(cp):
            cp = 0.0
        power_wind = 0.5 * self.rho * self.A * wind_speed**3
        return mechanical_efficiency * cp * power_wind


def calculate_torque(rotor_radius, wind_speed, angular_velocity):
    """
    Simple function to calculate torque for given conditions.

    Parameters:
    -----------
    rotor_radius : float
        Rotor radius in meters
    wind_speed : float
        Wind speed in m/s
    angular_velocity : float
        Angular velocity in rad/s

    Returns:
    --------
    float : Torque in N·m
    """
    turbine = IdealHAWT(rotor_radius)
    torque = turbine.total_torque(wind_speed, angular_velocity)
    return torque if np.isfinite(torque) else 0.0


def calculate_thrust(rotor_radius, wind_speed, angular_velocity):
    """
    Simple function to calculate thrust for given conditions.
    float : thrust in N
    """
    turbine = IdealHAWT(rotor_radius)
    thrust = turbine.total_thrust(wind_speed, angular_velocity)
    return thrust if np.isfinite(thrust) else 0.0


def calculate_power_coefficient(rotor_radius, tsr):
    """
    Simple function to calculate power coefficient.

    Parameters:
    -----------
    rotor_radius : float
        Rotor radius in meters
    tsr : float
        Tip speed ratio

    Returns:
    --------
    float : Power coefficient C_P
    """
    turbine = IdealHAWT(rotor_radius)
    cp = turbine.power_coefficient(tsr)
    return cp if np.isfinite(cp) else 0.0


def plot_tip_loss_distribution(tsr=7.5):
    """
    Plot tip loss factor distribution across the rotor
    """
    turbine = IdealHAWT(rotor_radius=1.0)

    radial_pos = np.linspace(0, 1.0, 100)
    tip_loss_factors = []

    for r in radial_pos:
        # Calculate tip loss factor for a typical axial induction
        F = turbine.prandtl_tip_loss_factor(r, 1 / 3, tsr)
        tip_loss_factors.append(F)

    plt.figure(figsize=(10, 6))
    plt.plot(radial_pos, tip_loss_factors, "b-", linewidth=2)
    plt.xlabel("Normalized Radial Position (r/R)")
    plt.ylabel("Tip Loss Factor, F")
    plt.title(f"Prandtl Tip Loss Factor Distribution (λ = {tsr})")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.show()


def plot_power_coefficient_vs_tsr():
    """
    Plot power coefficient vs tip speed ratio
    """
    turbine = IdealHAWT(rotor_radius=1.0)

    tsr_values = np.linspace(0.1, 10, 50)
    cp_values = [turbine.power_coefficient(tsr) for tsr in tsr_values]

    plt.figure(figsize=(10, 6))
    plt.plot(tsr_values, cp_values, "b-", linewidth=2, label="With Tip Loss Correction")
    plt.axhline(y=16 / 27, color="r", linestyle="--", linewidth=2, label="Betz Limit")
    plt.xlabel("Tip Speed Ratio, λ")
    plt.ylabel("Power Coefficient, C_P")
    plt.title("Power Coefficient vs Tip Speed Ratio for 3-Bladed Turbine")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 0.65)
    plt.show()


def main():
    """
    Demonstration of the 3-bladed wind turbine analysis.
    """
    # Initialize turbine
    turbine = IdealHAWT(rotor_radius=50.0)

    print("3-BLADED WIND TURBINE ANALYSIS")
    print("=" * 50)

    # Example calculations
    wind_speed = 12.0
    angular_velocity = 1.8

    # Calculate performance metrics
    tsr = turbine.tsr(wind_speed, angular_velocity)
    cp = turbine.power_coefficient(tsr)
    ct = turbine.thrust_coefficient(wind_speed, angular_velocity)
    torque = turbine.total_torque(wind_speed, angular_velocity)
    power = turbine.power_output(wind_speed, angular_velocity)

    print(f"Rotor radius: {turbine.R} m")
    print(f"Wind speed: {wind_speed} m/s")
    print(f"Angular velocity: {angular_velocity} rad/s")
    print(f"Tip speed ratio: {tsr:.3f}")
    print(f"Power coefficient C_P: {cp:.4f}")
    print(f"Thrust coefficient C_T: {ct:.4f}")
    print(f"Torque: {torque:,.0f} N·m")
    print(f"Power output: {power/1e6:.2f} MW")
    print(f"Available wind power: {0.5*turbine.rho*turbine.A*wind_speed**3/1e6:.2f} MW")

    print("\n" + "=" * 50)
    print("TORQUE CALCULATIONS FOR DIFFERENT CONDITIONS")
    print("=" * 50)

    # Torque for different conditions
    wind_speeds = [2, 5, 8, 10, 12, 15]
    angular_velocities = [0.1, 0.5, 1, 1.2, 1.5]

    print(
        f"{'Wind Speed':<12} {'Angular Velocity':<18} {'TSR':<8} {'Torque':<10} {'Thrust':<10} {'Ct':<8}"
    )
    print(f"{'(m/s)':<12} {'(rad/s)':<18} {'':<8} {'(kN·m)':<10} {'(kN·m)':<10} {'  -  ':<8}")
    print("-" * 65)

    for av in angular_velocities:
        for ws in wind_speeds:
            torque_val = calculate_torque(50.0, ws, av) / 1000
            thrust_val = calculate_thrust(50.0, ws, av) / 1000
            tsr_val = turbine.tsr(ws, av)
            ct = turbine.thrust_coefficient(ws, av)
            print(
                f"{ws:<12.1f} {av:<18.1f} {tsr_val:<8.2f} {torque_val:<12.1f} {thrust_val:<12.1f} {ct:<8.3f}"
            )

    # Create plots
    print("\nGenerating plots...")
    plot_tip_loss_distribution()
    plot_power_coefficient_vs_tsr()


if __name__ == "__main__":
    main()
