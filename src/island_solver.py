
import numpy as np
from scipy.optimize import brentq

class IslandSolver:
    """
    Solver for Chronology Protection in JT Gravity.
    
    Attributes:
        phi_r (float): Boundary coupling constant (phi_r/G).
        c (float): Central charge of the CFT.
        T (float): Time identification scale (T/G).
        G (float): Newton's constant (default=1).
        epsilon (float): UV cutoff for entropy calculation.
    """
    
    def __init__(self, phi_r, c, T, G=1.0, epsilon=0.01):
        self.phi_r = phi_r
        self.c = c
        self.T = T
        self.G = G
        self.epsilon = epsilon
        
        # Pre-compute the constant factor from Eq. 41 to save time
        # k = (3 * phi_r) / (4 * G * c)
        self.k_factor = (3 * phi_r) / (4 * G * c)

    def generalized_entropy(self, rho):
        """
        Computes S_gen(rho) based on Eq. 40 in the paper.
        S_gen = Area/4G + S_matter
        """
        # Avoid division by zero at boundary
        if rho <= 0 or rho >= self.T:
            return np.inf

        # Area term: Phi / 4G = (phi_r / rho) / 4G
        s_grav = self.phi_r / (4 * self.G * rho)
        
        # Matter term: (c/6) * log((T^2 - rho^2)/eps^2)
        arg = (self.T**2 - rho**2) / (self.epsilon**2)
        s_matter = (self.c / 6.0) * np.log(arg)
        
        return s_grav + s_matter

    def _island_equation(self, rho):
        """
        The cubic equation derived from dS_gen/drho = 0 (Eq. 41).
        f(rho) = rho^3 - k * (T^2 - rho^2) = 0
        """
        lhs = rho**3
        rhs = self.k_factor * (self.T**2 - rho**2)
        return lhs - rhs

    def find_island(self):
        """
        Finds the quantum extremal surface location rho_star.
        Uses Brent's method to find the root between 0 and T.
        """
        try:
            # We search for the root in the interval [1e-6, T - 1e-6]
            # The function changes sign because:
            # at rho -> 0, LHS=0, RHS > 0 => f(rho) is Negative
            # at rho -> T, LHS=T^3, RHS = 0 => f(rho) is Positive
            rho_star = brentq(self._island_equation, 1e-5, self.T - 1e-5)
            return rho_star
        except Exception as e:
            print(f"Error finding island: {e}")
            return None

    def check_semiclassical_regime(self):
        """
        Verifies if the approximation G*c / phi_r << 1 holds.
        """
        ratio = (self.G * self.c) / self.phi_r
        is_valid = ratio < 0.1  # Arbitrary threshold for "much less than 1"
        return is_valid, ratio
