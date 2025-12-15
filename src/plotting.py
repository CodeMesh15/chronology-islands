
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .island_solver import IslandSolver

# Set style for professional academic plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def plot_page_curve(solver, output_path="output/page_curve.pdf"):
    """
    Generates the 'Page Curve' showing the entropy minimum.
    Corresponds to Figure 3 in the paper.
    """
    # Create a grid of rho values from near-boundary to near-horizon
    rho_values = np.linspace(0.05 * solver.T, 0.99 * solver.T, 300)
    
    # Calculate entropy terms separately for visualization
    s_total = []
    s_grav = []
    s_cft = []
    
    for rho in rho_values:
        grav = solver.phi_r / (4 * solver.G * rho)
        arg = (solver.T**2 - rho**2) / (solver.epsilon**2)
        cft = (solver.c / 6.0) * np.log(arg)
        
        s_grav.append(grav)
        s_cft.append(cft)
        s_total.append(grav + cft)
        
    # Find the actual island location
    rho_star = solver.find_island()
    s_min = solver.generalized_entropy(rho_star)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(rho_values/solver.T, s_grav, '--', color='blue', alpha=0.6, label=r'Gravitational ($1/\rho$)')
    ax.plot(rho_values/solver.T, s_cft, '--', color='red', alpha=0.6, label=r'CFT ($\log(T^2-\rho^2)$)')
    ax.plot(rho_values/solver.T, s_total, '-', color='green', linewidth=2.5, label=r'Total $S_{gen}$')
    
    # Mark the minimum
    ax.scatter([rho_star/solver.T], [s_min], color='purple', zorder=5)
    ax.vlines(rho_star/solver.T, min(s_total)*0.9, s_min, linestyles=':', colors='purple')
    ax.text(rho_star/solver.T, s_min*1.05, f' QES\n $\\rho_* \\approx {rho_star/solver.T:.3f}T$', 
            color='purple', fontweight='bold')

    ax.set_xlabel(r'Island Location $\rho / T$')
    ax.set_ylabel(r'Entropy $S$')
    ax.set_title(r'Emergence of Quantum Extremal Island inside Chronology Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(s_total)*0.95, top=max(s_total)*1.1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_parameter_scan(c_fixed, T_fixed, phi_range, output_path="output/param_scan.pdf"):
    """
    Scans boundary coupling (phi_r) and plots how the island moves.
    """
    rho_stars = []
    
    for phi in phi_range:
        solver = IslandSolver(phi_r=phi, c=c_fixed, T=T_fixed)
        rho = solver.find_island()
        rho_stars.append(rho / T_fixed) # Normalize by T
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(phi_range, rho_stars, '-', color='darkblue', linewidth=2)
    
    ax.set_xlabel(r'Boundary Coupling $\phi_r / G$')
    ax.set_ylabel(r'Normalized Island Location $\rho_* / T$')
    ax.set_title(r'Stability of Island Location across Parameter Space')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Scan plot saved to {output_path}")
    plt.close()
