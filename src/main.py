# main.py
import os
import numpy as np
from src.island_solver import IslandSolver
from src.plotting import plot_page_curve, plot_parameter_scan

def main():
    # 0. Setup
    if not os.path.exists('output'):
        os.makedirs('output')
    
    print("--- Chronology Protection Solver ---")

    # 1. Reproduce the Specific Case from Paper (Section 3.7.2)
    # Parameters: c=12, phi_r=1000, T=10
    solver = IslandSolver(phi_r=1000, c=12, T=10)
    
    rho_star = solver.find_island()
    valid, ratio = solver.check_semiclassical_regime()
    
    print(f"\n[Paper Configuration]")
    print(f"Parameters: c=12, phi_r=1000, T=10")
    print(f"Island Location (rho_*): {rho_star:.4f} G")
    print(f"Ratio (rho_*/T):        {rho_star/10:.4f}")
    print(f"Semiclassical Check:     {valid} (Ratio = {ratio:.4f})")
    
    # 2. Generate Figure 3 (The Page Curve)
    print("\nGenerating Page Curve plot...")
    plot_page_curve(solver, "output/figure_3_page_curve.pdf")
    
    # 3. Generate Parameter Scan (Robustness Check)
    print("Generating Parameter Scan...")
    phi_values = np.logspace(1, 4, 100) # Scan phi_r from 10 to 10,000
    plot_parameter_scan(c_fixed=12, T_fixed=10, phi_range=phi_values, output_path="output/figure_4_scan.pdf")
    
    print("\nDone! Check the 'output/' folder.")

if __name__ == "__main__":
    main()
