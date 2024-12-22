# Partially generated using Claude 3.5 Sonnet

import numpy as np
import matplotlib.pyplot as plt

def plot_lcn_complexity(v=10, s=5):
    # Create meshgrid for F and d
    F = np.linspace(0.1, 1.0, 100)
    d = np.linspace(20, 124, 100)
    F_mesh, d_mesh = np.meshgrid(F, d)
    
    # Calculate m for each point
    m = v**(s-1)
    
    # Calculate P_LCN according to equation 5
    P_LCN = (F_mesh **  ((np.log(m) / np.log(s)) - 0.5)) * (d_mesh ** ((np.log(m) / np.log(s)) + 0.5))
    log_P_LCN = np.log(P_LCN)
    
    # Create the figure with specified size
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    im = plt.imshow(
        log_P_LCN,
        extent=[0.1, 1.0, 20, 124],
        origin='lower',
        aspect='auto',
        cmap='jet'
    )

    min_level = np.floor(log_P_LCN.min())
    max_level = np.ceil(log_P_LCN.max())
    levels = np.arange(min_level, max_level + 1)

    contours = plt.contour(
        F_mesh,
        d_mesh,
        log_P_LCN,
        levels=levels,
        colors='white',  # White contour lines
        alpha=0.5,       # Semi-transparent
        linewidths=0.5   # Thin lines
    )
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('$P_{LCN}$ (log scale)', rotation=270, labelpad=15)
    
    # Add labels and title
    plt.xlabel('Relevant Fraction F')
    plt.ylabel('Dimension d')
    plt.title(f'Sample Complexity of LCN (m = v^(s-1), v={v}, s={s})')
    
    # Add the m = v^(s-1) text at the top
    plt.text(0.5, 1.05, f'm = {v}^{s-1}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()

# Create and display the plot
plot_lcn_complexity()
plt.show()