import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec


def lotka_volterra(t, z, alpha, beta, gamma, delta):
    hare, lynx = z
    return [
        alpha * hare - beta * hare * lynx,
        delta * beta * hare * lynx - gamma * lynx
    ]


def plot_population_competition(a=0.1, b=0.02, c=0.3, d=0.01, initial_conditions=None, t_end=50):
    """
    Visualize the Lotka-Volterra population competition model.
    
    Parameters:
    a: Natural growth rate of species 1
    b: Competition coefficient of species 1 affected by species 2
    c: Natural death rate of species 2
    d: Competition coefficient of species 2 affected by species 1
    initial_conditions: List of initial conditions [(N1_0, N2_0), ...]
    t_end: Simulation time
    output_path: Path to save the output file
    """
    
    # Define Lotka-Volterra equations
    def competition_model(t, state):
        N1, N2 = state
        dN1dt = a * N1 - b * N1 * N2
        dN2dt = -c * N2 + d * N1 * N2
        return [dN1dt, dN2dt]
    
    # Set default initial conditions
    if initial_conditions is None:
        initial_conditions = [(10, 5), (20, 15), (5, 15)]
        
    # Create figure
    fig = plt.figure(figsize=(15, 10), dpi=130)
    gs = gridspec.GridSpec(2, 2)
    
    # Population time evolution plot
    ax1 = plt.subplot(gs[0, 0])
    # Phase portrait
    ax2 = plt.subplot(gs[0, 1])
    # Nullcline analysis
    ax3 = plt.subplot(gs[1, :])
    
    # Colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))
    
    # Simulate for different initial conditions
    for i, (N10, N20) in enumerate(initial_conditions):
        # Solve differential equations
        sol = solve_ivp(competition_model, [0, t_end], [N10, N20], t_eval=np.linspace(0, t_end, 500))
        N1 = sol.y[0]
        N2 = sol.y[1]
        t = sol.t
        
        # Population time evolution
        ax1.plot(t, N1, color=colors[i], linestyle='-', linewidth=2, label=f"Species 1: N₁(0)={N10}")
        ax1.plot(t, N2, color=colors[i], linestyle='--', linewidth=2, label=f"Species 2: N₂(0)={N20}")
        
        # Phase portrait
        ax2.plot(N1, N2, color=colors[i], linewidth=1.5, label=f"Initial point: ({N10}, {N20})")
        ax2.scatter([N10], [N20], color=colors[i], s=50, zorder=5)
        
    # Configure population time evolution plot
    ax1.set_title("Population Evolution over Time", fontsize=14)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Population Size", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='best', fontsize=10)
    
    # Configure phase portrait
    ax2.set_title("Population Phase Portrait (N₁ vs N₂)", fontsize=14)
    ax2.set_xlabel("Species 1 Population (N₁)", fontsize=12)
    ax2.set_ylabel("Species 2 Population (N₂)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='best', fontsize=10)
    
    # Nullcline analysis
    N1_vals = np.linspace(0, 50, 100)
    # dN1/dt = 0: a*N1 - b*N1*N2 = 0 => N2 = a/b
    N2_null1 = a/b * np.ones_like(N1_vals)
    # dN2/dt = 0: -c*N2 + d*N1*N2 = 0 => N1 = c/d
    N1_null2 = c/d * np.ones_like(N1_vals)
    
    # Plot nullclines
    ax3.plot(N1_vals, N2_null1, 'r-', linewidth=2, label=r"$\frac{dN_1}{dt}=0$")
    ax3.plot([N1_null2[0]]*100, np.linspace(0, 50, 100), 'b-', linewidth=2, label=r"$\frac{dN_2}{dt}=0$")
    
    # Mark equilibrium point
    ax3.scatter([c/d], [a/b], s=100, c='purple', marker='*', label=f"Equilibrium: ({c/d:.1f}, {a/b:.1f})")
    
    # Add vector field
    N1, N2 = np.meshgrid(np.linspace(0.1, 50, 12), np.linspace(0.1, 50, 12))
    dN1 = a * N1 - b * N1 * N2
    dN2 = -c * N2 + d * N1 * N2
    
    # Normalize vectors
    magnitude = np.sqrt(dN1**2 + dN2**2)
    dN1 = dN1 / magnitude
    dN2 = dN2 / magnitude
    
    # Plot vector field
    # Increasing 'scale' makes the arrows smaller
    # Adjusting 'width' and 'headwidth' helps keep them legible at a smaller size
    ax3.quiver(N1, N2, dN1, dN2, scale=40, color='green', width=0.005, headwidth=3, pivot='mid', alpha=0.6)
    
    # Configure nullcline plot
    ax3.set_title("Nullcline Analysis and Equilibrium", fontsize=14)
    ax3.set_xlabel("Species 1 Population (N₁)", fontsize=12)
    ax3.set_ylabel("Species 2 Population (N₂)", fontsize=12)
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, 50)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='best', fontsize=10)
    
    # Add model equations
    model_text = (
        r"Lotka-Volterra Competition Model: "
        r"$\frac{dN_1}{dt} = \alpha N_1 - \beta N_1 N_2$; "
        r"$\frac{dN_2}{dt} = -\gamma N_2 + \delta N_1 N_2$"
        f"\nα={a}, β={b}, γ={c}, δ={d}"
    )
    fig.text(0.5, 0.02, model_text, ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(pad=3.0)

    plt.show()




