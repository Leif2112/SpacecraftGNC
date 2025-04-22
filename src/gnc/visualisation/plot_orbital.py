import numpy as np
import plotly.graph_objects as go

def plot_orbit_eci(X, Re, mu):
    """
    3D ECI orbit around a real‐occluding Earth using Plotly.

    Parameters
    ----------
    X  : ndarray, shape (6, N)
         State history [x,y,z,vx,vy,vz] in km and km/s.
    Re : float
         Earth radius (km).
    mu : float
         (ignored here) gravitational parameter.
    """
    # --- build high‑res sphere mesh ---
    phi   = np.linspace(0, np.pi,   100)
    theta = np.linspace(0, 2*np.pi, 200)
    Φ, Θ  = np.meshgrid(phi, theta)
    xs = Re * np.sin(Φ) * np.cos(Θ)
    ys = Re * np.sin(Φ) * np.sin(Θ)
    zs = Re * np.cos(Φ)

    earth = go.Mesh3d(
        x=xs.flatten(), y=ys.flatten(), z=zs.flatten(),
        alphahull=0,
        opacity=1.0,
        color='white',
        flatshading=True,
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9),
        lightposition=dict(x=100, y=200, z=0),
    )

    # --- plot orbit line slightly offset so it never clips the surface ---
    offset = 1.001
    r = X[:3, :] * offset
    orbit = go.Scatter3d(
        x=r[0], y=r[1], z=r[2],
        mode='lines',
        line=dict(color='#FF6188', width=4),
        name='Orbit',
    )

    # --- start/end markers ---
    markers = go.Scatter3d(
        x=[r[0,0], r[0,-1]],
        y=[r[1,0], r[1,-1]],
        z=[r[2,0], r[2,-1]],
        mode='markers',
        marker=dict(size=6, color='#A9DC76'),
        name='Start/End'
    )

    # --- ECI axes ---
    L = Re * 1.6
    axes = []
    for vec, label in [((L,0,0),'I'), ((0,L,0),'J'), ((0,0,L),'K')]:
        axes.append(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+text',
            line=dict(color='black', width=3),
            text=['', label],
            textposition='top center',
            textfont=dict(size=12),
            showlegend=False
        ))

    # --- compose and show ---
    fig = go.Figure(data=[earth, orbit, markers, *axes])
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='Iₑ (km)', showbackground=False),
            yaxis=dict(title='Jₑ (km)', showbackground=False),
            zaxis=dict(title='Kₑ (km)', showbackground=False),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Spacecraft ECI Orbital Trajectory"
    )
    fig.show()