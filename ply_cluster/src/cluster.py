import numpy as np

import plotly.graph_objs as go

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def dbscan(points, eps=0.5, min_samples=200):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

def get_peaks(density, min_peak_points, plot=False):
    density = np.sort(density)
    density = density[int(0.1 * len(density)):]
    density_values, bin_edges = np.histogram(density, bins=100) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_density = gaussian_filter1d(density_values, sigma=2)

    peaks, _ = find_peaks(smoothed_density, height=min_peak_points)

    peak_boundaries = []
    for peak in peaks:
        start, end = peak, peak

        while start > 0 and smoothed_density[start - 1] < smoothed_density[start]:
            start -= 1
        
        while end < len(smoothed_density) - 1 and smoothed_density[end + 1] < smoothed_density[end]:
            end += 1

        peak_boundaries.append((start, end))

    if plot:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=density_values,
            mode='lines',
            name="Original Histogram"
        ))

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=smoothed_density,
            mode='lines',
            line=dict(color='orange'),
            name="Smoothed Histogram"
        ))

        fig.add_trace(go.Scatter(
            x=bin_centers[peaks],
            y=smoothed_density[peaks],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name="Detected Peaks"
        ))

        for idx, (start, end) in enumerate(peak_boundaries):
            fig.add_trace(go.Scatter(
                x=[bin_centers[start], bin_centers[start]],
                y=[0, max(density_values)],
                mode='lines',
                line=dict(color='green', dash='dash'),
                showlegend=idx == 0,
                name="Peak Start" if idx == 0 else None
            ))
            fig.add_trace(go.Scatter(
                x=[bin_centers[end], bin_centers[end]],
                y=[0, max(density_values)],
                mode='lines',
                line=dict(color='purple', dash='dash'),
                showlegend=idx == 0,
                name="Peak End" if idx == 0 else None
            ))

        fig.update_layout(
            title="Density Histogram with Peak Detection",
            xaxis_title="Density",
            yaxis_title="Number of Points",
            legend=dict(
                x=1.0,
                y=1.0,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='Black',
                borderwidth=1
            )
        )

        fig.show()
    return peak_boundaries, bin_centers


def monte_carlo_kde(points: np.ndarray, bandwidth: float, sample_size: int = 500) -> np.ndarray:
    sample_indices = np.random.choice(len(points), sample_size, replace=False)
    sample_points = points[sample_indices]
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(sample_points)
    
    log_density = kde.score_samples(points)
    density = np.exp(log_density)
    
    return density

def get_densest_cluster(points, min_peak_points, colors=None, plot=False):
    density = monte_carlo_kde(points, bandwidth=1, sample_size=max(len(points) // 100, 3000))  
    peak_boundaries, bin_centers = get_peaks(density, min_peak_points, plot=True)
    first_peak_end_index = peak_boundaries[0][1]
    first_peak_end = bin_centers[first_peak_end_index]
    print(f"First peak ends at density {first_peak_end}")
    points = points[density > first_peak_end]
    density = density[density > first_peak_end]
    fig = go.Figure()

    if plot:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                # color uses r, g, and b
                color=density,
                colorscale='Viridis',
                colorbar=dict(title='Density'),
            )
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title="3D Point Cloud with Density Color-Coding"
        )

        fig.show()
    return points#, colors
