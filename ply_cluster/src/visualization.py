import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_point_cloud(points, colors=None):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker=dict(size=1, color=colors)), row=1, col=1)
    fig.update_layout(width=1000, height=1000)
    fig.show()

def plot_points_3d(points, color='blue', size=2, opacity=0.6):
    """Create a basic 3D scatter plot for points."""
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=opacity
        ),
        name=f'Points ({len(points)} pts)'
    )

def plot_plane(plane_model, points, grid_size=20):
    """Create a surface plot for a plane within the points bounds."""
    a, b, c, d = plane_model
    
    # Get bounds from points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Create grid
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values for the plane
    Z = (-a * X - b * Y - d) / c
    
    return go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.3,
        showscale=False,
        name='Floor plane'
    )

def visualize_floor_detection(points, floor_points, non_floor_points, plane_model):
    """Visualize the floor detection step."""
    fig = go.Figure()
    
    # Add original points with low opacity
    fig.add_trace(plot_points_3d(points, color='gray', opacity=0.2))
    
    # Add floor points
    fig.add_trace(plot_points_3d(floor_points, color='green', opacity=0.8))
    
    # Add non-floor points
    fig.add_trace(plot_points_3d(non_floor_points, color='red', opacity=0.8))
    
    # Add floor plane
    fig.add_trace(plot_plane(plane_model, points))
    
    fig.update_layout(
        title='Floor Detection Results',
        scene=dict(
            aspectmode='data'
        ),
        showlegend=True
    )
    
    return fig

def visualize_orientation(points, plane_model, orientation_info):
    """Visualize the model orientation relative to the floor."""
    # Split points based on their position relative to the floor
    a, b, c, d = plane_model
    signed_distances = (points @ np.array([a, b, c]) + d)
    
    points_above = points[signed_distances > 0]
    points_below = points[signed_distances < 0]
    
    fig = go.Figure()
    
    # Add points above floor
    if len(points_above) > 0:
        fig.add_trace(plot_points_3d(points_above, color='blue', opacity=0.8))
    
    # Add points below floor
    if len(points_below) > 0:
        fig.add_trace(plot_points_3d(points_below, color='red', opacity=0.8))
    
    # Add floor plane
    fig.add_trace(plot_plane(plane_model, points))
    
    # Add floor normal vector at center of points
    center = points.mean(axis=0)
    normal = orientation_info['floor_normal'] * (points.max() - points.min()).mean() * 0.2
    
    fig.add_trace(go.Scatter3d(
        x=[center[0], center[0] + normal[0]],
        y=[center[1], center[1] + normal[1]],
        z=[center[2], center[2] + normal[2]],
        mode='lines+markers',
        line=dict(color='black', width=5),
        name='Floor normal'
    ))
    
    fig.update_layout(
        title=f"Model Orientation (Upside down: {orientation_info['is_upside_down']})",
        scene=dict(aspectmode='data'),
        showlegend=True
    )
    
    return fig

def visualize_alignment(original_points, aligned_points):
    """Visualize the alignment transformation."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Original Points', 'Aligned Points')
    )
    
    # Original points
    fig.add_trace(
        plot_points_3d(original_points, color='blue'),
        row=1, col=1
    )
    
    # Aligned points
    fig.add_trace(
        plot_points_3d(aligned_points, color='green'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Point Cloud Alignment Results',
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        showlegend=True
    )
    
    return fig

def visualize_final_result(original_points, final_points):
    """Visualize the original vs final processed point cloud."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Original Point Cloud', 'Processed Point Cloud')
    )
    
    # Original points
    fig.add_trace(
        plot_points_3d(original_points, color='blue'),
        row=1, col=1
    )
    
    # Final points
    fig.add_trace(
        plot_points_3d(final_points, color='green'),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Final Processing Results',
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        showlegend=True
    )
    
    return fig

def visualize_complete_pipeline(result_dict):
    """Visualize all steps of the pipeline in a single figure."""
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            'Floor Detection',
            'Orientation Analysis',
            'Alignment Result',
            'Final Result',
            'Denoised Result Statistical Outlier',
            'Denoised Result DBSCAN'
        )
    )
    
    # 1. Floor Detection
    floor_trace = plot_points_3d(result_dict['floor_points'], color='green')
    fig.add_trace(floor_trace, row=1, col=1)
    
    # 2. Orientation
    above_below_trace = plot_points_3d(result_dict['aligned_points'], color='blue')
    fig.add_trace(above_below_trace, row=1, col=2)
    
    # 3. Alignment
    aligned_trace = plot_points_3d(result_dict['aligned_points'], color='orange')
    fig.add_trace(aligned_trace, row=2, col=1)
    
    # 4. Final Result
    final_trace = plot_points_3d(result_dict['final_points'], color='red')
    fig.add_trace(final_trace, row=2, col=2)

    # 5. Denoised Result Statistical Outlier
    denoised_trace = plot_points_3d(result_dict['denoised_points'], color='purple')
    fig.add_trace(denoised_trace, row=3, col=1)

    # 6. Denoised Result DBSCAN
    # INSERT CODE HERE
    dbscan_labels = result_dict["dbscan_labels"]
    unique_labels = np.unique(dbscan_labels)
    
    for label in unique_labels:
        color = 'gray' if label == -1 else f"rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.6)"
        label_points = result_dict['final_points'][dbscan_labels == label]
        fig.add_trace(
            plot_points_3d(label_points, color=color),
            row=3, col=2
        )
        
    fig.update_layout(
        title='Complete Pipeline Visualization',
        height=1000,
        showlegend=True
    )
    
    return fig