import plotly.graph_objects as go
from scipy.spatial import cKDTree
import numpy as np
from plyfile import PlyData

def read_ply(file_path):
    plydata = PlyData.read(file_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    points = np.array([x, y, z]).T
    return points

def distance_based_metrics(original, ground_truth, generated, distance_threshold=0.1):
    gt_tree = cKDTree(ground_truth)
    gen_tree = cKDTree(generated)

    gt_to_gen_distances, _ = gt_tree.query(generated)
    gen_to_gt_distances, _ = gen_tree.query(ground_truth)

    chamfer_distance = (gt_to_gen_distances**2).mean() + (gen_to_gt_distances**2).mean()

    fp_dist_penalty = (gt_to_gen_distances > distance_threshold).sum()
    fn_dist_penalty = (gen_to_gt_distances > distance_threshold).sum()

    tp = ((gt_to_gen_distances <= distance_threshold).sum() + 
          (gen_to_gt_distances <= distance_threshold).sum()) // 2

    precision = tp / (tp + fp_dist_penalty) if (tp + fp_dist_penalty) > 0 else 0
    recall = tp / (tp + fn_dist_penalty) if (tp + fn_dist_penalty) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        # "Chamfer Distance": chamfer_distance,
        "Precision (Distance-Weighted)": precision,
        "Recall (Distance-Weighted)": recall,
        "F1-Score (Distance-Weighted)": f1_score,
        # "False Positives (Far)": fp_dist_penalty,
        # "False Negatives (Far)": fn_dist_penalty,
        # "GT Points": len(ground_truth),
        # "Generated Points": len(generated),
    }

# Example usage
original = np.random.rand(1000, 3)
ground_truth = read_ply("data/kitchen_gt.ply")
generated = read_ply("output/kitchen/cluster/kitchen.ply")

metrics = distance_based_metrics(original, ground_truth, generated, distance_threshold=0.05)

# Create a Plotly bar chart
fig = go.Figure()

# Add metrics to the bar chart
fig.add_trace(go.Bar(
    x=list(metrics.keys()),
    y=list(metrics.values()),
    text=[f"{v:.4f}" if isinstance(v, float) else v for v in metrics.values()],
    textposition='auto',
    marker=dict(color=['#636EFA', '#EF553B', '#00CC96'])
))

# Update layout for better presentation
fig.update_layout(
    title="Distance-Based Metrics for Point Cloud Comparison",
    xaxis_title="Metric",
    yaxis_title="Value",
    yaxis=dict(automargin=True),
    template="plotly_white",
    xaxis=dict(tickangle=-45)
)

# Show the figure
fig.show()
