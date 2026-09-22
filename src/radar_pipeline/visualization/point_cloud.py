import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_targets(targets, true_targets=None, title="3D Target Localization", save_path="results/baseline/3d_targets.png"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if true_targets:
        xs = [t.x for t in true_targets]
        ys = [t.y for t in true_targets]
        zs = [t.z for t in true_targets]
        ax.scatter(xs, ys, zs, c='red', marker='x', s=100, label='Ground Truth')
        
    if targets:
        xs = [t['x'] for t in targets]
        ys = [t['y'] for t in targets]
        zs = [t['z'] for t in targets]
        ax.scatter(xs, ys, zs, c='blue', marker='o', s=50, label='Estimated')
        
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (Boresight) (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
