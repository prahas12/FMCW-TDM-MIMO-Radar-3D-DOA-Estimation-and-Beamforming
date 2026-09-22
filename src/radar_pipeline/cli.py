import argparse
from radar_pipeline.config.loader import load_config
from radar_pipeline.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="FMCW TDM-MIMO Radar: 3D DOA Estimation and Beamforming")
    parser.add_argument('command', choices=['run', 'experiment'])
    parser.add_argument('--config', default='configs/baseline.yaml')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.command == 'run':
        from radar_pipeline.visualization.point_cloud import plot_3d_targets
        
        print("Running Baseline Pipeline...")
        targets, sim = run_pipeline(config)
        print(f"Detected {len(targets)} targets.")
        for i, t in enumerate(targets[:10]): # Print top 10 to avoid console spam
            print(f"Target {i+1}: R={t['range']:.2f}m, V={t['velocity']:.2f}m/s, "
                  f"Az={t['azimuth']:.1f}°, El={t['elevation']:.1f}°, SNR={t['snr']:.1f}")
        
        if len(targets) > 10:
            print(f"... and {len(targets) - 10} more detections.")
            
        # Generate 3D point cloud
        plot_3d_targets(targets, title="3D Target Detections", save_path="results/baseline/3d_targets.png")
        print("Saved 3D point cloud visualization to results/baseline/3d_targets.png")
                  
    elif args.command == 'experiment':
        print("Running Full Experiment Suite...")
        from radar_pipeline.evaluation.experiments import run_all
        run_all(args.config)
        print("All experiments completed and logged to results/manifest.json")

if __name__ == '__main__':
    main()
