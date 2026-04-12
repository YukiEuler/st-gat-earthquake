#!/usr/bin/env python
"""
ST-GAT: Spatio-Temporal Graph Attention Networks for Earthquake Magnitude Prediction

Main entry point for the project. 
Choose which training script to run based on your needs.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ST-GAT Earthquake Magnitude Prediction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "multiresolution", "event", "hybrid"],
        help="Training mode to use"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            from training.train import main as train_main
            print("Starting default training mode...")
            train_main()
        elif args.mode == "multiresolution":
            from training.train_multiresolution import main as train_mr_main
            print("Starting multi-resolution training mode...")
            train_mr_main()
        elif args.mode == "event":
            from training.train_event import main as train_event_main
            print("Starting event-based training mode...")
            train_event_main()
        elif args.mode == "hybrid":
            from training.train_hybrid import main as train_hybrid_main
            print("Starting hybrid training mode...")
            train_hybrid_main()
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Make sure all dependencies are installed and imports are correctly configured.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
