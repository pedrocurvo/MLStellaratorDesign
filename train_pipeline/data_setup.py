"""
Contains functionality for creating PyTorch DataLOaders for
the Stellators dataset.
"""
import os
import sys
from torch.utils.data import DataLoader

# Get the directory of the script or notebook
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from StellatorsDataSet import StellatorsDataSet
