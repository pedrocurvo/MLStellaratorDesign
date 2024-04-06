import torch 
import pandas as pd 
import numpy as np
from qsc import Qsc
from utils import predictor

if __name__ == "__main__":
    stel = predictor(axis_length=17,
                       iota=1.9,
                       max_elongation=7.3,
                       min_L_grad_B=0.20,
                       min_R0=0.26,
                       r_singularity=0.05,
                       L_grad_grad_B=0.11,
                       B20_variation=7.20,
                       beta=0.03,
                       DMerc_times_r2=9.5,
                       iterations=50,
                       device=None)
    
    stel.plot_boundary(r=0.01)
    

