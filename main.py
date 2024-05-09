from mdn_torch import qsc_predictor

if __name__ == "__main__":
    stel = qsc_predictor(axis_length=17,
                       iota=1.9,
                       max_elongation=7.3,
                       min_L_grad_B=0.20,
                       min_R0=0.26,
                       r_singularity=0.05,
                       L_grad_grad_B=0.11,
                       B20_variation=7.20,
                       beta=0.03,
                       DMerc_times_r2=9.5,
                       iterations=10,
                       device=None, 
                       prioritize_DMerc_times_r2_positive=False,
                       print_values=True)
    

