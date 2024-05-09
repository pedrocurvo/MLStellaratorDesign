import numpy as np

def check_criteria(output):
    """
    Check the criteria for a good stellarator.

    Criteria for a good stellarator:
    - axis_length > 0
    - |iota| >= 0.2
    - max_elongation <= 10
    - |min_L_grad_B| >= 0.1
    - |min_R0| >= 0.3
    - r_singularity >= 0.05
    - |L_grad_grad_B| >= 0.1
    - B20_variation <= 5
    - beta >= 1e-4
    - DMerc_times_r2 > 0

    Parameters
    ----------
    output : list
        List of output values from the Qscout model.
    
    Returns
    -------
    bool
        True if the stellarator satisfies the criteria, False otherwise.
    """

    [axis_length, iota, max_elongation,
     min_L_grad_B, min_R0, r_singularity,
     L_grad_grad_B, B20_variation, beta,
     DMerc_times_r2] = output

    try:
        assert axis_length > 0.
        assert np.fabs(iota) >= 0.2
        assert max_elongation <= 10.
        assert np.fabs(min_L_grad_B) >= 0.1
        assert np.fabs(min_R0) >= 0.3
        assert r_singularity >= 0.05
        assert np.fabs(L_grad_grad_B) >= 0.1
        assert B20_variation <= 5.
        assert beta >= 1e-4
        assert DMerc_times_r2 > 0.
        return True

    except AssertionError:
        return False