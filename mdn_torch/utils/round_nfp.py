import numpy as np

def round_nfp(sample):
    """
    Rounds the nfp value in the sample.
    
    Parameters
    ----------
    sample : list
        The sample containing the nfp value.
    
    Returns
    -------
    list
        The sample with the nfp value rounded.
    """
    nfp = sample[6]
    # Performs ceiling operation on nfp
    nfp = np.clip(nfp, 1., None)
    nfp = np.round(nfp)
    nfp = nfp.astype(int)
    sample[6] = nfp
    return sample