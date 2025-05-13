import numpy as np
import scipy.special as sp
import zuko
import torch

class ChronoFlow(object):
    """
    Class to handle working with the ChronoFlow model.
    """

    @staticmethod
    def loadModel(weights_file):
        """
        Load the model from the specified path.
        """
        model = zuko.flows.NSF(1, 3, transforms=3)
        weights = torch.load(weights_file)
        model.load_state_dict(weights)
        return model

    def __init__(self,
                 weights_file,
                 bounds_logA_Myr=(0, 4.14),
                 bounds_logProt=(-1.75,2.5),
                 bounds_BPRP0=(-0.5,5),
                prior_BPRP0_lims=(-0.05,3.8)
    ):
        
        """
        Initialize the ChronoFlow class with a model.
        """
        self.model = ChronoFlow.loadModel(weights_file)
        self.bounds_logA_Myr=bounds_logA_Myr
        self.bounds_logProt=bounds_logProt
        self.bounds_BPRP0=bounds_BPRP0
        self.prior_BPRP0_lims=prior_BPRP0_lims



        


    def calcAgePrior(self,
                     logA_Myr,
                    lim_low=None,
                    lim_high=None):
        """
        Function to calculate the prior on log age (in Myr) given a log age value.
        Parameters:
        logA_Myr : float
            The log age value (in Myr).
        lim_low : float
            The lower limit of the uniform log age distribution.
        lim_high : float
            The upper limit of the uniform log age distribution.
        Returns:
        val : float
            The probability value of the prior.
        """
        if lim_low is None:
            lim_low = self.bounds_logA_Myr[0]
        if lim_high is None:
            lim_high = self.bounds_logA_Myr[1]
        if lim_low >= lim_high:
            raise ValueError("Lower limit must be less than upper limit.")
        if logA_Myr < lim_low:
            raise ValueError("logA_Myr is less than the lower limit.")
        if logA_Myr > lim_high:
            raise ValueError("logA_Myr is greater than the upper limit.")

        prob_val = 1 / (lim_high - lim_low)
        if (logA_Myr < lim_low) | (logA_Myr > lim_high):
            return 0
        else:
            return prob_val

    def calcColourPrior(self,
                        C0,
                        sigma_C0,
                        lim_low,
                        lim_high):
        """
        Function to calculate the prior on (BP-RP)_0 given a colour and its uncertainty.
        Parameters:
        C0 : float
            The observed de-reddened (BP-RP)_0 value.
        sigma_C0 : float
            The photometric uncertainty of the colour.
        lim_low : float
            The lower limit of the uniform (BP-RP)_0 distribution.
        lim_high : float
            The upper limit of the uniform (BP-RP)_0 distribution.
        Returns:
        prob_val : float
            The probability value of the prior.
        """

        if lim_low is None:
            lim_low = self.bounds_BPRP0[0]
        if lim_high is None:
            lim_high = self.bounds_BPRP0[1]
        if lim_low >= lim_high:
            raise ValueError("Lower limit must be less than upper limit.")
        if C0 < lim_low:
            raise ValueError("C0 is less than the lower limit.")
        if C0 > lim_high:
            raise ValueError("C0 is greater than the upper limit.")
        if sigma_C0 <= 0:
            raise ValueError("sigma_C0 must be greater than 0.")

        mult_factor = 1 / ((lim_high-lim_low) * (np.sqrt(2 * np.pi * sigma_C0**2)))
        int_const = -np.sqrt(np.pi/2) * sigma_C0
        int_eval = (sp.erf((C0-lim_high)/(sigma_C0 * np.sqrt(2)))) - sp.erf((C0-lim_low)/(sigma_C0 * np.sqrt(2)))
        prob_val = mult_factor * int_const * int_eval
        
        return prob_val