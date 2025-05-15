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
                     logA_Myr):
        """
        Function to calculate the prior on log age (in Myr) given a log age value.
        Parameters:
        logA_Myr : float
            The log age value (in Myr).
        Returns:
        val : float
            The probability value of the prior.
        """
        lim_low = self.bounds_logA_Myr[0]
        lim_high = self.bounds_logA_Myr[1]
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
                        lim_low=None,
                        lim_high=None):
        """
        Function to calculate the prior on (BP-RP)_0 given a colour and its uncertainty.
        Parameters:
        C0 : float
            The observed de-reddened (BP-RP)_0 value.
        sigma_C0 : float
            The photometric uncertainty of the colour.
        Returns:
        prob_val : float
            The probability value of the prior.
        """

        lim_low = self.bounds_BPRP0[0]
        lim_high = self.bounds_BPRP0[1]
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
    
    def calcLogLikelihood(self,
                          logA_Myr,
                          BPRP0,
                          logProt,
                          logCerr=-1.55,
                          P_clmem=1,
                          P_out=0.05,
                ):
        """
        Function to calculate the conditional probability of P_rot given age, colour and photometric uncertainty.
        Parameters:
        logA_Myr : float
            The log age value (in Myr).
        BPRP0 : float
            The (BP-RP)_0 value.
        logProt : float 
            The log P_rot value.
        logCerr : float
            The log photometric uncertainty.
        P_clmem : float
            The probability of the star being a cluster member (this should be set to 1 when analyzing any individual star).
        P_out : float
            The probability of the star not following the anticipated rotational evolution pattern (we default this to 0.05).
        model : zuko.flows.NSF
            The normalizing flow model.
        Returns:
        prob_comb : float
            The conditional probability (as a natural log).
        """
        
        model = self.model

        # Calculate weighting of normalizing flow likelihood
        nf_weight = P_clmem * (1 - P_out)
        
        # Use model to evaluate probability
        ind_params = torch.tensor([logA_Myr,BPRP0,logCerr],requires_grad=False).to(torch.float32)
        cond_params = torch.tensor([logProt],requires_grad=False).to(torch.float32)
        nf_prob = model(ind_params).log_prob(cond_params).detach().numpy().item()
        
        # Background (ie. non-cluster member) probability
        bg_prob = np.log(1/(self.bounds_logProt[1] - self.bounds_logProt[0]))
        
        # Combine flow probability and background probability
        prob_val = sp.logsumexp([nf_prob,bg_prob],b=[nf_weight,1-nf_weight])

        return prob_val
    

    def calcPostAge(self,
                    logProt,
                    BPRP0,
                    logCerr=-1.55,
                    P_clmem=1,
                    P_out=0.05,
                    logA_Myr_grid=None,
                    ):

        """
        Function to calculate the posterior probability of age given P_rot, colour and photometric uncertainty.
        Parameters:
        logProt : float
            The log P_rot value.
        BPRP0 : float
            The (BP-RP)_0 value.
        logCerr : float
            The log photometric uncertainty.
        P_clmem : float
            The probability of the star being a cluster member (this should be set to 1 when analyzing any individual star).
        P_out : float
            The probability of the star not following the anticipated rotational evolution pattern (we default this to 0.05).
        logA_Myr_grid : np.ndarray
            The grid of log ages (in Myr) to evaluate the posterior probability.
        Returns:
        post : np.ndarray
            The posterior probability distribution of age.
        medLogA : float
            The median log age value (in yr).
        logA_Err : float
            The upper 1 sigma error on the log age value.
        logA_err : float
            The lower 1 sigma error on the log age value.
        """

        model=self.model
        if logA_Myr_grid is None:
            logA_Myr_grid = np.linspace(self.bounds_logA_Myr[0],
                                        self.bounds_logA_Myr[1],
                                        1000)

        logLikelihood = np.zeros(len(logA_Myr_grid))
        logColourPrior = np.zeros(len(logA_Myr_grid))
        logAgePrior = np.zeros(len(logA_Myr_grid))
        logPost = np.zeros(len(logA_Myr_grid))

        for i in range(len(logA_Myr_grid)):

            # Log likelihood probability
            logLikelihood[i] = self.calcLogLikelihood(logA_Myr=logA_Myr_grid[i],
                                                logProt=logProt,
                                                BPRP0=BPRP0,
                                                logCerr=logCerr,
                                                P_clmem=P_clmem,
                                                P_out=P_out)
            
            # Log colour prior probability
            logColourPrior[i] = np.log(self.calcColourPrior(BPRP0,10**logCerr))

            # Log age prior probability
            logAgePrior[i] = np.log(self.calcAgePrior(logA_Myr_grid[i]))
            
            # Combine all to get posterior
            logPost[i] = logLikelihood[i] + logColourPrior[i] + logAgePrior[i]

        # Convert to probability and normalize
        post = np.exp(logPost)
        post /= np.trapz(post,logA_Myr_grid)

        return post

    def getAgeSummStats(self,
                        post,
                        logA_Myr_grid=None,
                        n=1000):
        """
        Function to calculate the summary statistics of the posterior probability distribution.
        Parameters:
        post : np.ndarray
            The posterior probability distribution of age.
        n : int
            The number of samples to use for the summary statistics.
        Returns:
        medLogA : float
            The median log age value (in Myr).
        logA_Err : float
            The upper 1 sigma error on the log age value.
        logA_err : float
            The lower 1 sigma error on the log age value.
        """

        if logA_Myr_grid is None:
            logA_Myr_grid = np.linspace(self.bounds_logA_Myr[0],
                                        self.bounds_logA_Myr[1],
                                        1000)

        # Draw samples
        prob = post / np.sum(post)
        indices = np.random.choice(len(logA_Myr_grid), size=n, p=prob)
        samples = logA_Myr_grid[indices]

        # Get summary stats
        medLogA = np.median(samples)
        p84 = np.percentile(samples, 84)
        p16 = np.percentile(samples, 16)
        logA_Err = p84 - medLogA
        logA_err = p16 - medLogA

        return medLogA, logA_Err, logA_err