import numpy as np


def CR(pre, real, cap):
    pre = np.array(pre)
    real = np.array(real)
    msc = 0
    for i in range(len(pre)):
        msc += ((pre[i]-real[i])/cap)**2
    return 1-np.sqrt(msc/len(pre))

def MAE(pre, real, cap):
    pre = np.array(pre)
    real = np.array(real)
    msc = 0
    for i in range(len(pre)):
        msc += np.abs(pre[i]-real[i])
    return msc/(len(pre)*cap)

def compute_gte(actual_power, predicted_power):
    n = len(actual_power)
    if n < 2:
        raise ValueError("The length of actual_power and predicted_power must be at least 2.")
    
    summation = 0
    for i in range(n - 1):
        Ppi = actual_power[i]
        Ppi1 = actual_power[i + 1]
        Pmi = predicted_power[i]
        Pmi1 = predicted_power[i + 1]
        
        term1 = 1 + (Ppi1 - Ppi) * (Pmi1 - Pmi)
        term2 = np.sqrt((Ppi1 - Ppi)**2 + 1)
        term3 = np.sqrt((Pmi1 - Pmi)**2 + 1)
        
        summation += term1 / (term2 * term3 + 1)
    
    gte = 1 - 0.5 * (1 / (n - 1)) * summation
    return gte

def find_extrema(data):
    """
    Find local minima and maxima in the data.
    Returns indices of extrema points.
    """
    minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
    maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1
    return minima, maxima

def compute_pte(actual_power, predicted_power):
    n = len(actual_power)
    minima, maxima = find_extrema(actual_power)
    extrema = np.sort(np.concatenate((minima, maxima)))
    z = len(extrema)
    
    if z == 0:
        raise ValueError("No extrema found in the actual power data.")
    
    summation = 0
    for i in extrema:
        Ppi = actual_power[i]
        Pmi = predicted_power[i]
        summation += 1 / (abs(Ppi - Pmi) + 1)
    
    pte = 1 - (1 / z) * summation
    return pte, extrema

def time_delay_error(P_p, P_m):
    """
    Calculate the Time-Delay Error (TDE).

    Parameters:
    P_p (list or numpy array): Actual power at each time step.
    P_m (list or numpy array): Predicted power at each time step.

    Returns:
    float: The Time-Delay Error (TDE).
    """
    n = len(P_p)
    
    # Ensure the lengths of P_p and P_m are the same
    if n != len(P_m):
        raise ValueError("The lengths of P_p and P_m must be equal.")
    
    # Calculate E1
    E1 = np.sum((P_p - P_m) ** 2)
    
    # Function to calculate d(P_pi, P_mj)
    def d(P_pi, P_mj, P_p_prev=None, P_m_prev=None):
        if P_p_prev is None or P_m_prev is None:
            return (P_pi - P_mj) ** 2
        else:
            return (P_pi - P_mj) ** 2 + min(
                d(P_p_prev, P_m_prev),
                d(P_p_prev, P_mj),
                d(P_pi, P_m_prev)
            )
    
    # Calculate E2 using dynamic programming to store intermediate results
    dp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0:
                dp[i][j] = (P_p[i] - P_m[j]) ** 2
            else:
                dp[i][j] = (P_p[i] - P_m[j]) ** 2 + min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1]
                )
    
    E2 = dp[-1][-1]
    
    # Calculate TDE
    TDE = 1 - (E2 / E1)
    
    return TDE