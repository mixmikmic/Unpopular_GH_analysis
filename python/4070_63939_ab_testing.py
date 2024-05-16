def ab_test_statistic(n_A, N_A, n_B, N_B):
    p_A = float(n_A) / N_B
    p_B = float(n_B) / N_B
    sigma_A = (p_A * (1.0 - p_A) / N_A)**0.5
    sigma_B = (p_B * (1.0 - p_B) / N_B)**0.5
    return (p_B - p_A) / (sigma_A**2 + sigma_B**2)**0.5

def two_sided_p_value(Z):
    from scipy.stats import norm
    return 2.0 * norm.cdf(-abs(Z))

Z = ab_test_statistic(n_A=227, N_A=500, n_B=250, N_B=500)
print "Z =", Z
p_value = two_sided_p_value(Z)
print "p =", p_value

