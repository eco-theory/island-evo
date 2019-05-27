
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.
    ind0 = int(sys.argv[1])
    # ind1 = int(sys.argv[2])
    # ind2 = int(sys.argv[3])

    K_list = [75,100,200,300,400]

    file_name = 'saturation3_K{}'.format(ind0)

    seed = 13
    
    D = 50
    K = K_list[ind0]
    M = np.log(1e5)
    gamma = -0.9
    thresh = -4*M
    inv_fac = np.exp(5.)

    dt = 0.1
    mu = 5
    epoch_timescale = 16
    epoch_num = 1000
    sample_num = 20

    ie.AntisymEvo2(file_name, D, K, M, gamma, thresh, inv_fac, dt, mu, seed, epoch_timescale, epoch_num, sample_num)

if __name__ == '__main__':
    main()
