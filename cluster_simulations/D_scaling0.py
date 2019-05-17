
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.
    ind0 = int(sys.argv[1])
    ind1 = int(sys.argv[2])
    ind2 = int(sys.argv[3])

    D_list = [15,30,50,100]
    g_list = [-0.9,-0.77]

    file_name = 'D_scaling0_D{}_g{}_rep{}'.format(ind0,ind1,ind2)

    seed = 10*ind2
    
    D = D_list[ind0]
    K = 50
    M = np.log(1e5)
    gamma = g_list[ind1]
    thresh = -4*M
    inv_fac = np.exp(5.)

    dt = 0.1
    mu = 5
    epoch_timescale = 40
    epoch_num = 150
    sample_num = 20

    ie.AntisymEvo(file_name, D, K, M, gamma, thresh, inv_fac, dt, mu, seed, epoch_timescale, epoch_num, sample_num)

if __name__ == '__main__':
    main()
