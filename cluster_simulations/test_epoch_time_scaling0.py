
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.
    t_ind = int(sys.argv[1])
    thresh_ind = int(sys.argv[2])
    rep_ind = int(sys.argv[3])

    epoch_time_list = [2, 3, 4, 6, 10, 30]
    thresh_list = [1.5, 2, 3, 4, 6, 10]

    file_name = '/home/groups/dsfisher/island_evo/epoch_timescale/epoch_time_scaling0_t{}_thresh{}_rep{}'.format(t_ind,thresh_ind,rep_ind)

    seed = 200+300*rep_ind
    
    D = 15
    K = 50
    M = 20
    gamma = -0.9
    thresh = -thresh_list[thresh_ind]*M
    inv_fac = np.exp(5.)

    dt = 0.1
    mu = 5
    epoch_timescale = epoch_time_list[t_ind]
    epoch_num = 3
    sample_num = 20

    ie.AntisymEvo(file_name, D, K, M, gamma, thresh, inv_fac, dt, mu, seed, epoch_timescale, epoch_num, sample_num)

if __name__ == '__main__':
    main()
