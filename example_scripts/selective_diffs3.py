
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.
    S_ind = int(sys.argv[1])
    rep_ind = int(sys.argv[2])

    sig_S_list = [0.16,.2,.24,.28]
    K_list = [400, 600, 600, 600]

    file_name = 'data/selective_diffs3_S{}_rep{}'.format(S_ind,rep_ind)

    seed = 123*rep_ind
    
    K = K_list[S_ind]
    D = 50
    m = 1e-5
    gamma = -0.8
    thresh = -40
    invasion_freq_factor = 1
    sample_num = 5
    mu = 4
    epoch_timescale = 3
    epoch_num = np.inf
    corr_mut = 0
    sig_S = sig_S_list[S_ind]

    epochs_to_save_traj = None
    long_epochs = [30,100,300,600,900,1200,1500]
    long_factor = 3

    ie.IslandsEvoAdaptiveStep(file_name, D, K, m, gamma, thresh, mu, seed, epoch_timescale, epoch_num,
                 sample_num, corr_mut=corr_mut, sig_S=sig_S, epochs_to_save_traj = epochs_to_save_traj,
                 long_epochs=long_epochs,long_factor=long_factor,invasion_freq_factor = invasion_freq_factor)

if __name__ == '__main__':
    main()
