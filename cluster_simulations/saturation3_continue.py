
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.
    ind0 = int(sys.argv[1])
    ind1 = int(sys.argv[2])
    # ind2 = int(sys.argv[3])

    # K_list = [75,100,200,300,400]

    file_name0 = 'saturation3_K{}.npz'.format(ind0)
    file_name1 = 'saturation3.2_K{}'.format(ind0)
    epoch_num = 1000
    seed = 987
    
    with np.load(file_name0) as file:
        data = file['class_obj'].item()

    n_init = data['n_init_list'][-1]
    epochs = len(data['n_init_list'])
    V = data['V']
    n_alive = data['n_alive'][:,epochs-1]
    V_init = V[n_alive,:][:,n_alive]
    
    D = n_init.shape[0]
    K = n_init.shape[1]
    M = data['M']
    gamma = data['gamma']
    thresh = data['thresh']
    inv_fac = data['inv_fac']

    dt = data['dt']
    mu = data['mu']
    epoch_timescale = data['epoch_timescale']
    
    sample_num = data['sample_num']

    ie.AntisymEvo2(file_name1, D, K, M, gamma, thresh, inv_fac, dt, mu, seed, epoch_timescale, epoch_num, sample_num,n_init=n_init,V_init=V_init)

if __name__ == '__main__':
    main()
