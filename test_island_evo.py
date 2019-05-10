
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.

    file_name = 'test_island_evo'

    seed = 878943+32948
    
    D = 20
    K = 100
    M = 10  #M = log(1/m)
    gamma = -0.998
    logN = 10*M
    thresh = -logN

    t0 = K**(3/2)
    dt = .1
    sample_time = 10
    epoch_time = 5*t0
    epoch_num = 4
    sample_epoch_start = 3

    epoch_times = epoch_time*np.ones((epoch_num))

    # ie.ManyIslandsSim(file_name,D,K,M,gamma,thresh,dt,seed,epoch_time,epoch_num,sample_time)
    
    # ie.InfiniteIslandsSim(file_name,K,M,gamma,dt,seed,epoch_times,sample_time)
    
    ie.ExtinctionTimeLong(file_name,D,K,M,gamma,thresh,dt,seed,epoch_time,epoch_num,sample_time)

if __name__ == '__main__':
    main()
