
import numpy as np
import time
import sys
import island_evo_simulation_methods as ie


def main():
    #sys.argv[0] is name of script.

    file_name = 'test_single_epoch'

    seed = 878943+32948
    np.random.seed(seed=seed)
    
    sim_params = {}
    species_params = {}

    D = 20
    K = 50
    M = 10  #M = log(1/m)
    m = np.exp(-M)
    gamma = -0.9
    sim_params['logN'] =  10*M

    sim_params['D'] = D
    sim_params['K'] = K
    sim_params['M'] = M
    sim_params['m'] = m
    sim_params['gamma'] = gamma

    t0 = K**(3/2)
    dt = .1
    sample_time = 10
    coarse_sample_time = t0
    epoch_time = 5*t0
    sim_params = ie.set_time_params(dt,epoch_time,sample_time,coarse_sample_time,sim_params)

    n0 = ie.initialization_many_islands(D,K,1,m,'equal')
    species_params['V'] = ie.generate_interactions_with_diagonal(K,gamma)
    species_params['fitness_diffs'] = 0
    species_params['self_interactions'] = 0

    species_params['extinction_times'] = np.inf*np.ones((K))

    sim_time = 0
    n1, sim_time, averages, species_params = ie.single_epoch_island_evo_sim(n0,sim_time,species_params,sim_params)

    np.savez(file_name,n1=n1,sim_time=sim_time,averages=averages,species_params=species_params,sim_params=sim_params)

if __name__ == '__main__':
    main()
