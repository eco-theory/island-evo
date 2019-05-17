import numpy as np
import sys

def main():
    #sys.argv[0] is name of script.
    
    epoch_time_list = [2, 3, 4, 6, 10, 30]
    K_list = [30, 50, 100, 150]

    K_num = len(K_list)
    t_num = len(epoch_time_list)
    rep_num = 5

    epoch_num = 300


    epoch_timescale_vec = np.zeros((t_num))  #units of sqrt(K) M
    K_vec = np.zeros((K_num))   #units of M
    seed_vec = np.zeros((rep_num))

    num_alive_array = np.zeros((K_num,t_num,rep_num,epoch_num+2))
    num_alive_array[:] = np.nan
    

    for K_ind in range(K_num):
        for t_ind in range(t_num):
            for rep_ind in range(rep_num):
                
                file_name = 'epoch_time_scaling0_K{}_t{}_rep{}.npz'.format(K_ind,t_ind,rep_ind)

                with np.load(file_name) as data:
                    exp_data = data['class_obj'].item()

                num_alive = np.sum(exp_data['n_alive'],axis=0)
                epochs = len(num_alive)
                num_alive_array[K_ind,t_ind,rep_ind,:epochs] = num_alive

                seed_vec[rep_ind] = exp_data['seed']

            

            epoch_timescale_vec[t_ind] = exp_data['epoch_timescale']
        K_vec[K_ind] = -exp_data['K']


    D = exp_data['D']
    m = exp_data['m']
    M = exp_data['M']
    mu = exp_data['mu']
    gamma = exp_data['gamma']
    logN = -exp_data['thresh']

    data = {'t_num':t_num, 'K_num': K_num, 'rep_num': rep_num, 'epoch_num': epoch_num, 'epoch_timescale_vec':epoch_timescale_vec, 
        'logN_vec':logN_vec, 'seed_vec':seed_vec, 'num_alive_array':num_alive_array,'D':D, 'm':m, 'M':M, 'mu':mu, 'gamma':gamma,'logN':logN}

    file_name = 'K_init_scaling0_summary'
    np.savez(file_name,data = data)


if __name__ == '__main__':
    main()