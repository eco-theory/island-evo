import numpy as np
import sys

def main():
    #sys.argv[0] is name of script.
    
    t_num = 6
    thresh_num = 6
    rep_num = 5

    epoch_num = 300


    epoch_timescale_vec = np.zeros((t_num))  #units of sqrt(K) M
    logN_vec = np.zeros((thresh_num))   #units of M
    seed_vec = np.zeros((rep_num))

    num_alive_array = np.zeros((t_num,thresh_num,rep_num,epoch_num+2))
    num_alive_array[:] = np.nan
    

    for t_ind in range(t_num):
        for thresh_ind in range(thresh_num):
            for rep_ind in range(rep_num):
                
                file_name = 'epoch_time_scaling0_t{}_thresh{}_rep{}.npz'.format(t_ind,thresh_ind,rep_ind)

                with np.load(file_name) as data:
                    exp_data = data['class_obj'].item()

                num_alive = np.sum(exp_data['n_alive'],axis=0)
                epochs = len(num_alive)
                num_alive_array[t_ind,thresh_ind,rep_ind,:epochs] = num_alive

                seed_vec[rep_ind] = exp_data['seed']

            logN_vec[thresh_ind] = -exp_data['thresh']

        epoch_timescale_vec[t_ind] = exp_data['epoch_timescale']


    D = exp_data['D']
    m = exp_data['m']
    M = exp_data['M']
    mu = exp_data['mu']
    gamma = exp_data['gamma']

    data = {'t_num':t_num, 'thresh_num': thresh_num, 'rep_num': rep_num, 'epoch_num': epoch_num, 'epoch_timescale_vec':epoch_timescale_vec, 
        'logN_vec':logN_vec, 'seed_vec':seed_vec, 'num_alive_array':num_alive_array,'D':D, 'm':m, 'M':M, 'mu':mu, 'gamma':gamma}

    file_name = 'epoch_time_scaling0_summary'
    np.savez(file_name,data = data)


if __name__ == '__main__':
    main()