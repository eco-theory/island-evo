import numpy as np
import sys
import os.path

def main():
    #sys.argv[0] is name of script.
    

    epoch_num = 1000
    K_vec = np.array([75,100,200,300,400])
    num1 = len(K_vec)


    for ind0 in [3]:
        num_alive_array = np.zeros((num1,epoch_num+2))
        num_alive_array[:] = np.nan

        extinct_time_list = []
        epoch_time_list = []
    
        for ind1 in range(1,num1):
                
            file_name = 'saturation{}.2_K{}.npz'.format(ind0,ind1)

            if os.path.isfile(file_name):

                with np.load(file_name) as data:
                    exp_data = data['class_obj'].item()

                num_alive = np.sum(exp_data['n_alive'],axis=0)
                num_alive = np.float32(num_alive)
                num_alive[num_alive==0] = np.nan
                num_alive_array[ind1,:] = num_alive

                extinct_time_list.append(exp_data['extinct_time_array'])
                epoch_time_list.append(exp_data['epoch_time_list'])


        M = exp_data['M']
        mu = exp_data['mu']
        logN = -exp_data['thresh']
        epoch_timescale = exp_data['epoch_timescale']
        gamma = exp_data['gamma']
        D = exp_data['D']

        data = {'epoch_num': epoch_num, 'K_vec':K_vec, 
            'logN':logN, 'num_alive_array':num_alive_array, 'D':D, 'M':M, 'mu':mu, 'gamma':gamma, 'epoch_timescale ':epoch_timescale,
            'extinct_time_list':extinct_time_list,'epoch_time_list':epoch_time_list}

        file_name = 'saturation{}.2_summary'.format(ind0)
        np.savez(file_name,data = data)


    for ind0 in range[3]:
        num_alive_array = np.zeros((num1,epoch_num+2))
        num_alive_array[:] = np.nan

        extinct_time_list = []
        epoch_time_list = []
    
        for ind1 in range(0,num1):
                
            file_name = 'saturation{}_K{}.npz'.format(ind0,ind1)

            if os.path.isfile(file_name):

                with np.load(file_name) as data:
                    exp_data = data['class_obj'].item()

                num_alive = np.sum(exp_data['n_alive'],axis=0)
                num_alive = np.float32(num_alive)
                num_alive[num_alive==0] = np.nan
                num_alive_array[ind1,:] = num_alive

                extinct_time_list.append(exp_data['extinct_time_array'])
                epoch_time_list.append(exp_data['epoch_time_list'])


        M = exp_data['M']
        mu = exp_data['mu']
        logN = -exp_data['thresh']
        epoch_timescale = exp_data['epoch_timescale']
        gamma = exp_data['gamma']
        D = exp_data['D']

        data = {'epoch_num': epoch_num, 'K_vec':K_vec, 
            'logN':logN, 'num_alive_array':num_alive_array, 'D':D, 'M':M, 'mu':mu, 'gamma':gamma, 'epoch_timescale ':epoch_timescale,
            'extinct_time_list':extinct_time_list,'epoch_time_list':epoch_time_list}

        file_name = 'saturation{}_summary'.format(ind0)
        np.savez(file_name,data = data)


if __name__ == '__main__':
    main()