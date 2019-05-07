import os
import numpy as np



def main():
    
    epoch_time_list = [2, 3, 4, 6, 10, 30]
    thresh_list = [1.5, 2, 3, 4, 6, 10]
    rep_num = 5

    for t_ind in range(len(epoch_time_list)):
        for thresh_ind in range(len(thresh_list)):
            for rep_ind in range(rep_num):
    
                file_name0 = '/home/groups/dsfisher/island_evo/epoch_timescale/epoch_time_scaling0_t{}_thresh{}_rep{}.npz'.format(t_ind,thresh_ind,rep_ind)
                file_size = os.path.getsize(file_name0)

                if file_size < 1e6:
                    hours = 5
                    mins =  0


                    file_name = 'submit_epoch_time_scaling0_t{}_thresh{}_rep{}'.format(t_ind,thresh_ind,rep_ind)

                    with open(file_name,'w') as file:
                        file.write("""#!/bin/bash
#
#SBATCH --job-name=ets0_{t_ind}_{thresh_ind}_{rep_ind}
#SBATCH --output=error_files/ets0_{t_ind}_{thresh_ind}_{rep_ind}.out
#SBATCH --error=error_files/ets0_{t_ind}_{thresh_ind}_{rep_ind}.err
#SBATCH --array=0
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

srun python3  epoch_time_scaling0.py {t_ind} {thresh_ind} {rep_ind}""".format(t_ind = t_ind, thresh_ind = thresh_ind,rep_ind = rep_ind, rep = rep_num-1,hours=hours,mins=mins))

                    os.system("sbatch {}".format(file_name))



if __name__ == '__main__':
    main()


    