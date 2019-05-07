import os
import numpy as np



def main():
    


    epoch_time_list = [2, 3, 4, 6, 10, 30]
    thresh_list = [1.5, 2, 3, 4, 6, 10]
    rep_num = 5

    for t_ind in range(len(epoch_time_list)):
        for thresh_ind in range(len(thresh_list)):
    
            hours = 5
            mins =  0


            file_name = 'submit_epoch_time_scaling0_t{}_thresh{}'.format(t_ind,thresh_ind)

            with open(file_name,'w') as file:
                file.write("""#!/bin/bash
#
#SBATCH --job-name=ets0_{t_ind}_{thresh_ind}
#SBATCH --output=error_files/ets0_{t_ind}_{thresh_ind}_%a.out
#SBATCH --error=error_files/ets0_{t_ind}_{thresh_ind}_%a.err
#SBATCH --array=0-{rep}
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

srun python3  epoch_time_scaling0.py {t_ind} {thresh_ind} $SLURM_ARRAY_TASK_ID""".format(t_ind = t_ind, thresh_ind = thresh_ind,rep = rep_num-1,hours=hours,mins=mins))

            os.system("sbatch {}".format(file_name))

if __name__ == '__main__':
    main()