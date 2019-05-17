import os
import numpy as np



def main():
    


    epoch_time_list = [2, 3, 4, 6, 10, 30]
    K_list = [30, 50, 100, 150]
    rep_num = 5

    for K_ind in range(len(K_list)):
        for t_ind in range(len(epoch_time_list)):
    
            hours = 5
            mins =  0


            file_name = 'submit_K_init_scaling0_K{}_t{}'.format(K_ind,t_ind)

            with open(file_name,'w') as file:
                file.write("""#!/bin/bash
#
#SBATCH --job-name=K0_{K_ind}_{t_ind}
#SBATCH --output=error_files/K0_{K_ind}_{t_ind}_%a.out
#SBATCH --error=error_files/K0_{K_ind}_{t_ind}_%a.err
#SBATCH --array=0-{rep}
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

srun python3  K_init_scaling0.py {K_ind} {t_ind} $SLURM_ARRAY_TASK_ID""".format(K_ind = K_ind, t_ind = t_ind,rep = rep_num-1,hours=hours,mins=mins))

            os.system("sbatch {}".format(file_name))

if __name__ == '__main__':
    main()