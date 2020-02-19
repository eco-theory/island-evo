import os
import numpy as np



def main():
    
    sig_S_list = [0.16,.2,.24,.28]
    rep_num = 10

    for S_ind in [1,2,3]:
    # for S_ind in [0]:

        hours = 48
        mins =  0


        file_name = 'submit_seldiffs3_S{}'.format(S_ind)

        with open(file_name,'w') as file:
            file.write("""#!/bin/bash
#
#SBATCH --job-name=sd3_{S_ind}
#SBATCH --output=error_files/sd3_{S_ind}_%a.out
#SBATCH --error=error_files/sd3_{S_ind}_%a.err
#SBATCH --array=0-{rep}
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH -p normal
#SBATCH --cpus-per-task=1

srun python3  selective_diffs3.py {S_ind} $SLURM_ARRAY_TASK_ID""".format(S_ind=S_ind, rep = rep_num-1,hours=hours,mins=mins))

        os.system("sbatch {}".format(file_name))

if __name__ == '__main__':
    main()