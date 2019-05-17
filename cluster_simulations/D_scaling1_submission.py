import os
import numpy as np



def main():

    D_list = [15,30,50,100]
    g_list = [-0.9,-0.77]
    rep_num = 3

    for ind0 in range(len(D_list)):
        for ind1 in range(len(g_list)):
            for ind2 in range(rep_num):
    
                hours = 5
                mins =  0


                file_name = 'submit_D_scaling1_D{}_g{}_rep{}'.format(ind0,ind1,ind2)

                with open(file_name,'w') as file:
                    file.write("""#!/bin/bash
#
#SBATCH --job-name=D1_{ind0}_{ind1}_{ind2}
#SBATCH --output=error_files/D1_{ind0}_{ind1}_{ind2}.out
#SBATCH --error=error_files/D1_{ind0}_{ind1}_{ind2}.err

#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

srun python3  D_scaling1.py {ind0} {ind1} {ind2}""".format(ind0 = ind0, ind1 = ind1, ind2 = ind2,hours=hours,mins=mins))

                os.system("sbatch {}".format(file_name))

if __name__ == '__main__':
    main()