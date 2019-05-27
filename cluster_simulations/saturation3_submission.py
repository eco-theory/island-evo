import os
import numpy as np



def main():

    K_list = [75,100,200,300,400]
    rep_num = 1

    for ind0 in range(1,len(K_list)):
    
        hours = 24
        mins =  0


        file_name = 'submit_saturation3_K{}'.format(ind0)

        with open(file_name,'w') as file:
            file.write("""#!/bin/bash
#
#SBATCH --job-name=s3_{ind0}
#SBATCH --output=error_files/s3_{ind0}.out
#SBATCH --error=error_files/s3_{ind0}.err

#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

srun python3 saturation3.2.py {ind0}""".format(ind0 = ind0,hours=hours,mins=mins))

        os.system("sbatch {}".format(file_name))

if __name__ == '__main__':
    main()