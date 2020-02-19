import os
import numpy as np
import island_evo_simulation_methods as ie



def main():
    
    sig_S_list = [0.16,.2,.24,.28]
    rep_num = 10

    for S_ind in range(len(sig_S_list)):
        for rep_ind in range(rep_num):
        	file_name = 'data/selective_diffs3_S{}_rep{}'.format(S_ind,rep_ind)
        	ie.combine_save_files(file_name)


if __name__ == '__main__':
    main()