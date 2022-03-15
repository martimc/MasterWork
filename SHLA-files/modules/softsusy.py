# coding: utf8

#######################################
#                                     #
#  softsusy module for point_sampler  #
#                                     #
#######################################

from mpi4py import MPI
from modules import pyslha
from modules import shelltimeout
from collections import OrderedDict
import sys
import subprocess
import os


# ========== Module configuration class ==========

class mcfg:
    timeout_limit = 30.
    poll_interval = 5.


# ========== Initialization function ==========

def init(cfg):

    print(('{} Initializing module: softsusy'.format(cfg.pref)))

    # Root process: Check for BLOCK SOFTSUSY in base SLHA input file
    if cfg.my_rank == 0:
        pyslha_dict = pyslha.readSLHAFile(cfg.base_slha_file)
        if 'SOFTSUSY' in pyslha_dict:
            print(('{} BLOCK SOFTSUSY already exists in file {}. Will use existing values.'.format(cfg.pref, cfg.base_slha_file)))
        else:
            print(('{} No BLOCK SOFTSUSY found in file {}. Will use default softsusy settings.'.format(cfg.pref, cfg.base_slha_file)))

    return cfg



# ========== Point-by-point calculation ==========

def pointcalc(cfg, slha_path):

    is_accepted = True

    # Setup for softsusy run
    #ss_path           = './softsusy_bin/softpoint.x'
    ss_path           = './softsusy_bin/lt-softpoint.x' # Use this executable found in ./libs in Softsus-directory if possible
    
    # Defining the shell command
    cmd  = ss_path + ' ' + 'leshouches' + ' < ' + slha_path 


    # Run softsusy
    try:
        proc, output, timed_out = shelltimeout.shrun(cmd, mcfg.timeout_limit, use_exec=False, poll_interval=mcfg.poll_interval)
    except Exception as e:
        print(("{} ERROR: Caught error: {}".format(cfg.pref, str(e))))
        print(("{} Will reject point.".format(cfg.pref)))
        is_accepted = False
        return is_accepted

    # Check for timeout
    if timed_out:
        print(("{} WARNING: SOFTSUSY timeout".format(cfg.pref)))    
        is_accepted = False
        return is_accepted

    # Check for error in spectrum generation
    slha_file_dict = pyslha.readSLHA(str(output))
    
    if ('SPINFO' not in slha_file_dict):
        # JV hack: print out whole slha-file when softsusy fails without message 
        sys.stdout.write('%s' % (slha_file_dict))                                        
        fail_reason = 'Unknown reason'
        print(("{} SOFTSUSY failed: {}".format(cfg.pref, fail_reason)))
        is_accepted = False
        return is_accepted
    elif (4 in list(slha_file_dict['SPINFO'].entries.keys())):
        # JV hack: print out whole slha-file when softsusy fails without message
        # sys.stdout.write('%s' % (slha_file_dict))
        fail_reason = ' '.join(slha_file_dict['SPINFO'].entries[4][1:])
        print(("{} SOFTSUSY failed: {}".format(cfg.pref, fail_reason)))
        is_accepted = False
        return is_accepted

    # If no error in spectrum generation, perform LSP check
    accepted_LSP_list = [1000022]
    susy_mass_dict    = {}
    no_check_list     = [24,25,35,36,37]      # don't include the masses of these particles when checking for LSP

    for entry in list(slha_file_dict['MASS'].entries.values()):
        if entry[0] not in no_check_list:
            susy_mass_dict[entry[0]] = abs(entry[1])

    min_pdg = min(susy_mass_dict, key=susy_mass_dict.get)

    if not min_pdg in accepted_LSP_list:
        print(("{} Point rejected after running SOFTSUSY: LSP is {}".format(cfg.pref, min_pdg)))
        is_accepted = False
        return is_accepted

    # If everything is A-OK make SLHA file
    pyslha.writeSLHAFile(slha_path,slha_file_dict)
      
    # Done
    return is_accepted
