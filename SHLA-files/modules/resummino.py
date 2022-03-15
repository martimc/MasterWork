# coding: utf8

########################################
#                                      #
#  resummino module for point_sampler  #
#                                      #
########################################


"""
Runs resummino for a parameter point specified in SLHA file

@ Are Raklev (2020)
"""

from mpi4py import MPI
from modules import pyslha
from modules import shelltimeout
#from collections import OrderedDict
from math import sqrt
import sys
import subprocess32
import os
import shutil


# ========== Module configuration class ==========

class mcfg:
    timeout_limit = 60.*60.*2
    poll_interval = 10
    cmd = [ ]


# ========== Initialization function ==========

def init(cfg):

    print(('{} Initializing module: resummino'.format(cfg.pref)))

    # Each process except 0 prepares a directory with the resummino binary and settings file
    if cfg.my_rank > 0 :
        mcfg.resummino_bin_dir = os.path.join(cfg.temp_dir, 'resummino_bin_'+ str(cfg.my_rank)+'/')
        try:
            shutil.copytree('resummino_bin/', mcfg.resummino_bin_dir)
        except OSError as e:
            if e.errno == 17:
                pass
            else:
                raise

    return cfg


# ========== Point-by-point calculation ==========                                                                                                   

def pointcalc(cfg, slha_path):

    is_accepted = True

    # Copy SLHA file to directory with resummino binary
    shutil.copy(slha_path, os.path.join(mcfg.resummino_bin_dir,'slha.in'))

    # Save base directory  
    base_dir = os.getcwd()

    # Resummino run command:                                                                                                                        
    cmd_resummino = './resummino'

    # Create resummino block header for SLHA file 
    block_name = 'RESUMMINO_OUTPUT'
    resummino_block = pyslha.Block(block_name)
    resummino_block.addComment('title', 'PID1 PID2 COM-energy [GeV] LO [fb] LO_err [fb] NLO [fb] NLO_err [fb]  NLO+NLL [fb] NLO+NLL_err [fb]')
    
    # Move to directory with resummino executable and output files 
    os.chdir(mcfg.resummino_bin_dir)

    ##
    # Loop over processes and energies
    ##
    processes = []
    energies = [7000, 8000, 13000, 14000]
    gauginos = [1000022, 1000023, 1000025, 1000035, 1000024, 1000037, -1000024, -1000037]

    # All neutralino and chargino production processes  
    processes = [(1000022, 1000022), (1000022, 1000023), (1000022, 1000025), (1000022, 1000035), (1000022, 1000024), (1000022, 1000037), (1000022, -1000024), (1000022, -1000037),                    (1000023, 1000023), (1000023, 1000025), (1000023, 1000035), (1000023, 1000024), (1000023, 1000037), (1000023, -1000024), (1000023, -1000037), (1000025, 1000025),                    (1000025, 1000035), (1000025, 1000024), (1000025, 1000037), (1000025, -1000024), (1000025, -1000037), (1000035, 1000035), (1000035, 1000024), (1000035, 1000037),                    (1000035, -1000024), (1000035, -1000037), (1000024, -1000024), (1000024, -1000037), (1000037, -1000037)]

    #print processes

    for energy in energies:
        for process in processes:
            PID1 = process[0]
            PID2 = process[1]

            pid1_opt = "--particle1="+str(PID1)
            pid2_opt = "--particle2="+str(PID2)
            com_opt = "--center_of_mass_energy="+str(energy)

            # Form resummino command
            mcfg.cmd = [cmd_resummino, pid1_opt, pid2_opt, com_opt, "--lo", "resummino.in"]
            #mcfg.cmd = [cmd_resummino, pid1_opt, pid2_opt, com_opt, "resummino.in"]

            # Run calculation (and print info about process to terminal if you want to) 
            print("{} : Process {} {} {}: {}".format(cfg.pref, PID1, PID2, energy, mcfg.cmd))
            #proc, output_, timed_out = shelltimeout.shrun(mcfg.cmd, mcfg.timeout_limit, poll_interval=mcfg.poll_interval, prospino=True)
            #proc, output_, timed_out = shelltimeout.shrun(mcfg.cmd, mcfg.timeout_limit, poll_interval=mcfg.poll_interval)

            # Trying this for LXPLUS which doesn't seem to like shelltimeout
            process = subprocess32.Popen(mcfg.cmd, stdout=subprocess32.PIPE, stderr=subprocess32.PIPE)
            output_, stderr = process.communicate()
            #print  stderr
            print output_

            # Read results file
            result = open("resummino_result").readline().split()
            print result

            # Writing results to resummino-block                                       
            resummino_block.newEntry(result)

    # Check for acceptable result
    is_accepted = True

    # Move back to base dir for file writing                                                                                                            
    os.chdir(base_dir)

    # Finally write cross sections to SLHA-file 
    slha_dict = pyslha.readSLHAFile(slha_path)
    slha_dict[block_name] = resummino_block
    pyslha.writeSLHAFile(slha_path, slha_dict, precision=4)

    return is_accepted


# ========== Reading of prospino.dat and writing SLHA file ==========

def ReadProspinoDAT(cfg, output, timed_out, prospino_block, PID1, PID2, is_accepted, base_dir, value_dict):
    """Function for reading prospino.dat
    Args:
        cfg             : object containing info about process rank, directory, etc.
        output          : (str) terminal output from prospino
        timed_out       : (bool) True if Prospino used too much time
        prospino_block  : pyslha block object for prospino output
        PID1            : (str) PID-code for finalstate particle
        PID2            : (str) PID-code for finalstate particle
        is_accepted     : (bool) True if point is accepted
        base_dir        : (str) base directory of point sampler
        value_dict      : (dict) containing all values to be written in slha-file
    """


    # Turn output into list of lines
    output = output.split('\n')
    if output[-1] == '':
        output = output[:-1]

    # Check for timeout
    if timed_out:
        print(("{} WARNING: PROSPINO timeout".format(cfg.pref)))    
        # Print last lines of output
        for line in output[-20:]:
            print(('{} PROSPINO output (last 20 lines): {}'.format(cfg.pref, line)))
        # Writing -1 for all values if some error occurs.
        for key in value_dict:
            if key != 'com_energy': value_dict[key] = -1
        # writing results to prospino-block                                                                       
        prospino_block.newEntry([PID1, PID2, int(value_dict['com_energy']),
                                 value_dict['LO'], value_dict['rel_LO_err'], value_dict['NLO'], value_dict['rel_NLO_err'], value_dict['NLO_ms'],
                                 value_dict['NLO_ms_at_05'], #value_dict['rel_NLO_ms_err_at_05'],                       
                                 value_dict['NLO_ms_at_2'],
                                 value_dict['NLO_ms_PDF'],
                                 value_dict['NLO_ms_aup'],
                                 value_dict['NLO_ms_adn'] ]) #, value_dict['rel_NLO_ms_err_at_2']])
        is_accepted = False
        return is_accepted

    # Check for problems in ouput
    for i,line in enumerate(output):
        if any(x in line.lower() for x in [ "problem with beta", "energy too small"]):
            for key in value_dict:
                if key != 'com_energy': value_dict[key] = 0.
            # writing results to prospino-block
            prospino_block.newEntry([PID1, PID2, int(value_dict['com_energy']),
                                value_dict['LO'], value_dict['rel_LO_err'], value_dict['NLO'], value_dict['rel_NLO_err'], value_dict['NLO_ms'],
                                value_dict['NLO_ms_at_05'], #value_dict['rel_NLO_ms_err_at_05'],
                                value_dict['NLO_ms_at_2'],
                                value_dict['NLO_ms_PDF'],
                                value_dict['NLO_ms_aup'],
                                value_dict['NLO_ms_adn'] ])  #, value_dict['rel_NLO_ms_err_at_2']])
            is_accepted = True
            return is_accepted
        
        elif any(x in line.lower() for x in ['abort', 'hard_stop', 'error']):
            # Writing -1 for all values if some error occurs.
            for key in value_dict:
                if key != 'com_energy': value_dict[key] = -1
            # writing results to prospino-block                                                                       
            prospino_block.newEntry([PID1, PID2, int(value_dict['com_energy']),
                                     value_dict['LO'], value_dict['rel_LO_err'], value_dict['NLO'], value_dict['rel_NLO_err'], value_dict['NLO_ms'],
                                     value_dict['NLO_ms_at_05'], #value_dict['rel_NLO_ms_err_at_05'],                       
                                     value_dict['NLO_ms_at_2'],
                                     value_dict['NLO_ms_PDF'],
                                     value_dict['NLO_ms_aup'],
                                     value_dict['NLO_ms_adn'] ]) # ,value_dict['rel_NLO_ms_err_at_2']])
            if i>0:
                print("{} PROSPINO failed at process {} {} {}: {}, {}".format(cfg.pref, PID1, PID2, value_dict['com_energy'], output[i].strip("\n"), output[i-1].strip("\n")))
            else:
                print( '{} PROSPINO failed at process {} {} {}: {} '.format(cfg.pref, PID1, PID2, value_dict['com_energy'], output[i-1].strip('\n')) )
            is_accepted = True
            return is_accepted


    # With no problems in output we are ready to process results in prospino.dat
    # Each point has 35 entries (0-30 PDF sets, 31 & 32 \alpha_s sets, 33 & 34 scale) 
    # For each line the content is
    # Index   Content
    #  9      LO cross section
    # 10      LO relative error
    # 11      NLO cross section
    # 12      NLO relative error
    # 13      K-factor (NLO/LO)
    # 14 = -2 LO_ms cross section
    # 15 = -1 NLO_ms cross section
    nsets = 35
    with open( os.path.join(mcfg.prosp_bin_dir, 'prospino.dat') ) as f:
        lines = f.readlines()
        for i in range(0,len(lines)-nsets,nsets):       # Loop over 35 entries at a time 
            try:
                # Lists with results
                scaleFac1  = lines[i].split()          # Scale = 1
                alphaup    = lines[i+nsets-4].split()  # \alpha_s at upper value
                alphadn    = lines[i+nsets-3].split()  # \alpha_s at lower value
                scaleFac05 = lines[i+nsets-2].split()  # Scale = 0.5
                scaleFac2  = lines[i+nsets-1].split()  # Scale = 2

            except Exception as e:
                print(("{} WARNING: Cannot parse PROSPINO output file. Process {} {} {} ignored: ".format(cfg.pref, PID1, PID2, value_dict['com_energy'], str(e))))
                is_accepted = False
                return is_accepted

            # Checking that the lines belong to each other (not necessary)
            try:
                assert  scaleFac05[6] == scaleFac1[6] and scaleFac1[6] == scaleFac2[6] and \
                        scaleFac05[7] == scaleFac1[7] and scaleFac1[7] == scaleFac2[7], "Something is wrong with prospino.dat. I read three lines where the masses does not agree."
                assert  scaleFac05[3] == scaleFac1[3] and scaleFac1[3] == scaleFac2[3], "Check your prospino.dat-file, the energies are not equal for the process I'm looking at."
                assert  scaleFac05[5] == '0.5' and scaleFac1[5] == '1.0' and scaleFac2[5] == '2.0', "Check your prospino.dat-file, the scales does not make sense for me. :("
            except Exception as e:
                print(("{} Process: {} {} {} Oooops, I did not dare to read this process in prospino.dat because {}:".format(cfg.pref, PID1, PID2, value_dict['com_energy'], str(e))))

            # Converting strings to floats
            try:
                scaleFac1[1:] = [float(entry) for entry in scaleFac1[1:]]
                alphaup[1:]   = [float(entry) for entry in alphaup[1:]]
                alphadn[1:]   = [float(entry) for entry in alphadn[1:]]
                scaleFac05[1:]= [float(entry) for entry in scaleFac05[1:]]
                scaleFac2[1:] = [float(entry) for entry in scaleFac2[1:]]
            except Exception as e:
                print(("{} Process: {} {} {} Oooops, I could not convert some of the stuff in prospino.dat to floats: {}".format(cfg.pref, PID1, PID2, value_dict['com_energy'], str(e))))

            try:
                # For processes containing squarks
                if scaleFac1[0] in ['sg', 'ss', 'sb']:
                    # First pass doing NLO calculations to get and store K-facors for all relevant quantities
                    if mcfg.cmd[1] == '1':
                        value_dict['k_fac05'] = scaleFac05[11] / scaleFac05[9]
                        value_dict['k_fac2']  = scaleFac2[11] / scaleFac2[9]
                        value_dict['k_fac1']  = scaleFac1[11] / scaleFac1[9]
                        value_dict['k_aup']   = alphaup[11] / alphaup[9]
                        value_dict['k_adn']   = alphadn[11] / alphadn[9]
#                        value_dict['rel_NLO_ms_err_at_05']  = sqrt(scaleFac05[12]**2 + 2*scaleFac05[10]**2) # rel NLO_ms err at 0.5
#                        value_dict['rel_NLO_ms_err_at_2']   = sqrt(scaleFac2[12]**2 + 2*scaleFac2[10]**2)   # rel NLO_ms err at 2
                        value_dict['NLO']                   = scaleFac1[11]*1e3                             # NLO [fb]
                        value_dict['rel_NLO_err']           = scaleFac1[12]                                 # rel_NLO_err
                        
                        # K-factors for PDF sets (*sigh*)
                        value_dict['k_PDF1'] = float(lines[i+1].split()[11]) / float(lines[i+1].split()[9])
                        value_dict['k_PDF2'] = float(lines[i+2].split()[11]) / float(lines[i+2].split()[9])
#                        print 'k_fac1', scaleFac1[11] / scaleFac1[9]
#                        print 'k_PDF1', float(lines[i+1].split()[11]) / float(lines[i+1].split()[9])
#                        print 'k_PDF2', float(lines[i+2].split()[11]) / float(lines[i+2].split()[9])

                    # Once again with more feeling!
                    # Second pass, do LO calulations and multiply by K-factor, add to SLHA-block
                    elif mcfg.cmd[1] == '0':
                        value_dict['com_energy']    = int(scaleFac1[3])             # Indices in SLHA-file should be integers
                        value_dict['LO']            = scaleFac1[9]*1e3              # LO [fb]
                        value_dict['rel_LO_err']    = scaleFac1[10]                 # rel_LO_err
                        value_dict['NLO_ms']        = value_dict['k_fac1'] * scaleFac1[-2]*1e3    # NLO_ms [fb]
                        value_dict['NLO_ms_at_05']  = value_dict['k_fac05'] * scaleFac05[-2]*1e3  # NLO_ms [fb]
                        value_dict['NLO_ms_at_2']   = value_dict['k_fac2'] * scaleFac2[-2]*1e3    # NLO_ms [fb]
                        # Approximation that K-factor is the same for all PDF members. Remove later? Must store all K-factors...
                        value_dict['NLO_ms_aup']    = value_dict['k_aup'] * alphaup[-2]*1e3       # NLO_ms [fb]
                        value_dict['NLO_ms_adn']    = value_dict['k_adn'] * alphadn[-2]*1e3       # NLO_ms [fb]  

                        # Calculate cross section error, see Sec 6.2 of 1510.03865
                        # Approximation that K-factor is the same for all PDF members. Remove later? Must store all K-factors...
                        ssum = 0
                        sigma0 = float(scaleFac1[14])*1E3
                        for j in range(1,nsets-4):
                            sigmaj = float(lines[i+j].split()[14])*1E3
#                            print 'Error xsec',j,sigmaj
                            ssum += (sigma0 - sigmaj)**2
                        ssum = sqrt(ssum)
#                        print 'Final error', ssum
                        value_dict['NLO_ms_PDF']     = value_dict['k_fac1'] * ssum                 # NLO_ms error from PDF

                        # Writing results to prospino-block
                        prospino_block.newEntry([PID1, PID2, value_dict['com_energy'],
                                        value_dict['LO'], value_dict['rel_LO_err'], value_dict['NLO'], 
                                        value_dict['rel_NLO_err'],
                                        value_dict['NLO_ms'],
                                        value_dict['NLO_ms_at_05'],
                                        value_dict['NLO_ms_at_2'],
                                        value_dict['NLO_ms_PDF'],
                                        value_dict['NLO_ms_aup'],
                                        value_dict['NLO_ms_adn']])

                # For processes without light squarks in final state
                elif scaleFac1[0] not in ['sg', 'ss', 'sb']:
                    value_dict['com_energy']            = int(scaleFac1[3])                             # Indices in SLHA-file should be integers
#                    value_dict['rel_NLO_ms_err_at_05']  = sqrt(scaleFac05[12]**2 + 2*scaleFac05[10]**2) # rel NLO_ms err at 0.5
                    value_dict['LO']                    = scaleFac1[9]*1e3                              # LO [fb]
                    value_dict['rel_LO_err']            = scaleFac1[10]                                 # rel_LO_err
                    value_dict['NLO']                   = scaleFac1[11]*1e3                             # NLO [fb]
                    value_dict['rel_NLO_err']           = scaleFac1[12]                                 # rel_NLO_err
                    value_dict['LO_ms']                 = scaleFac1[14]*1e3                             # LOms [fb]
                    value_dict['NLO_ms']                = scaleFac1[15]*1e3                             # NLOms [fb]
                    value_dict['NLO_ms_at_05']          = scaleFac05[15]*1e3                            # NLO_ms [fb]
                    value_dict['NLO_ms_at_2']           = scaleFac2[15]*1e3                             # NLO_ms [fb]
#                    value_dict['rel_NLO_ms_err_at_2']   = sqrt(scaleFac2[12]**2 + 2*scaleFac2[10]**2)   # rel NLO_ms err at 2
                    value_dict['NLO_ms_aup']            = alphaup[15]*1e3                               # NLO_ms [fb]
#                    value_dict['rel_NLO_ms_err_aup']    = sqrt(alphaup[12]**2 + 2*alphaup[10]**2)       # rel NLO_ms err
                    value_dict['NLO_ms_adn']            = alphadn[15]*1e3                               # NLO_ms [fb]
#                    value_dict['rel_NLO_ms_err_adn']    = sqrt(alphadn[12]**2 + 2*alphadn[10]**2)       # rel NLO_ms err

                    # Calculate cross section error, see Sec 6.2 of 1510.03865
#                    print 'Starting xsec calc...'
                    ssum = 0
                    sigma0 = float(scaleFac1[15])*1E3
                    for j in range(1,nsets-4):
                        sigmaj = float(lines[i+j].split()[15])*1E3
#                        print 'Error xsec',j,sigmaj      
                        ssum += (sigma0 - sigmaj)**2
                    ssum = sqrt(ssum)
#                    print 'Final error', ssum
                    value_dict['NLO_ms_PDF']            = ssum                                          # NLO_ms error from PDF

                    # Writing results to prospino-block
                    print 'Writing results for ', PID1, PID2
                    prospino_block.newEntry([PID1, PID2, value_dict['com_energy'],
                                        value_dict['LO'], value_dict['rel_LO_err'],
                                        value_dict['NLO'], value_dict['rel_NLO_err'], 
                                        value_dict['NLO_ms'],
                                        value_dict['NLO_ms_at_05'],
                                        value_dict['NLO_ms_at_2'],
                                        value_dict['NLO_ms_PDF'],
                                        value_dict['NLO_ms_aup'],
                                        value_dict['NLO_ms_adn']])

            except Exception as e:
                print(("{} Process: {} {} {} Oooops, something went wrong when fetching values from prospino.dat, process ignored. : {}".format(cfg.pref, PID1, PID2, value_dict['com_energy'], str(e))))
                is_accepted = False
                return is_accepted

    return is_accepted



