# coding: utf8

#######################################
#                                     #
#  prospino module for point_sampler  #
#                                     #
#######################################


"""
- Change of which processes to be calculated is done at approx. line 270.

@ Anders Kvellestad & Jon Vegard Sparre (2018)
"""

from mpi4py import MPI
from modules import pyslha
from modules import shelltimeout
from collections import OrderedDict
from math import sqrt
import sys
import subprocess
import os
import shutil


# ========== Module configuration class ==========

class mcfg:
    timeout_limit = 60.*60.*2
    poll_interval = 10
    cmd = [ ]


# ========== Initialization function ==========

def init(cfg):

    print(('{} Initializing module: prospino'.format(cfg.pref)))

    # Each process prepares a directory with the necessary Prospino files
    mcfg.prosp_bin_dir = os.path.join(cfg.temp_dir, 'prosp_bin_'+ str(cfg.my_rank)+'/')
    try:
        shutil.copytree('prospino_bin', mcfg.prosp_bin_dir)
    except OSError as e:
        if e.errno == 17:
            pass
        else:
            raise

    return cfg


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

# ========== Doing everything-function  ==========

def DeleteCalculateWrite(cfg, prospino_block, PID1, PID2, is_accepted, base_dir, value_dict):
    """Function for removing .dat-files from previous run, run new calculation,
    then passing the result from latest run to read-function

    Args:
        prospino_block  : pyslha-block object
        PID1            : (int) PID-code for final state particle
        PID2            : (int) PID-code for final state particle
        is_accepted     : (bool) True if point is accepted
        base_dir        : (str) base directory of point sampler
        value_dict      : (dict) containing all values to be written in slha-file
    """

    # Move to directory with Prospino executable and output files
    os.chdir(mcfg.prosp_bin_dir)

    # If output files from previous calculation exist, remove them
    try:
        os.remove('prospino.dat')
        os.remove('prospino.dat2')
        os.remove('prospino.dat3')
    except OSError as e:
        #os.chdir(base_dir)
        if e.errno == 2:
            pass
        else:
            raise
    # Run calculation (and print info about process to terminal if you want to)
    # print("{} : Process {} {} {}: {}".format(cfg.pref, PID1, PID2, value_dict['com_energy'], mcfg.cmd))
    proc, output_, timed_out = shelltimeout.shrun(mcfg.cmd, mcfg.timeout_limit, poll_interval=mcfg.poll_interval, prospino=True)

    # Move back to base dir for filereading
    os.chdir(base_dir)
    is_accepted = ReadProspinoDAT(cfg, output_, timed_out, prospino_block, PID1, PID2, is_accepted, base_dir, value_dict)

    return is_accepted


# ========== Point-by-point calculation ==========

def pointcalc(cfg, slha_path):

    is_accepted = True

    # Copy SLHA file to mcfg.prosp_bin_dir
    shutil.copy(slha_path, os.path.join(mcfg.prosp_bin_dir,'prospino.in.les_houches'))

    # Write prospino block header to SLHA file
    block_name = 'PROSPINO_OUTPUT'
    prospino_block = pyslha.Block(block_name)
    prospino_block.addComment('title', 'PID PID COM-energy [GeV] LO [fb] LO_relnumerr NLO [fb] NLO_relnumerr NLO_ms [fb] NLO_ms_scale05 [fb] NLO_ms_scale2 [fb] NLO_ms_PDFerr [fb] NLO_ms_aup [fb] NLO_ms_adn [fb]')

    # Save base directory
    base_dir = os.getcwd()

    # Prospino run command:
    cmd_prosp = './prospino.run'

    ####################################################################################
    # 
    # finalState and energy decides which processes to calculate at the chosen energies.
    #
    # We loop over all processes here instead of in Prospino
    processes = {
#                'finalState'   : [ 'gg', 'sg', 'sb', 'ss', 'tb', 'bb', 'nn', 'll' ],
#                'finalState'   : [ 'gg', 'sg', 'sb', 'ss', 'tb', 'bb'],
                'finalState'   : [ 'gg'],
#                'finalState'   : [ 'nn'],
#                'energy'       : [ '13000'],
                'energy'       : [ '7000' , '8000', '13000', '14000' ],
                 'inlo'         : [ '1', '0' ],
                'isquark1'     : [ '-4', '-3', '-2', '-1', '1', '2', '3', '4' ],
                'isquark1_int' : [ -4, -3, -2, -1, 1, 2, 3, 4 ],
                }
    ####################################################################################

    pid_codes = {
                'gluino'    : 1000021,
                '-4'        : 1000004,              # PID-codes related to squark-numbering in Prospino (and PDFs...)
                '-3'        : 1000003,
                '-2'        : 1000001,
                '-1'        : 1000002,
                '1'         : 2000002,
                '2'         : 2000001,
                '3'         : 2000003,
                '4'         : 2000004,
                '-4bar'     : -1000004,             # Minus indicating anti-particle
                '-3bar'     : -1000003,
                '-2bar'     : -1000001,
                '-1bar'     : -1000002,
                '1bar'      : -2000002,
                '2bar'      : -2000001,
                '3bar'      : -2000003,
                '4bar'      : -2000004,
                'tb1'       : 1000006,              # Lightest sbottom
                'tb2'       : 2000006,
                'bb1'       : 1000005,              # Lightest stop
                'bb2'       : 2000005,
                'tb1bar'    : -1000006,
                'tb2bar'    : -2000006,
                'bb1bar'    : -1000005,
                'bb2bar'    : -2000005,
                'nn'        : { '1'       : 1000022, # Neutralinos and charginos
                                '2'       : 1000023,
                                '3'       : 1000025,
                                '4'       : 1000035,
                                '5'       : 1000024,
                                '6'       : 1000037,
                                '7'       : -1000024,
                                '8'       : -1000037},
                'll'        : { '1'       : [1000011, -1000011],  # sel,sel         
                                '2'       : [2000011, -2000011],  # ser,ser         
                                '3'       : [1000012, -1000012],  # snel,snel       
                                '4'       : [-1000011, 1000012],  # sel+,snl        
                                '5'       : [1000011, -1000012],  # sel-,snl        
                                '6'       : [1000015, -1000015],  # stau1,stau1     
                                '7'       : [2000015, -2000015],  # stau2,stau2     
                                '8'       : [1000015, -2000015],  # stau1,stau2     
                                '9'       : [1000016, -1000016],  # sntau,sntau     
                                '10'      : [-1000015, 1000016], # stau1+,sntau    
                                '11'      : [1000015, -1000016],  # stau1-,sntau    
                                '12'      : [-2000015, 1000016],  # stau2+,sntau    
                                '13'      : [2000015, -1000016]   # stau2-,sntau
                                }
                }
    
    # Default values of Prospino args.
    isquark1   =  isquark2  = '0'
    ipart1     =  ipart2 = '1'
    icoll_in   = '1'              # Collider Tevatron [0] or LHC [1]
    inlo       = '0'              # LO only [0] or full NLO [1] calculation
    isq_ng_in  = '1'              # Degenerate [0] or free [1] squark masses
    i_error_in = '1'              # Central scale [0] or scale variation [1]

    for energy_in in processes['energy']:
        for finalState in processes['finalState']:
            is_accepted = True
            energy_in = str(energy_in)

            # Dictionary for saving numbers from prospino.dat
            # TODO: Check if this really needs to be complete? Only com_energy?
            value_dict = dict.fromkeys(['com_energy', 'LO', 'rel_LO_err', 'NLO', 'rel_NLO_err', 'NLO_ms', 'NLO_ms_at_05',
                                        'NLO_ms_at_2', 'NLO_ms_PDF', 'NLO_ms_aup', 'NLO_ms_adn', 'k_fac05', 'k_fac1', 'k_fac2', 'k_aup', 'k_adn'])
            value_dict['com_energy'] = energy_in
            
            # Treat according to process
            if finalState == 'gg':
                # Simples! Only one sub-process.                                                                       
                mcfg.cmd = [cmd_prosp, '1', isq_ng_in, icoll_in, energy_in, i_error_in, finalState, ipart1, ipart2, isquark1, isquark1 ]
                DeleteCalculateWrite(cfg, prospino_block, pid_codes['gluino'], pid_codes['gluino'], is_accepted, base_dir, value_dict)

            elif finalState == 'sg':
                # First do complete NLO calculation of gluino-squark to get k-factors (average squark mass assumption)
                mcfg.cmd = [cmd_prosp, '1', isq_ng_in, icoll_in, energy_in, i_error_in, finalState, ipart1, ipart2, '-4', '-4' ]
                DeleteCalculateWrite(cfg, prospino_block, pid_codes['gluino'], pid_codes['-4'], is_accepted, base_dir, value_dict)
                
                # Then, to save resources, do only LO calculation for each individual flavour
                for isquark1_in in processes['isquark1']:
                    if isquark1_in != '0':
                        isquark2_in = isquark1_in
                        mcfg.cmd = [cmd_prosp, inlo, isq_ng_in, icoll_in, energy_in, i_error_in, finalState, ipart1, ipart2, isquark1_in,isquark2_in ]
                        DeleteCalculateWrite(cfg, prospino_block, pid_codes['gluino'], pid_codes[isquark2_in], is_accepted, base_dir, value_dict)

            elif finalState == 'ss' or finalState == 'sb':
                # First do complete NLO calculation of squark-(anti)squark to get k-factors (average squark mass assumption)
                mcfg.cmd = [cmd_prosp, '1', isq_ng_in, icoll_in, energy_in, i_error_in, finalState, ipart1, ipart2, '-4', '-4' ]
                DeleteCalculateWrite(cfg, prospino_block, pid_codes['-4'], pid_codes['-4'], is_accepted, base_dir, value_dict)

                # Then, to save resources, do only LO calculation for each individual flavour
                # isquark1_in <=  isquark2_in to avoid double counting (these give zero xsec anyway)
                for isquark1_in in processes['isquark1_int']:
                    for isquark2_in in range(isquark1_in, processes['isquark1_int'][-1]+1):
                        if isquark1_in != 0 and isquark2_in != 0:
                            isquark1_in = str(isquark1_in)
                            isquark2_in = str(isquark2_in)
                            mcfg.cmd = [cmd_prosp, inlo, isq_ng_in, icoll_in, energy_in, i_error_in, finalState, ipart1, ipart2, isquark1_in, isquark2_in ]
                            if finalState == 'sb': isquark2_in += 'bar'
                            DeleteCalculateWrite(cfg, prospino_block, pid_codes[isquark1_in], pid_codes[isquark2_in], is_accepted, base_dir, value_dict)

            elif finalState == 'tb' or finalState == 'bb':
                for i in range(1,3):
                    mcfg.cmd = [cmd_prosp, '1', isq_ng_in, icoll_in, energy_in, i_error_in, finalState, str(i), str(i), isquark1, isquark2 ]
                    DeleteCalculateWrite(cfg, prospino_block, pid_codes[finalState+str(i)], pid_codes[finalState+str(i)+'bar'], is_accepted, base_dir, value_dict)

            elif finalState == 'nn':
                # neutralino/chargino pair combinations
                # combinations 5,5 5,6 6,6 will not be generated (same charge charginos)
                for ipart1_in in range(1,7):
                    for ipart2_in in range(ipart1_in,9):
                        print 'Starting calculation for nn ', ipart1_in, ipart2_in
                        mcfg.cmd = [cmd_prosp, '1', isq_ng_in, icoll_in, energy_in, i_error_in, finalState, str(ipart1_in), str(ipart2_in), isquark1, isquark2 ]
                        DeleteCalculateWrite(cfg, prospino_block, pid_codes['nn'][str(ipart1_in)], pid_codes['nn'][str(ipart2_in)], is_accepted, base_dir, value_dict)
                        
            elif finalState == 'll':
                # sleptons
                for ipart1_in in range(1,14):
                    mcfg.cmd = [cmd_prosp, '1', isq_ng_in, icoll_in, energy_in, i_error_in, finalState, str(ipart1_in), str(ipart2_in), isquark1, isquark2 ]
                    DeleteCalculateWrite(cfg, prospino_block, pid_codes['ll'][str(ipart1_in)][0], pid_codes['ll'][str(ipart1_in)][1], is_accepted, base_dir, value_dict)

    #############################################################################

    # Writing stuff to SLHA-file
    slha_dict = pyslha.readSLHAFile(slha_path)
    slha_dict[block_name] = prospino_block
    pyslha.writeSLHAFile(slha_path, slha_dict, precision=4)


    # Return is_accepted is actually not necessary in prospino.py, just some legacy from the old version and how point_sampler.py works...
    return is_accepted
