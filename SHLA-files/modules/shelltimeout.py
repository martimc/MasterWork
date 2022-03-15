#####################
#                   #
#  shelltimeout.py  #
#                   #
#####################

#  
# Simple module for executing a command
# or executable from a shell, with a given
# time limit.
#

import subprocess32
import time, os
import tempfile

def shrun(cmd, timeout_limit, use_exec=False, poll_interval=0.5, return_duration=False, prospino=False):

    # Should we use the 'exec' command to substitute the shell process with the 
    # command specified by the user? (This means that if we have to terminate the process
    # we will kill the process started by 'cmd', not only the shell it is called from.)
    if use_exec:
        use_cmd = 'exec ' + cmd
    else:
        use_cmd = cmd

    # Start process
    output_tmpfile = tempfile.TemporaryFile()
    if prospino:
        proc = subprocess32.Popen(use_cmd, stdout=subprocess32.PIPE, stderr=subprocess32.PIPE, shell=False)
        output_content, err = proc.communicate()
        #print(output_content)
    else:
        proc = subprocess32.Popen(use_cmd, stdout=output_tmpfile, stderr=output_tmpfile, shell=True)
        #output_content, err = proc.communicate()
        #print(output_content)
    
    # Start timer
    time_start = time.time()
    timed_out = False

    # Poll process every poll_interval seconds
    while proc.poll() is None:
    
        if (time.time() - time_start) > timeout_limit:
            timed_out = True
            proc.kill()
            proc.wait()
            break
        else:
            time.sleep(poll_interval)

    # Stop timer
    time_stop = time.time()

    if not prospino:
        # Return results
        output_tmpfile.seek(0)
        output_content = output_tmpfile.read()
        output_tmpfile.close()

    if return_duration:
        duration = time_stop - time_start
        return proc, output_content, timed_out, duration
    else:
        return proc, output_content, timed_out

