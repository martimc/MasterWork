####################################
#                                  #
#  Prior transformation functions  #
#                                  #
####################################

from math import log10, log, exp
from random import random
 
#
# Uniform[0:1] --> Uniform[minval:maxval]
#    
def flat_prior(r,minval,maxval):
 
   if not ((r >= 0) and (r <= 1)):
       raise Exception('first argument must be a float in the range [0,1]')
   else:
       transformed_val = minval + r*(maxval-minval)

   return transformed_val


#
# Uniform[0:1] --> LogUniform[minval:maxval]
#    
def log_prior(r,minval,maxval):
   if not ((r >= 0) and (r <= 1)):
       raise Exception('first argument must be a float in the range [0,1]')
   else:
       log_minval = log10(minval)
       log_maxval = log10(maxval)
       transformed_val = 10**(log_minval + r*(log_maxval-log_minval))

   return transformed_val


#
# Uniform[0:1] --> log_twosided[minval : maxval]
#    
def log_twosided_prior(r,minval,maxval):
    if not ((r >= 0) and (r <= 1)):
        raise Exception('first argument must be a float in the range [0,1]')
    else:
        log_minval = log10(minval)
        log_maxval = log10(maxval)
        transformed_val = 10**(log_minval + r*(log_maxval-log_minval))

    # Switch sign with 50% prob.
    if random() <= 0.5:
    	transformed_val = -transformed_val    

    return transformed_val



#
# Uniform[0:1] --> log_and_flat[minval : flat_end : maxval]
#    
def log_and_flat_prior(r,x1,x2,x3):
  
  x1 = float(x1)
  x2 = float(x2)
  x3 = float(x3)
  
  if not ((r >= 0) and (r <= 1)):
    raise Exception('first argument must be a float in the range [0,1]')

  # Useful quantities
  C   = 1./( (x2-x1) + x2*log(abs(x3/x2)) )  # Normalization factor 
  P12 = C*(x2-x1)                            # Prob. of x in (x1,x2) (flat region)
  P23 = C*x2*log(abs(x3/x2))                 # Prob. of x in (x2,x3) (~1/x region)

  # Transformation, based on: P(r<r') = r' = P(x<x') 
  if r <= P12:
    x = x1 + r/C
  
  elif (r > P12) and (r <= (P12+P23)):
    x = x2*exp( (1./(C*x2)) * (r - C*(x2-x1)) )

  else:
    raise Exception('Problem transforming r-value %f' % r)
    
  return x




#
# Uniform[0:1] --> log_and_flat[minval : flat_start : flat_end : maxval]
#    
def log_and_flat_twosided_prior(r,x0,x1,x2,x3):
  
  x0 = float(x0)
  x1 = float(x1)
  x2 = float(x2)
  x3 = float(x3)
  
  if not ((r >= 0) and (r <= 1)):
    raise Exception('first argument must be a float in the range [0,1]')

  # Useful quantities
  C   = 1. / ( (x2-x1)/x2 + log( abs((x3*x0)/(x2*x1)) ) )  # Normalization factor 
  P01 = C * log( abs(x0/x1) )                              # Prob. of x in (x0,x1)
  P12 = C * (x2-x1)/x2                                     # Prob. of x in (x1,x2)
  P23 = C * log( abs(x3/x2) )                              # Prob. of x in (x2,x3)

  # Transformation, based on: P(r<r') = r' = P(x<x') 
  if r <= P01:
    x = x0 * exp(-r/C)
  
  elif (r > P01) and (r < (P01+P12)):
    x = x1 + x2*((r-P01)/C)
  
  elif (r > (P01+P12)) and (r <= (P01+P12+P23)):
    x = x2*exp( (r-(P01+P12))/C )

  else:
    raise Exception('Problem transforming r-value %f' % r)
    
  return x
  
    

#
# Uniform[0:1] --> {-1, +1}
#    
def plus_minus_prior(r):
  
  if not ((r >= 0) and (r <= 1)):
    raise Exception('first argument must be a float in the range [0,1]')

  # Transformation to discrete set {-1,+1}
  if r <= 0.5:
    x = -1

  elif r > 0.5:
    x = 1
  
  else:
    raise Exception('Problem transforming r-value %f' % r)
    
  return x
