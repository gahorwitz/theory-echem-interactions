# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:13:19 2023

@author: gh513
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

import os.path, glob
import scipy.optimize

from datetime import datetime

from autograd import numpy as np
from autograd import elementwise_grad, value_and_grad, hessian
from scipy.optimize import minimize

print ("Test")
def E_app(c, E0, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
    return E0+R*T/F*np.log((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/(1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3))

def k6_theo(c, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
    return k6*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/Ka1R/Ka2R)**-alpha*((1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3)/Ka1O)**(alpha-1)*c**2)

def k5_theo(c, k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
    return k5*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/Ka1R)**-alpha*(1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3)**(alpha-1)*c)

def k2_theo(c, k2, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
    return k2*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/Ka1R)**-alpha*((1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3)/Ka1O)**(alpha-1)*c)

def k1_theo(c, k1, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
    return k1*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3))**-alpha*((1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3))**(alpha-1))
    
def loss(x0,df):
    value = 0
    #E0, k1,k2,k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O = x0 
    #E0, k2,k5 , Ka1R, Ka2R, Ka3R, Ka1O = x0 
  #  E0, k1, k2, k5, k6, Ka1R, Ka2R, Ka3R, Ka1O = x0 
   # E0, k1, k2, k5, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O = x0
    E0, k1, k2, k5, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O = x0
   # k1, k2, k5, k6= x0 
    # Ka2O = 0
    # Ka3O = 0
    
    if Ka1O < 0:
        value = 1000     
    if Ka3R < 0:
        value = 1000
    if Ka1R < 0:
        value = 1000
    if Ka2R < 0:
        value = 1000
    if Ka2O < 0:
        value = 1000
    if Ka3O < 0:
        value = 1000

    if Ka1R < Ka1O:
        value = 1000
    if Ka1R < Ka2R:
        value = 1000
    if Ka2R < Ka3R:
        value = 1000
    if Ka1O < Ka2O:
        value = 1000
    if Ka2R < Ka2O:
        value = 1000
    if Ka3R < Ka3O:
        value = 1000
    if Ka2O < Ka3O:
        value = 1000
        
    

    if k2 < 0:
        value = 1000
    if k5 < 0:
         value = 1000
    if k1 < 0:
         value = 1000
    if k6 < 0:
         value = 1000

    
    E_fit  = E_app(df["c_LiTFSI (M)"],E0, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
    k_fit  = (k5_theo(df["c_LiTFSI (M)"],k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
              +k2_theo(df["c_LiTFSI (M)"], k2, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
                  +k6_theo( df["c_LiTFSI (M)"], k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
                      +k1_theo( df["c_LiTFSI (M)"], k1, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O))
    #value = value + np.sum(np.abs((E_fit-df["E0"])/df["E0"])+((np.abs(k_fit-df["ks"]))*5))
    value_k = np.sum(np.abs((k_fit-df["ks"]))**2*10)
    value_E = np.sum(np.abs((E_fit-df["E0"])/df["E0"])**2)
   # print (value_k, value_E)
    value = value + np.sum(value_k + value_E)
  #  value = value + np.sum((np.abs(k_fit-df["ks"]))**2)
  #  value = value + np.sum((np.abs(k_fit-df["ks"])/df["ks"]))
 
    return value

def takestep(x):
    """ This function determines how the parameters will be adjusted at each 
    basinhopping optimiser step. It allows us to specify different step sizes 
    for each of the three parameters."""
    stepsize=[0.01,0.01,0.01,0.01,0.01, 1,1,1,1,1,1]   # Maximum amount to step in one go.
  #  stepsize=[0.001,0.01,0.01,0.001]   # Maximum amount to step in one go.
    dx = [np.random.uniform(low=-stepsize[i],high=stepsize[i]) for i in range(len(x0))]
    return x+dx


def main():
    """ Main program """
    start_time = datetime.now()
    print(start_time)
    
    """ Get the experimental data"""
    
    file    = path
    df = pd.read_excel(file)
    c  = np.linspace(np.log(1e-2),np.log(2.5),100)
    c= np.exp(c)
  #  loss(df, E0, k1,k2,k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
    bnds = ((-0.6,-0.5), (0, None), (0,None), (0,None), (0,None), (0,None), (0,None))
    LA = scipy.optimize.basinhopping(loss, x0,  T=0.1, niter= 0, take_step=takestep,
                                       minimizer_kwargs= {"args":(df), "method":"Nelder-Mead",
                                       "bounds":bnds})
    
  #  E0, k1, k2, k5, k6 ,Ka1R, Ka2R, Ka3R, Ka1O = LA.x
  #  E0, k1, k2, k5, k6 ,Ka1R, Ka2R, Ka3R, Ka1O, Ka2O = LA.x
    E0, k1, k2, k5, k6 ,Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O = LA.x
  #  k1, k2, k5, k6 = LA.x
  #  k1 = 2.230405e-09
  #  k2 = 6.064541e-02
  #  k5 = 5.671643e-03
  #  k6 = 1.870149e-03
    
    print(LA)
    LA_df = pd.DataFrame(LA.x)
    LA_df = pd.DataFrame.transpose(LA_df)
  #  LA_df.columns = ["E0", "k2","k5" , "Ka1R", "Ka2R", "Ka3R", "Ka1O"]
  #  LA_df.columns = ["E0", "k1", "k2","k5","k6", "Ka1R", "Ka2R", "Ka3R", "Ka1O"]
  #  LA_df.columns = ["E0", "k1", "k2","k5","k6", "Ka1R", "Ka2R", "Ka3R", "Ka1O", "Ka2O"]
    LA_df.columns = ["E0", "k1", "k2","k5","k6", "Ka1R", "Ka2R", "Ka3R", "Ka1O", "Ka2O", "Ka3O"]
 #   LA_df.columns = ["k1", "k2","k5","k6"]
    LA_df = pd.DataFrame.transpose(LA_df)
    
  #   print(LA)
    
    print(LA_df)
    
    fig, ax = plt.subplots(3,1, figsize = (5,10))
    ax[0].plot(np.log(df["c_LiTFSI (M)"]), df["E0"], "o" )
    ax[0].plot(np.log(c), E_app(c, E0, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O))
    ax[0].set_ylabel(r'$E^{0}_{app}$', fontsize = 16) 
    ax[0].set_xlabel ("log[Li$^{+}$]", fontsize = 16)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
   
    
           
    ax[1].plot(np.log(df["c_LiTFSI (M)"]), df["ks"], "o" )
    ax[1].plot(np.log(c), k5_theo(c, k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
        +k2_theo(c, k2, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
           +k6_theo(c, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)
               +k1_theo(c, k1, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O))
    ax[1].set_ylabel(("k$^{0}_{app}$"), fontsize = 16)
    ax[1].set_xlabel ("log[Li$^{+}$]", fontsize = 16)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
               
    ax[2].plot(np.log(c), k1_theo(c, k1, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O), label = "k1")
    ax[2].plot(np.log(c), k2_theo(c, k2, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O), label = "k2")
    ax[2].plot(np.log(c), k5_theo(c, k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O), label = "k5")
    ax[2].plot(np.log(c), k6_theo(c, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O), label = "k6")
    ax[2].legend()
    fig.tight_layout()    
   # k2 = np.array(np.linspace(0.001,0.05,10))

    # for k in k2:
    #     x1 = [E0, k, Ka1R, Ka2R, Ka3R, Ka1O]
    #     ax[2].plot(np.log(df["c_LiTFSI (M)"]), df["ks"], "o" )
    #     ax[2].plot(np.log(c), k5_theo(c, k, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)+k2_theo(c, k, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O))
    #     print(k, loss(x1, df))
    
 #   k=0.04
 #   x1 = [E0, k, Ka1R, Ka2R, Ka3R, Ka1O]
   # ax[2].plot(np.log(df["c_LiTFSI (M)"]), df["ks"], "o" )
 ##   ax[1].plot(np.log(c), k5_theo(c, k, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)+k2_theo(c, k, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O))
 #   ax[2].plot(np.log(df["c_LiTFSI (M)"]), df["ks"], "o" )
 #   ax[2].plot(np.log(c), k5_theo(c, k, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O)+k2_theo(c, k, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O))
    plt.savefig("DMSO exp Fit.pdf")
    plt.savefig("DMSO exp Fit.png")
    print('Duration: {}'.format(datetime.now() - start_time))
    
#This is because Anaconda writes a lot of temporary files, so this deletes them    
    for file in glob.glob("tmp*"):
        os.remove(file)
        
if __name__ == "__main__":
    
    #The file needs to be in .xls (.xlsx takes much longer). 
    path    = "C:/Users/gh513/Dropbox/Postdoc Cambridge/Other people works/Vera/Raw data/DMSO/DMSO_Exp_param.xls"
   
    #physical constants
    R = 8.3144598 # J⋅mol^−1⋅K^−1.
    T = 298    # K
    F = 96485 # C/mol
    
    alpha = 0.5
    
    #### Initial guesses of parameters
    E0   = -0.555084
    k1   = 0.0001
    k2   = 0.05
    k5   = 0.05
    k6   = 0.00001
    Ka1R = 4
    Ka2R = 4
    Ka3R = 4
    Ka1O = 2
    Ka2O = 0.01
    Ka3O = 0.01
   
   # x0 = [E0, k1,k2,k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O]
   # x0 = [E0, k2,k5, Ka1R, Ka2R, Ka3R, Ka1O]
   # x0 = [E0, k1, k2, k5, k6, Ka1R, Ka2R, Ka3R, Ka1O]
  #  x0 = [E0, k1, k2, k5, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O] 
    x0 = [E0, k1, k2, k5, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O] 
 #   x0 = [k1, k2, k5, k6]
    #Call main program
    main()
