# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:15:50 2023

@author: gh513
"""
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5
k0 = 1
Ka1R = 100000
Ka2R = 1000
Ka3R = 1
Ka1O = 1000
Ka2O = 10
Ka3O = 1

KaRs = [1, Ka1R, Ka2R, Ka3R]
KaOs = [1, Ka1O, Ka2O, Ka3O]



c  = np.linspace(np.log(1e-5),np.log(10),100)
c= np.exp(c)



gamma = 1 + Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3
delta = 1 + Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3

k0apps = []
k0appsc = []
for i in range (len(KaRs)):
  #  print (i)
    k0apps.append((k0*(np.cumprod(KaRs)[i])/gamma)**alpha*
                  ((np.cumprod(KaOs)[i])/delta)**(1-alpha)*c**(i))

for i in range (len(KaRs)-1):
    print("i=",i)
    print (np.cumprod(KaRs)[i+1])
    print(np.cumprod(KaOs)[i])
    k0appsc.append((k0*(np.cumprod(KaRs)[i+1])/gamma)**alpha*
                  ((np.cumprod(KaOs)[i])/delta)**(1-alpha)*c**(i+1))
#print(k0appsc)

   
#print(k0apps)

# for i in range (len(KaRs)):
#     plt.plot(np.log10(c), k0apps[i],  label = f"path = {i+1}")
    
# for i in range (len(KaRs)-1):
#     plt.plot(np.log10(c), k0appsc[i], label = f"path = {i+1}c")
    
plt.plot(np.log10(c), k0apps[0],  label = "Path 1")
plt.plot(np.log10(c), k0apps[1],  label = "Path 2")

#plt.plot(np.log10(c), k0appsc[0]*3,  label = "1c scaled")

plt.plot(np.log10(c), k0apps[2],  label = "Path 3")
plt.plot(np.log10(c), k0apps[3],  label = "Path 4")

plt.plot(np.log10(c), k0appsc[0],  label = "Path 1c")
plt.plot(np.log10(c), k0appsc[1],  label = "Path 2c")
plt.plot(np.log10(c), k0appsc[2],  label = "Path 3c")

#plt.plot(np.log10(c),k0apps[0]+k0apps[1]+k0apps[2]+ k0apps[3]+k0appsc[0]+k0appsc[1]+k0appsc[2], label = "sum all")
#plt.plot(np.log10(c),k0apps[0]+k0apps[1]+k0apps[2]+ k0apps[3], label = "sum stepwise")
plt.legend(fontsize = 12)
plt.ylabel(("$k^{0}_{i,app}/k^{0}_{i}$"), fontsize = 14)
plt.xlabel ("log[Li$^{+}$]", fontsize = 14)



# def E_app(c, E0, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
#     return E0+R*T/F*np.log((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/(1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3))

# def k6_theo(c, k6, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
#     return k6*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/Ka1R/Ka2R)**-alpha*((1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3)/Ka1O)**(alpha-1)*c**2)

# def k5_theo(c, k5, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
#     return k5*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/Ka1R)**-alpha*(1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3)**(alpha-1)*c)

# def k2_theo(c, k2, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
#     return k2*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3)/Ka1R)**-alpha*((1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3)/Ka1O)**(alpha-1)*c)

# def k1_theo(c, k1, Ka1R, Ka2R, Ka3R, Ka1O, Ka2O, Ka3O):
#     return k1*(((1+Ka1R*c+Ka1R*Ka2R*c**2+Ka1R*Ka2R*Ka3R*c**3))**-alpha*((1+Ka1O*c+Ka1O*Ka2O*c**2+Ka1O*Ka2O*Ka3O*c**3))**(alpha-1))
