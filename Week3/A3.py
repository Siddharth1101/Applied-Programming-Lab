import numpy as np 
from pylab import *
import scipy.special as sp 

def g(t,A,B):
    y=A*sp.jn(2,t)+B*t  #defining the function to be returned
    return(y)   
def stddev(data,t,A0,B0):
    return math.sqrt(sum([(data[i]-g(t[i],A0,B0))**2 for i in range(0,len(data))])/len(data)) 

#Part2 Defining the function and loading data
'''data=loadtxt('fitting.dat')
t=data[:,0]
val=data[:,1:]'''
data=[np.loadtxt('fitting.dat',usecols=(i)) for i in range(0,10)]
t=np.array(data[0])
A0=1.05
B0=-0.105
figure('fig1')
[plot(t,data[i],label=r'$\sigma=$' + str(round(stddev(data[i],t,A0,B0),3))) for i in range(1,len(data))]    
true_value = g(t,A0,B0)
plot(t,true_value,label='True Value')
grid(True);legend(loc='upper right');xlabel('Time t');ylabel('f(t)')
show()

'''true_fn=g(t,1.05,-0.105)
val=c_[val,true_fn]


scl=logspace(-1,-3,9)    #creating noise
#for i in range(0,len(scl)):
#    scl[i]=round(scl[i],3)




#Part3 Fitting the data and displaying the result

figure('Figure 1')
title(r'Plot of the data')
true_value=g(t,A0,B0)
[plot(t,data[i],label=r'$\sigma=$' + str(scl[i])) for i in range(1,len(data))]    
plot(t,true_value,label='True Value')
grid(True); legend(loc='upper right'); xlabel('Time t'); ylabel('f(t)')
show()
#val=c_[val,true_fn]
#print(t)
#print(val)

"""#Part3&4 



title('Plot of the data')






#Part5 Error bars'''