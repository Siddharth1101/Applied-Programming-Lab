import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as sp
import math 

#Question 1
def exp(t):
    return(np.exp((t)%(2*np.pi)))

def coscos(t):
    return(np.cos(np.cos((t)%(2*np.pi))))

plt.figure(1)
t=np.linspace(-2*np.pi,4*np.pi,2000,endpoint=False)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.plot(t,coscos(t))
plt.title('cos(cos(t)) from -2\u03C0 to 4\u03C0')
plt.grid()

plt.figure(2)
plt.semilogy(t,exp(t))
plt.title('Semilog exp(t), Periodic extension from -2\u03C0 to 4\u03C0')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()

#Question 2
ae=np.zeros(26)
ac=np.zeros(26)
be=np.zeros(26)
bc=np.zeros(26)
ue = lambda t,k: exp(t)*np.cos(k*t)
ve = lambda t,k: exp(t)*np.sin(k*t)
uc = lambda t,k: coscos(t)*np.cos(k*t)
vc = lambda t,k: coscos(t)*np.sin(k*t)

#Question 3
for k in range (0,25):
    if k==0:
        ae[k]=(sp.quad(ue,0,2*np.pi,args=(k)))[0]/(2*np.pi)
        ac[k]=(sp.quad(uc,0,2*np.pi,args=(k)))[0]/(2*np.pi)
    else:
        ae[k]=(sp.quad(ue,0,2*np.pi,args=(k)))[0]/np.pi
        ac[k]=(sp.quad(uc,0,2*np.pi,args=(k)))[0]/np.pi
    be[k]=(sp.quad(ve,0,2*np.pi,args=(k)))[0]/np.pi
    bc[k]=(sp.quad(vc,0,2*np.pi,args=(k)))[0]/np.pi
ce_fs = [ae[0]]
cc_fs = [ac[0]]
for i in range(1,26): 
    ce_fs.append(ae[i])
    ce_fs.append(be[i])  
    cc_fs.append(ac[i])
    cc_fs.append(bc[i]) 

plt.figure(3)
plt.semilogy(abs(np.array(ce_fs)),'ro')
plt.title('Fourier coefficients of exp(x) from integration Semilogy')
plt.xlabel('n')
plt.ylabel('Magnitude')
#plt.label()
plt.grid()

plt.figure(4)
plt.loglog(abs(np.array(ce_fs)),'ro')
plt.title('Fourier coefficients of exp(x) from integration Loglog')
plt.xlabel('n')
plt.ylabel('Magnitude')
#plt.legend()
plt.grid()
     
plt.figure(5)
plt.semilogy(abs(np.array(cc_fs)),'ro')
plt.title('Fourier coefficients of coscos(x) from integration Semilogy')
plt.xlabel('n')
plt.ylabel('Magnitude')
#plt.legend()
plt.grid()

plt.figure(6)
plt.loglog(abs(np.array(cc_fs)),'ro')
plt.title('Fourier coefficients of coscos(x) from integration Loglog')
plt.xlabel('n')
plt.ylabel('Magnitude')
#plt.legend()
plt.grid()

#Question 4
x=np.linspace(0,2*np.pi,401)
expo=[]
cosc=[]
for n in range(26):
    expo.append(ae[0]*np.cos(n*x[0])+be[0]*np.sin(n*x[0]))
    cosc.append(ac[0]*np.cos(n*x[0])+bc[0]*np.sin(n*x[0]))
for i in range(1,400):
    for n in range(26):
        expo.append(ae[i]*np.cos(n*x[i])+be[i]*np.sin(n*x[i]))
        cosc.append(ac[i]*np.cos(n*x[i])+bc[i]*np.sin(n*x[i]))
x=x[:-1]                # drop last term to have a proper periodic integral
b=expo                   # f has been written to take a vector
A=np.zeros((400,51))    # allocate space for A
A[:,0]=1                # col 1 is all ones
for k in range(1,26):
    A[:,2*k-1]=np.cos(k*x)  # cos(kx) column
    A[:,2*k]=np.sin(k*x)    # sin(kx) column

#Question 5
c1=np.linalg.lstsq(A,b)[0] # the ’[0]’ is to pull out the best fit vector. lstsq returns a list.


plt.show()