import scipy.signal as sp
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

#f=(np.cos(1.5*t))*(np.exp(-(0.5*t)))
F1=sp.lti([1,0.5],[1,1,2.5])
X = sp.lti([1,0.5],np.polymul([1,1,2.5],[1,0,2.25]))
H1=sp.lti(1,(1,0,2.25))
t,x=sp.impulse(X,None,np.linspace(0,50,501))

plt.figure(1)
plt.plot(t,x,'b-')
plt.title("Solution of differential equation with decay 0.5")
plt.grid('True')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

#f=(np.cos(1.5*t))*(np.exp(-(0.05*t)))
F2=sp.lti([1,0.05],[1,0.1,2.2525])
Y = sp.lti([1,0.05],np.polymul([1,0.1,2.2525],[1,0,2.25]))
H2=sp.lti(1,(1,0,2.25))
t,y=sp.impulse(Y,None,np.linspace(0,50,501))

plt.figure(2)
plt.plot(t,y,'b-')
plt.title("Solution of differential equation with decay 0.05")
plt.grid('True')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

plt.figure(3)
#defining the for loop for the variable frequencies from 1.4 to 1.6
#with decay remaining fixed at 0.05
for i in range(0,5,1):
    freq=np.around(1.4+i*0.05,2)
    p=0.0025+freq*freq
    F=sp.lti([1,0.05],[1,0.1,p])
    Z = sp.lti([1,0.05],np.polymul([1,0.1,p],[1,0,2.25]))
    t,z=sp.impulse(Z,None,np.linspace(0,50,501))
    plt.plot(t,z,label='Frequency = {}'.format(freq))
    
plt.title("Frequency varying from 1.4 to 1.6 with decay fixed at 0.05")
plt.grid('True')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.show()

#Coupled spring displacements. Using the manually calculated vales of X(s)
#and Y(s)
V = sp.lti([1,0,2],[1,0,3,0])
t,v=sp.impulse(V,None,np.linspace(0,20,201))
W = sp.lti([2],[1,0,3,0])
t,w=sp.impulse(W,None,np.linspace(0,20,201))

plt.figure(4)
plt.plot(t,v, label='Horizontal x(t)')
plt.plot(t,w,label='Vertical y(t)')
plt.title("Displacement as a function of time")
plt.grid('True')
plt.xlabel("t")
plt.ylabel("Displacement")
plt.legend()
plt.show()

#Transfer function of the given RLC circuit from the manually calculated value
plt.figure(5)
Hc=sp.lti([10**12],[1,10**8,10**12])
w,mod,phi=Hc.bode()
plt.subplot(2,1,1)
plt.title('Bode plot of magnitude')
plt.xlabel("\u03C9")
plt.ylabel("|H(j\u03A9|")
plt.semilogx(w,mod)
plt.subplot(2,1,2)
plt.title('Bode plot of phase')
plt.xlabel("\u03C9")
plt.ylabel("<H(j\u03C9)")
plt.semilogx(w,phi)
plt.tight_layout()
plt.show()

#Now, given the input sinusoidal input, finding the output by convolving it
#with the transfer function
plt.figure(6)
plt.subplot(2,1,1)
t=np.arange(0,10e-3,10e-8)
u=np.cos(1e3*t)-np.cos(1e6*t)
t,vo,svec=sp.lsim(Hc,u,t)
plt.plot(t,vo+0.5,'b-')
plt.title('Output response till 10ms')
plt.xlabel("Time")
plt.ylabel("Vout")
plt.grid('True')

'''
t1=t[0:3000]
v1=np.cos(1e3*t1)-np.cos(1e6**t1)
t1,vo,svec=sp.lsim(Hc,v1,t1)
plt.title('Output response till 30\u03BCs')
plt.xlabel("Time (\u03BCs)")
plt.ylabel("Vout")
plt.show()'''
plt.subplot(2,1,2)
t1= np.linspace(0,30e-6,100001)
input_func_6 = lambda t : np.cos(10**3 * t) - np.cos(10**6 * t)
t_7,y_7,vsec = sp.lsim(Hc,input_func_6(t1),t1)

plt.plot(t_7,y_7)
plt.title('Output response till 30\u03BCs')
plt.xlabel('Time')
plt.ylabel('Vout')
plt.grid('True')
plt.tight_layout()
plt.show()


