from sympy import *
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

s=symbols('s')
#defining the lowpass filter function
def lowpass(R1,R2,C1,C2,G,Vi):
    A=Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b=Matrix([0,0,0,-Vi/R1])     
    V=A.inv()*b
    return (A,b,V)

#defining the highpass filter function
def highpass(R1,R2,C1,C2,G,Vi):
    s=symbols('s')
    A1=Matrix([[0,0,1,-1/G],[-(s*R2*C2)/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-s*C2-s*C1,s*C2,0,1/R1]])
    b1=Matrix([0,0,0,-Vi*s*C1])
    V1=A1.inv()*b1
    return (A1,b1,V1)

#converting from sympy expression into numpy expression        
def convert(V_s): 
    V_s=simplify(V_s)
    n,d =fraction(V_s)
    n=poly(n,s).all_coeffs()
    d=poly(d,s).all_coeffs()
    n,d = [float(i) for i in n], [float(i) for i in d]
    Vo_sig = sp.lti(n,d)
    return Vo_sig 

A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
Ah,bh,Vh=highpass(10000,10000,1e-9,1e-9,1.586,1)
Voh=Vh[3]

#Bode plots of LPF
plt.figure(1)
ww=np.logspace(0,8,801)
ss=1j*ww
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
plt.subplot(2,1,1)
plt.loglog(ww,abs(v),lw=2)
plt.xlabel('ww')
plt.ylabel('f(ss)')
plt.grid()
plt.title('Bode plot of LPF step response')
plt.subplot(2,1,2)
hhf=lambdify(s,Voh,'numpy')
vh=hhf(ss)
plt.loglog(ww,abs(vh),lw=2)
plt.xlabel('ww')
plt.ylabel('f(ss)')
plt.title('Bode plot of HPF step response')
plt.tight_layout()
plt.grid()

#step response of LPF
plt.figure(2)
Vo_t = convert(Vo)
t,step_resp = sp.step(Vo_t,None,np.linspace(1e-5,1e-3,2001),None)
plt.plot(t,step_resp,'b-')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.title('Unit Step response for the Lowpass filter')
plt.grid(True)

#step response of HPF
plt.figure(3)
Vo_th = convert(Voh)
t,step_resph = sp.step(Vo_th,None,np.linspace(1e-5,1e-3,2001),None)
plt.plot(t,step_resph,'b-')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.title('Unit Step response for the Lowpass filter')
plt.grid(True)

#response to sum of sinusoids LPF
plt.figure(4)
Vo_t = convert(Vo)
t=np.linspace(0,1e-2,2001)
V_inp=np.sin(2000*np.pi*t)+np.cos(2e6*np.pi*t)
t,resp,svec = sp.lsim(Vo_t,V_inp,t)
plt.plot(t,resp,'b-')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.title('Response for the sum of sinusoids for the Lowpass filter')
plt.grid(True)

#response to the sum of sinusoids HPF
plt.figure(5)
Vo_th = convert(Voh)
th = np.linspace(0,1e-5,2001)
V_inp=np.sin(2000*np.pi*th)+np.cos(2e6*np.pi*th)
th,resph,svec = sp.lsim(Vo_th,V_inp,th)
plt.plot(th,resph,'b-')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.title('Response for the sum of sinusoids for the Highpass filter')
plt.grid(True)

#response to a damped sinusoid LPF
plt.figure(6)
Vo_t = convert(Vo)
t = np.linspace(0,1e-2,2001)
V_inp=np.cos(2*1e3*np.pi*t)*np.exp(-100*t) # cos(2*10^3*pi*t)e−0.05t
t,resp,svec = sp.lsim(Vo_t,V_inp,t)
plt.plot(t,resp,label='Frequency = 1kHz')
thf = np.linspace(0,1e-2,2001)
V_inphf=np.cos(2*1e5*np.pi*thf)*np.exp(-100*thf) # cos(2*10^5*pi*t)e−0.05t
thf,resphf,svec = sp.lsim(Vo_t,V_inphf,thf)
plt.plot(t,resphf,label='Frequency = 1MHz')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.legend()
plt.title('Response to a damped sinusoid for the Lowpass filter')
plt.grid(True)

#response to a damped sinusoid HPF
plt.figure(7)
Vo_th = convert(Voh)
th = np.linspace(0,0.06,4001)
V_inp=np.cos(2*1e3*np.pi*th)*np.exp(-100*th) # cos(2*10^3*pi*t)e−0.05t
th,resph,svec = sp.lsim(Vo_th,V_inp,th)
plt.plot(th,resph,label='Frequency = 1kHz')
thhf = np.linspace(0,0.006,4001)
V_inphhf=np.cos(2*1e5*np.pi*thhf)*np.exp(-100*thhf) # cos(2*10^5*pi*t)e−0.05t
thhf,resphhf,svec = sp.lsim(Vo_th,V_inphhf,thhf)
plt.plot(thhf,resphhf,label='Frequency = 1MHz')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.legend()
plt.title('Response to a damped sinusoid for the Highpass filter')
plt.grid(True)

#finding the step response of the highpass filter
plt.figure(8)
Ah,bh,Vh=highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Voh=Vh[3]
Vo_th = convert(Voh)
th,step_resph = sp.impulse(Vo_th,None,np.linspace(0.00,1e-3,2001),None)
plt.plot(th,step_resph,'b-')
plt.xlabel('Time(s)')
plt.ylabel('Vo(t)')
plt.title('Unit Step response for the Highpass filter using V_i=1/s')

plt.show()
