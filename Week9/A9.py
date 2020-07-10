from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#<eg 1 2>
t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=sin(sqrt(2)*t)
y[0]=0          # the sample corresponding to -tmax should be set zero
y=fftshift(y)   # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2} t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#<eg 2 4>
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
# y=sin(sqrt(2)*t)
figure(2)
plot(t1,sin(sqrt(2)*t1),'b',lw=2)
plot(t2,sin(sqrt(2)*t2),'r',lw=2)
plot(t3,sin(sqrt(2)*t3),'r',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$")
grid(True)

#<eg 3 5>
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
y=sin(sqrt(2)*t1)
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
grid(True)
show()

#<eg 4 7>
t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=t
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure(4)
semilogx(abs(w),20*log10(abs(Y)),lw=2)
xlim([1,10])
ylim([-20,0])
xticks([1,2,5,10],["1","2","5","10"],size=16)
ylabel(r"$|Y|$ (dB)",size=16)
title(r"Spectrum of a digital ramp")
xlabel(r"$\omega$",size=16)
grid(True)

#<eg 5 8>
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t1)*wnd
figure(5)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
grid(True)

#<eg 6 10>
t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t)*wnd
y[0]=0              # the sample corresponding to -tmax should be set zeroo
y=fftshift(y)       # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
figure(6)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#<eg7 12>
t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
wnd=fftshift(0.54+0.46*cos(2*pi*n/256))
y=sin(sqrt(2)*t)
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure(7)
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

# co^3(wot). Without window
t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
wo=0.86
y=(cos(wo*t))*(cos(wo*t))*(cos(wo*t))
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure(8)
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^{3}(\omega_{o} t)$ without window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

# co^3(wot). With window
#t=linspace(-4*pi,4*pi,257);t=t[:-1]
#dt=t[1]-t[0];fmax=1/dt
n=arange(256)
wnd=fftshift(0.54+0.46*cos(2*pi*n/256))
#wo=0.86
y=(cos(wo*t))*(cos(wo*t))*(cos(wo*t))
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
#w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]
figure(9)
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^{3}(\omega_{o} t)$ with window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

#Question 3
t=linspace(-pi,pi,129);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(128)
wnd=fftshift(0.54+0.46*cos(2*pi*n/128))
wo=1.5
d=0.5
y=cos(wo*t+d)
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/128.0
w=linspace(-pi*fmax,pi*fmax,129);w=w[:-1]
figure(10)
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(\omega_o t+\delta)$ with $\omega_o$={}, $\delta$={}".format(wo,d))
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

def est_omega(w,Y):
    ii = where(w>0)
    omega = (sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2))#weighted average
    print ("Omega = ",omega)

def est_delta(w,Y,sup = 1e-4,window = 1):
    ii_1=np.where(np.logical_and(np.abs(Y)>sup, w>0))[0]
    np.sort(ii_1)
    points=ii_1[1:window+1]
    print ("Delta = ", np.sum(np.angle(Y[points]))/len(points))#weighted average for first 2 points

est_omega(w,Y)
est_delta(w,Y)

#Question 4. With added white Gaussian noise
t=linspace(-pi,pi,129);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(128)
wnd=fftshift(0.54+0.46*cos(2*pi*n/128))
wo=1.5
d=0.5
y=cos(wo*t+d)
y=y*wnd+0.1*randn(128)
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/128.0
w=linspace(-pi*fmax,pi*fmax,129);w=w[:-1]
figure(11)
subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(\omega_o t+\delta)$ with $\omega_o$={}, $\delta$={} with added Gaussian noise".format(wo,d))
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

est_omega(w,Y)
est_delta(w,Y)
show()

#Question 5. With and without windowing
t=linspace(-pi,pi,1025);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y1=cos(16*(1.5+t/(2*pi))*t)
n=arange(1024)
wnd=fftshift(0.54+0.46*cos(2*pi*n/1024))
y=y1*wnd
y1[0]=0; y[0]=0
y1=fftshift(y1); y=fftshift(y)
Y1=fftshift(fft(y1))/1024.0; Y=fftshift(fft(y))/1024.0
w=linspace(-pi*fmax,pi*fmax,1025);w=w[:-1]
ph=angle(Y); ph1=angle(Y1)
mag = abs(Y); mag1 = abs(Y1)
ph[where(mag<3e-3)] = 0; ph1[where(mag1<3e-3)] = 0
figure(12)
subplot(2,1,1)
plot(w,abs(Y1),'b',lw=2)
xlim([-75,75])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of chirp function without window")
grid(True)
subplot(2,1,2)
plot(w,angle(Y1),'ro',lw=2)
xlim([-75,75])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
figure(13)
subplot(2,1,1)
plot(w,abs(Y),'b',lw=2)
xlim([-75,75])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of chirp function with window")
grid(True)
subplot(2,1,2)
plot(w,ph,'ro',lw=2)
xlim([-75,75])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#Question 6
t_arrays=split(t,16)

Y_mags=zeros((16,64))
Y_angles=zeros((16,64))
#splitting array and doing fft
for i in range(len(t_arrays)):
    t = t_arrays[i]
    dt=t[1]-t[0];fmax=1/dt
    y=cos(16*(1.5+t/(2*pi))*t)
    y[0]=0
    y=fftshift(y)
    Y=fftshift(fft(y))/64.0
    w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]
    Y_mags[i] =  abs(Y)
    Y_angles[i] =  angle(Y)
#plotting
fig = figure(14)
ax = fig.add_subplot(111, projection='3d')

t=np.linspace(-pi,pi,1025);t=t[:-1]
fmax = 1/(t[1]-t[0])
t=t[::64]
w=linspace(-fmax*pi,fmax*pi,65);w=w[:-1]
t,w=np.meshgrid(t,w)

surf=ax.plot_surface(w,t,Y_mags.T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ylabel("Time")
xlabel("Frequency")
title("Surface time-frequency plot of the Chirp function")

show()

