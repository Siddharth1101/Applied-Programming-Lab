from pylab import *
'''
x=rand(64)
X=fft(x)
y=ifft(X)
c_[x,y]
print (abs(x-y).max())

#<eg1 3>
x=linspace(0,2*pi,128)
y=sin(5*x)
Y=fft(y)
figure(1)
subplot(2,1,1)
plot(abs(Y),lw=2)
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(unwrap(angle(Y)),lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
'''
#<eg2 4>
x=linspace(0,2*pi,129);x=x[:-1]
y=sin(5*x)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure(2)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
'''
#<eg3 6>≡
t=linspace(0,2*pi,129);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure(3)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $(1+0.1cos(t))cos(10t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
'''
#<eg4 8>
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/512.0
#print(Y)
#print("size of Y=",len(Y))
#print("\n")
w=linspace(-64,64,513);w=w[:-1]
#print(w)
#print("size of w=",len(w))
figure(4)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
'''
#sin^3(t)
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(sin(t))**3
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure(5)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
xticks(arange(-10, 10, step=2))
title(r"Spectrum of $sin^3 (t)$")
grid(True)
subplot(2,1,2)
#plot(w,angle(Y),'ro',lw=2)
xticks(arange(-10, 10, step=2))
xlim([-10,10])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#cos^3(t)
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(cos(t))**3
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure(6)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
xticks(arange(-10, 10, step=2))
title(r"Spectrum of $cos^3 (t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'bo',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
xticks(arange(-10, 10, step=2))
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#cos(20t+5cos(t))
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=cos(20*t+5*cos(t))
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure(7)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-50,50])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of cos(20t+5cos(t))")
grid(True)
subplot(2,1,2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-50,50])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)

#exp(-t^2/2)
t=linspace(-8*pi,8*pi,513);t=t[:-1]
y=exp(-(t**2)/2)
Y=fftshift(fft(ifftshift(y)))/64
#Normalizing the Gaussian
w=linspace(-32,32,513);w=w[:-1]
Yactual=(1/sqrt(2*pi))*exp(-(w**2)/2)
print("Max error is {}".format(abs(Y-Yactual).max()))

figure(8)
subplot(2,1,1)
plot(w,abs(Y),lw=2,label='FFT Gaussian')
#plot(w,y,label='Original Gaussian')
xlim([-5,5])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $exp(-t^2/2)$")
legend()
grid(True)
subplot(2,1,2)
ii=where(abs(Y)>1e-3)
plot(w,angle(Y),'go',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'ro',lw=2)
xlim([-5,5])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
'''
show()