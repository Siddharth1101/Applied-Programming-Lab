import numpy as np
from pylab import *
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

with open("h.csv") as file: # Use file to refer to the file object
   data = file.read()
data=np.fromstring(data, sep='\n')

w, h = signal.freqz(data)

fig = plt.figure()

plt.subplot(2,1,1)
plt.title('Digital filter frequency response')
plt.plot(w, 20*np.log10(abs(h)), 'b')
plt.ylabel('Magnitude', color='b')
plt.xlabel('Frequency [rad/sample]')
plt.grid()

plt.subplot(2,1,2)
angles = np.unwrap(np.angle(h))
#angles=np.pi/180*angles
plt.plot(w, angles, 'g')
plt.ylabel('Angle in radians(unwrap)', color='g')
plt.xlabel('Frequency [rad/sample]')

plt.grid()
plt.axis('tight')
fig.tight_layout()

n=np.arange(1,2**10+1,1)
x=np.cos(0.2*np.pi*n)+np.cos(0.85*np.pi*n)
plt.figure()
plt.plot(n,x)
plt.title("Plot of cos(0.2$\pi$n)+cos(0.85$\pi$n)")
plt.xlabel("n")
plt.ylabel("f(n)")
plt.grid()
plt.xlim([0,200])

#Linear convolution using direct summation
y=convolve(x, data)

plt.figure()
plt.plot(range(len(n)+len(data)-1), y)
plt.title("Linear convolution using direct summation")
plt.xlabel("n")
plt.ylabel("f(n)")
plt.grid()
plt.xlim([0,200])

#Circular convolution using DFTs
x_=np.concatenate((x, zeros(len(data)-1)))
y1=ifft(fft(x_)*fft(concatenate((data,zeros(len(x_)-len(data))))))

plt.figure()
plt.plot(range(len(y1)), y1)
plt.title("Output of circular convolution")
plt.xlabel("n")
plt.ylabel("f(n)")
plt.grid()
plt.xlim([0,200])

#Linear convolution using circular convolution
def lin_circular_conv(x,data):
    P = len(data)
    n_ = int(ceil(log2(P)))
    data_ = np.concatenate((data,np.zeros(int(2**n_)-P)))
    P = len(data_)
    n1 = int(ceil(len(x)/2**n_))
    x_ = np.concatenate((x,np.zeros(n1*(int(2**n_))-len(x))))
    y = np.zeros(len(x_)+len(data_)-1)
    for i in range(n1):
        temp = np.concatenate((x_[i*P:(i+1)*P],np.zeros(P-1)))
        y[i*P:(i+1)*P+P-1] += ifft(fft(temp) * fft( np.concatenate((data_,np.zeros(len(temp)-len(data_)))))).real
    return y

y2 = lin_circular_conv(x,data)
len(y2)

plt.figure()
plt.plot(range(len(y2)), y2)
plt.title("Linear convolution using circular convolution")
plt.xlabel("n")
plt.ylabel("f(n)")
plt.grid()
plt.xlim([0,200])

#Circular correlation of Zadoff-Chu sequence
with open("x1.csv") as file2:
    data2=file2.readlines()
    data2=asarray([complex(i[:-1].replace('i','j')) for i in data2], dtype = 'complex')    

Dshifted=np.roll(data2, 5)
corr=ifft(conj(fft(data2))*(fft(Dshifted)))

plt.figure()
plt.title("Correlation of cyclically shifted Zadoff Chu sequence")
plt.stem(range(len(corr)), abs(corr))
plt.xlabel("n")
plt.ylabel("f(n)")
plt.grid()
xlim([0,30])

plt.show()



