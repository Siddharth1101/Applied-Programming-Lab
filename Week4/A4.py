import numpy  as np
import matplotlib.pyplot as plt
import scipy.integrate

#Part 1. Defining the functions to be read and returned
def exp(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))

def plotter(fig_no,plot_x,plot_y,label_x,label_y,type=None,kind='b-',title=''):
    plt.figure(fig_no)
    plt.grid(True)
    if type =="semilogy":
        plt.semilogy(plot_x,plot_y,kind)
    elif type =='ll':
        plt.loglog(plot_x,plot_y,kind)
    elif type ==None:
        plt.plot(plot_x,plot_y,kind)
    plt.xlabel(label_x,size =18)
    plt.ylabel(label_y,size =18)
    plt.title(title)
    
#Part2. Defining the additional functions needed for the Fourier series
def u1(x,k):
    return(coscos(x)*np.cos(k*x))

def v1(x,k):
    return(coscos(x)*np.sin(k*x))

def u2(x,k):
    return(exp(x)*np.cos(k*x))

def v2(x,k):
    return(exp(x)*np.sin(k*x))

#Defining the Fourier coefficient integrating function
def integrate():
    a = np.zeros(51)
    b = np.zeros(51)
    a[0] =  scipy.integrate.quad(exp,0,2*np.pi)[0]/(2*np.pi)
    b[0] =  scipy.integrate.quad(coscos,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,51,1):
        a[i] = scipy.integrate.quad(u2,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
        b[i] = scipy.integrate.quad(u1,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
        #a[i+1] = scipy.integrate.quad(v2,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
        #b[i+1] = scipy.integrate.quad(v1,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
    return a,b

 
#Part3. Plotting all the semilogy and loglog plots

t = np.linspace(-2*np.pi,4*np.pi,1200)
fr_length = np.arange(1,52) #Fourier Coefficient Number
plotter(1,t,exp(t),r"$t$",r"exp(t)","semilogy",title ="Plotting an exponential semilog plot")
plotter(2,t,coscos(t),r"$t$",r"cos(cos(t))","semilogy",title="The cos(cos(.)) function on a log plot")
plotter(1,t,np.concatenate((exp(t)[400:800],exp(t)[400:800],exp(t)[400:800])),r"$t$",r"exp(t)","semilogy",'r-')
plt.legend(("The actual function","periodic extension"))
plotter(2,t,np.concatenate((coscos(t)[400:800],coscos(t)[400:800],coscos(t)[400:800])),r"$t$",r"cos(cos(t)","semilogy",'r-')
plt.legend(("The actual function","Its periodic extension"))
frexp,frcos = integrate()
plotter(3,fr_length,np.absolute(frexp),"n","Magnitude","semilogy",'ro',title="Semilog Fourier Coefficients for exp(t)")
plotter(4,fr_length,np.absolute(frexp),"n","Magnitude","ll",'ro',title="Loglog Fourier Coefficients for exp(t)")
plotter(5,fr_length,np.absolute(frcos),"n","Magnitude","semilogy",'ro',title="Semilog Fourier Coefficients for cos(cos(t))")
plotter(6,fr_length,np.absolute(frcos),"n","Magnitude","ll",'ro',title="Loglog Fourier Coefficients for cos(cos(t))")

#Parts 4&5

x =np.linspace(0,2*np.pi,400)#,endpoint =True)
bexp = exp(x)
bcoscos =coscos(x)
A = np.zeros((400,51))
A[:,0] =1
for k in range(1,26):
    A[:,2*k-1] = np.cos(k*x)
    A[:,2*k] = np.sin(k*x)
cexp = np.linalg.lstsq(A,bexp,rcond=None)[0]
ccoscos = np.linalg.lstsq(A,bcoscos,rcond=None)[0]
print('\nThe length of Coefficient matrix of:\n1. Least Squares: exponential is {}, cos(cos(.)) is {}\n2. Integration: exponential is {}, cos(cos(.)) is {}'.format(len(cexp),len(ccoscos),len(frexp),len(frcos)),"\n")
plotter(3,fr_length,np.abs(cexp),"n","Magnitude","semilogy",'go',title="Semilog Fourier Coefficients for exp(t)")
plt.legend(("Value from integration","Value from lstsq"))
plotter(4,fr_length,np.abs(cexp),"n","Magnitude","ll",'go',title="Loglog Fourier Coefficients for exp(t)")
plt.legend(("Value from integration","Value from lstsq"))
plotter(5,fr_length,np.abs(ccoscos),"n","Magnitude","semilogy",'go',title="Semilog Fourier Coefficients for cos(cos(t))")
plt.legend(("Value from integration","Value from lstsq"))
plotter(6,fr_length,np.abs(ccoscos),"n","Magnitude","ll",'go',title="Loglog Fourier Coefficients for cos(cos(t))")
plt.legend(("Value from integration","Value from lstsq"))

#Part6
diffexp = np.absolute(cexp-frexp)
diffcos = np.absolute(ccoscos-frcos)
print('The maximum difference in exponential is {} and cos(cos(.)) is {}'.format(np.amax(diffexp),np.amax(diffcos)))
#print(np.amax(diffexp),np.amax(diffcos)
print('\n')
print(np.around(diffexp,5),'\n',np.around(diffcos,20),'\n')

#Part7. Plotting the fourier coefficients in the same figures

Acexp = A@cexp
Accos = A@ccoscos
plotter(1,t,np.concatenate((np.zeros(400),Acexp,np.zeros(400))),r"$t$",r"exp(t)","semilogy",'go')
plt.legend(("The actual graph","Its periodic extension","The Fourier series coefficients predicted"))
plotter(2,t,np.concatenate((np.zeros(400),Accos,np.zeros(400))),r"$t$",r"coscos(t)",None,'go')
plt.legend(("The actual graph","Its periodic extension","The Fourier series coefficients predicted"))

plt.show()