# Numerical Laplace Equation Solution using Finite Difference Method
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
#from scipy.linalg import lstsq
#import matplotlib.pyplot as plt
import sys

# Set maximum iteration
Niter=1500 #number of iterations to perform

# Set Dimension and delta
Nx = Ny = 25 #we set it rectangular
delta = 1

# Boundary condition
Tbottom=0
Trod=1.0
Tguess=0.35

#colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Initiazlizing φ to a 25 by 25 zero matrix
phi = np.zeros((Ny, Nx))

# Set meshgrid
y = np.linspace(-0.5,0.5,Ny)
x = np.linspace(-0.5,0.5,Nx)
Y,X = np.meshgrid(y,x)
#rody,rodx=where(x**2+y**2<=64)
#plt.plot(rody, rodx, 'or')
#plt.plot(Y, X, marker=',', color='k', linestyle='none')
ii=where(X**2+Y**2<=0.35*0.35)
phi[ii]=1.0
figure(1)
contourf(x,y,phi,cmap=cm.coolwarm)
plt.plot(X, Y, marker=',', color='b', linestyle='none')
title('Contour of potential')
xlabel('X-axis')
ylabel('Y-axis')
colorbar()
#plot(ii[0]/Nx,ii[1]/Ny,'ro')
#plt.plot(ii[1],ii[0],'or')

print("Please wait for a moment")
errors=np.zeros(Niter)

for k in range(Niter):
    oldphi=phi.copy()
    phi[1:-1,1:-1] = 0.25 * (phi[1:-1,0:-2] + phi[1:-1,2:] + phi[0:-2,1:-1] + phi[2:,1:-1])
    phi[1:-1,0]=phi[1:-1,1]       #left  # Set Boundary conditions
    phi[1:-1,Nx-1]=phi[1:-1,Nx-2] #right
    phi[0,1:-1]=phi[1,1:-1]       #top
    #phi[0,0]=phi[0,1]
    #phi[0,Nx-1]=phi[0,Nx-2]
    phi[Ny-1:,1:-1]=Tbottom       #bottom=0
    phi[ii]=Trod                  #central portion=1V
    errors[k]=(abs(phi-oldphi)).max()
k=list(range(Niter))

figure(2)
grid(True)
semilogy(k[::50],errors[::50],'o-')
plt.title('Errors in y-axis Semilog versus iteration number k')
plt.xlabel('Iteration')
plt.ylabel('Error')
print("Iteration finished")

figure(3)
grid(True)
semilogy(k[500::50],errors[500::50],'o-')
title('Error versus iteration number above 500')
plt.xlabel('Iteration')
plt.ylabel('Error')

figure(4)
grid(True)
loglog(k[::50],errors[::50],'o-')
title('Errors in y-axis Loglog versus iteration number k')
loglog(k[500:],errors[500:],color='r',label='Loglog error beyond 500')
legend()
plt.xlabel('Iteration')
plt.ylabel('Error')

# initialise funstions for the fitting
flog = lambda t : log(t)
fexp = lambda t : e**t

error_log = flog(errors)
a_log = np.zeros((Niter,2)) # a_log*x_fit = error_log. Hence forming the matrix A
a_log[:,0]=1
#print(a_log)
for i in range(1,Niter+1): 
    a_log[i-1,1] = i
#print(a_log)
x_fit = lstsq(a_log,error_log,rcond = None)[0] # calculating the x_fit
#print(x_fit)
error_obtained = a_log.dot(x_fit) # calculating the error obtained by fitting curve
figure(5)
semilogy(list(range(Niter)),fexp(error_obtained), 'b')
title('Fit considering entire range of data') # plotting the error plot for fitted data
xlabel('Iteration')
ylabel('Error')

error_log_500 = flog(errors[500:]) # this block of code computes the error and fits the data for 500 point onwards
a_log_500 = np.zeros((Niter-500,2))
a_log_500[:,0]=1
for i in range(501,Niter+1):
    a_log_500[i-501,1] = i
x_fit_500 = lstsq(a_log_500,error_log_500,rcond = None)[0]
error_obtained_500 = a_log_500.dot(x_fit_500)
figure(6)
semilogy(list(range(501,Niter+1)),fexp(error_obtained_500), 'g')
title('Fit considering 500 data point onwards')
xlabel('Iteration')
ylabel('Error')

'''lst =lstsq(np.c_[np.ones(Niter-500),np.arange(Niter-500)],np.log(errors[500:]))
a,b =np.exp(lst[0][0]),lst[0][1] 
print(a,b)

figure(4)
grid(True)
title('Expected vs actual error(>500 iterations')
semilogy(arange(500,Niter),a*np.exp(b*np.arange(Niter-500),'r-')
semilogy(arange(500,Niter),errors[500:],'b-')
plt.legend(("Estimated Exponential","Actual Exponential"))
legend()
plt.xlabel('Iteration')
plt.ylabel('Error')

#plotter(1,np.arange(500,Niter),a*np.exp(b*np.arange(Niter-500)),"Iteration number","error",type=plt.semilogy,arg3="r-")
#plotter(1,np.arange(500,Niter),errors[500:],"Iteration number","Error",type=plt.semilogy,title='Expected vs actual error (>500 iter)')
#plt.legend(("Estimated Exponential","Actual Exponential"))
lstapprox =lstsq(np.c_[np.ones(Niter),np.arange(Niter)],np.log(errors))
a,b = np.exp(lstapprox[0][0]),lstapprox[0][1]
print(a,b)
figure(5)
grid(True)
title('Expected vs actual error(>500 iterations')
semilogy(arange(500,Niter),a*np.exp(b*np.arange(Niter),'r-')
semilogy(arange(500,Niter),errors[500:],'b-')
plt.legend(("Estimated Exponential","Actual Exponential"))
legend()
plt.xlabel('Iteration')
plt.ylabel('Error')

#plotter(2,np.arange(Niter),a*np.exp(b*np.arange(Niter)),"Iteration number","Error",type=plt.semilogy,title='Error versus iteration number',arg3 = 'r-')
#plotter(2,np.arange(Niter),errors,"Iteration number","Error",type=plt.semilogy,title='Expected vs actual')
#plt.legend(("Estimated Exponential","Actual Exponential"))
'''

fig7=figure(7) # plot the 3d plot for the obtained potential as a function of X and Y coordinates
ax = p3.Axes3D(fig7)
title('3-D graph of Potential')
surf = ax.plot_surface(Y,X,phi.T,rstride=1,cstride=1)
ax.set_xlabel('X coordinates')
ax.set_ylabel('Y Coordinates')


ii= list(ii) # b1 and b2 are composed of the scaled version of ii to [-0.5,0.5]
#print(ii[0])
b1 = ii[0].copy()
b1.dtype = np.float32
b2 = ii[1].copy()
b2.dtype = np.float32
for i in range(len(ii[0])) :
    b1[i]=x[(ii[0])[i]]
for j in range(len(ii[1])) :
    b2[j]=y[(ii[1])[j]]

Jx = np.zeros((Ny,Nx))
Jy = np.zeros((Ny,Nx))
Jx[:,1:-1] = 0.5*(phi[:,0:-2]-phi[:,2:])
Jy[1:-1,:] = 0.5*(phi[0:-2,:]-phi[2:,:])

#Contour plot of potential
figure(8)
title('The contour plot of potential')
contourf(x,-y,phi,cmap=cm.hot)
plt.plot(ii[0]/Nx-0.48,ii[1]/Ny-0.48,'ro')
colorbar()
grid()

#Vector plot of current
figure(9)
plt.quiver(x,-y,Jx,-Jy,scale=5)
scatter(b1,b2,s=6,c='r')
#plt.plot(ii[0]/Nx-0.48,ii[1]/Ny-0.48,'ro')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Current vectors in the plate")


plt.ion()
phij = np.ones((Nx,Ny))*300
for i in range(Niter*2):
    phij[1:-1,1:-1]=0.25*(phij[1:-1,0:-2]+ phij[1:-1,2:]+
        phij[0:-2,1:-1]+ phij[2:,1:-1] + Jx[1:-1,1:-1]*Jx[1:-1,1:-1]+Jy[1:-1,1:-1]*Jy[1:-1,1:-1]);
    phij[1:-1,0]=phij[1:-1,1]
    phij[1:-1,-1]=phij[1:-1,-2]
    phij[-1,:]=phij[-2,:]
    phij[0,:] = 300
    phij[ii] =300
plt.ioff()

figure(10)
grid(True)
contourf(y,-x,phij,cmap=cm.hot)
plt.title('Contour plot of temperature')
plt.xlabel('X axis')
plt.ylabel('Y axis')
colorbar()
#plotter(1,x,y,"X axis","Y axis",plt.contourf,phij,"Contour plot of Temperature",cmap=matplotlib.cm.hot)
plt.show()

show()
