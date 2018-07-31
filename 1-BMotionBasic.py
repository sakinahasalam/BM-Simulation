
#Coding for Simulation of Brownian Motion

#IMPORT LIBRARIES:
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt

#DEFINED THE FUNCTION FOR BROWNIAN MOTION:


def brownian(x0, n, dt, delta, outPut=None):

    print('\n Initializing x0..')
    x0 = np.asarray(x0)

    print('output',outPut)

    print('x0 is',x0)


    print('\nCalculate normal distribution..')
    NormDistribution = norm.rvs(size=x0.shape + (n,), scale=delta * (dt**2))
    print('Normal distribution', NormDistribution)

    print('\nChecking output')
    if outPut is None:
        outPut = np.empty(NormDistribution.shape)
        print('Output is', outPut)

    print('\noutput before: cummulative sum ', outPut)

    print('\nFind cummulative sum..')
    np.cumsum(NormDistribution, axis=-1, out=outPut)

    print('\noutput before expand dims: ', outPut)
    print('\nx0:', x0)

    outPut += np.expand_dims(x0, axis=-1)

    print('\noutput after: ', outPut)
    print('\nX: ',x)

    print('\n EXIT FUNCTION \n')

    return outPut

# INITIALIZATION OF PARAMETERS:
#  -----------------------------

# The Wiener process parameter.
delta = 0.25

	# Total time of the particle being observed.
T = 15.0

	# Number of steps.
	# By varying the number of step, the motion become more complicated
	# By increasing the step will increase the movement of particles : 100,500,1000
N = 5

	# Time step size
dt = T/N

# empty array to store x.
x = np.empty((2,N+1))

print('\n all value of x',x)

# initial condition for first value of X0.
x[:, 0] = 0.0

print('Initial x first position: ', x[:,0])


print('Output x[:,1]', x[:,1:])

#PASS DATA INTO FORMULA:
#----------------------

#pass data into brownian fuction to find the movement of particle
brownian(x[:,0], N, dt, delta, outPut=x[:,1:])

print('brownian x:', x)



#brownianstep

print('x axis:', x[0])
print('y axis:', x[1])
print('x1 :', x[0,0])
print('y1 :', x[1,0])
print('x2 :', x[0,-1])
print('y2 :', x[1,-1])

#PLOT THE TRAJECTORY OF PARTICLES IN X AND Y
	#-------------------------------------------

# Plot 2D trajectory.
plt.plot(x[0],x[1])

# Start (green) and end (red) points.
plt.plot(x[0,0],x[1,0], 'go')
plt.plot(x[0,-1], x[1,-1], 'ro')

# plot titles and axis.
# plt.title('Brownian Motion in 2D')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()
plt.savefig('BrownianMotion2D.png')










