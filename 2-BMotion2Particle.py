from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
import time

# Initialized number of particle

numberofparticle = 3

def Brownian(Xint,n,dt,delta,outPut=None):

    print('Calculating Brownian motion')
    Xi0 = np.asarray(Xint)

    #finding stand deviation by using
    # random variates =rvs
    #based on formula N(0,(delta)^2*dt;t,dt+t)

    NDist = norm.rvs(size=Xi0.shape + (n,), scale=dt*(np.power(delta,2)) )

    if outPut is None:
        #create empty array and shape the array
        outPut =np.empty(NDist.shape)


    #else find cummulative sum of array NDist
    # and put it in another shape of array with axis = -1
    np.cumsum(NDist, axis= -1, out=outPut)

    outPut += np.expand_dims(Xi0,axis=-1)

    return outPut


def appendlist(particleobj,particle):
    '''
        This function is executed by thread
        The execution of this function is based on same PID but different thread.
        where each time thread is received, particle is also received.
        One thread is for One particle.
        the thread is check by particle to
        determine which particle is calculated by using if else.
        the value of each thread is return based on x0,x1 and x2 in brownian function

    '''

    # to get the pid for the thread being execute
    pid = os.getpid()

    # for particle initially start from 0 and what being send is 0
    # therefore it need to add by 1 to show that it actually partical 1 not 0
    particle += 1

    if particle == 1:
        #show which particle is calculated
        print('\nParticle: ', particle)
        # get the thread id for that patrticle
        threadid0 = threading.get_ident()
        # shows the thread id and pid of that particle
        print('PID: %s Thread ID: %s' % (pid, threadid0))
        # calculate brownian function for particle one and keep it in x0
        a = Brownian(x0[:, 0], N, dt, delta, outPut=x0[:, 1:])
        # print a to check later on in the plotting is it the same value of x0
        print('Particle ID: %s Brownian motion: %s  \n' % (threadid0,a) )

    elif particle == 2:
        print('\nParticle: ', particle)
        threadid1 = threading.get_ident()
        print('PID: %s Thread ID: %s' % (pid, threadid1))
        b = Brownian(x1[:, 0], N, dt, delta, outPut=x1[:, 1:])
        print('Particle ID: %s Brownian motion: %s  \n' % (threadid1,b) )

    else:
        print('\nParticle: ', particle)
        threadid2 = threading.get_ident()
        print('PID: %s Thread ID: %s' % (pid, threadid2))
        c = Brownian(x2[:, 0], N, dt, delta, outPut=x2[:, 1:])
        print('Particle ID: %s Brownian motion: %s  \n' % (threadid2,c) )

    particleobj.wait()

# Initialization
delta = 0.05
T = 12.0
N = 10
dt = T/N

# Three particle is used therefore three x list are created
x1 = np.empty((2, N+1))
x2 = np.empty((2, N+1))
x0 = np.empty((2, N+1))

# First array where it start at x and y is at origin
x0[:, 0] = 0.0
x1[:, 0] = 0.0
x2[:, 0] = 0.0

print('Creating thread')

# create threadlist for the function
threadlist = []

time.sleep(0.5)

# thread event of the particle
particleobj = threading.Event()

# Iteration of threads based on number of particle
for particle in range(numberofparticle):
    # send thread to that consist of  particleobj and particle function appendlist
    t = threading.Thread(target = appendlist, args = (particleobj,particle, ))
    threadlist.append(t)

print('Main Pid: %s' % os.getpid())

# start process of thread
print('Start thread')
for tpart in threadlist:
    tpart.start()

time.sleep(1)
print('Notify all thread')
particleobj.set()

# closed thread
print('Closed thread')
for tpart in threadlist:
    tpart.join()

time.sleep(2)

# Shows position of particle 1, 2 and 3
print('Position of x and y for each particle:')
print('\nParticle 1: ', x0[0], x0[1])
print('\nParticle 2: ', x1[0], x1[1])
print('\nParticle 3: ', x2[0], x2[1])


print('Generating graph...')

# marker o is to show where the position of particle everytime it moves
plt.plot(x0[0],x0[1], linestyle='--', marker='o', color = "blue")
plt.plot(x1[0],x1[1], linestyle='--', marker='o', color = "orange")
plt.plot(x2[0],x2[1], linestyle='--', marker='o', color = "green")

# x and y initial of particle 1 with purple color
plt.plot(x0[0,0],x0[1,0], 'mo')
# x and y final of particle 1 with red color
plt.plot(x0[0,-1], x0[1,-1], 'ro')

plt.plot(x1[0,0],x1[1,0], 'mo')
plt.plot(x1[0,-1], x1[1,-1], 'ro')

plt.plot(x2[0,0],x2[1,0], 'mo')
plt.plot(x2[0,-1], x2[1,-1], 'ro')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

# show the legend
plt.legend(['Particle 1','Particle 2','Particle 3'], loc='upper left')
plt.title('Brownian Motion in 2D')
plt.show()

plt.savefig('BrownianMotion2D-3particles.png')

print('Graph completed')

