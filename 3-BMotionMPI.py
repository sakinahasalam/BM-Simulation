from mpi4py import MPI
import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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

# Initialization
delta = 0.05
T = 12.0
N = 5
dt = T/N

# empty x list
x0 = np.empty((2, N+1))


# First array where it start at x and y is at origin
x0[:, 0] = 0.0



# MPI  by rank 1 Mother 3 children

print('-----------Master giving command----------- ')
# check how many np is running. this case is 3 mp which later divided into 3 worker
comm = MPI.COMM_WORLD
# generate rank based on comm
rank = comm.Get_rank()

print('Worker = ', rank)

if rank == 0:
    print('\n Master dividing job to worker \n')
    # N+1 is the space of the array
    xlist = np.array_split(np.linspace(0, T, 3), 3)
    # 0 to T is the time, 3 is how many data want to be calculated
    # the next 3 after the linspace is np / worker being decided previously

else:
    xlist = None

# Scattering the xlist value
xlisting = comm.scatter(xlist, root=0)
result = []
workresult = []
finallist = []

starttime = time.time()
for xnum in xlisting:
    print('Worker is doing his job.. please be patient..')
    # set worker to do Brownian function everytime executing the job
    WorkerJob = Brownian(x0[:, 0], N, dt, delta, outPut=x0[:, 1:])
    print('Particle position:', WorkerJob)
    result.append(WorkerJob)

print('Collecting...')
workresult.append(result)
processtime = time.time() - starttime

# merging the result

print("\n --->   Merging the Result \n")
mainresult = comm.gather(workresult, root=0)
timeresult = comm.gather(processtime, root=0)

# pass all the data to master list
if rank == 0:
    print('------------------------')
    print('-------MASTERLIST-------')
    print('------------------------')

    finallist.append(mainresult)
    print('Masterlist = ', finallist)


# marker o is to show where the position of particle everytime it moves
plt.plot(finallist[0],finallist[1], linestyle='--', marker='o', color = "blue")


# x and y initial of particle 1 with purple color
plt.plot(finallist[0,0],finallist[1,0], 'mo')
# x and y final of particle 1 with red color
plt.plot(finallist[0,-1], finallist[1,-1], 'ro')



plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

# show the legend
plt.legend(['Particle 1'], loc='upper left')
plt.title('Brownian Motion in 2D')

plt.savefig('BrownianMotion2D-MPI.png')