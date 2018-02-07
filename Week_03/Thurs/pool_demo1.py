%%file demo1.py

from multiprocessing import Pool, TimeoutError
import multiprocessing
import time
import os
import numpy

def zzz(z):
    pnum, t = z    # Distribute tuple to variables.
    id = os.getpid()
    print("In process {} ({:2d}) is waiting {:8.4f} seconds".format(id,pnum,t))
    time.sleep(t)
    return (pnum,t,os.getpid())

if __name__ == '__main__':
    np = 8
    njobs = 16
    pool = Pool(processes=16)              # start 4 worker processes

    print("Launching {} jobs on {} cores".format(njobs,np))    
    
    # launching multiple evaluations asynchronously *may* use more processes
    print("")
    sleep_times = 5*numpy.random.rand(njobs)
    pnum = range(njobs)
    z = zip(pnum,sleep_times)

    total_time = 0
    
    # Pool.map(func,iterable) <==> pool.map_async(func,iterable).get()
    t0 = time.time()
    res = pool.map(zzz,z)
    t1 = time.time()
    print("{:>30s} {:>12.6f}".format("Time in pool.map call",t1-t0))    
    
    
    
    
    elif method is 'map_async':
        t0 = time.time()
        # map_async only schedules tasks;  to run them, we need
        # pool.close(), pool.join()
        # OR 
        # multiple_results.wait() or multiple_results.get()
        multiple_results = pool.map_async(zzz,z)
        pool.close()
        pool.join()
        t1 = time.time()
        print("{:>30s} {:>12.6f}".format("Time in pool.map_async call",t1-t0))
        
        t0 = time.time()
        multiple_results.wait()    # same as below
        # while not multiple_results.ready():
        #    pass
        t1 = time.time()
        print("{:>30s} {:>12.6f}".format("Waiting for results",t1-t0))
        
        t0 = time.time()
        res = multiple_results.get()   # waits for all results to be returned
        t1 = time.time()
        print("{:>30s} {:>12.6f}".format("Getting results",t1-t0))        
        

    # how much time was spent in each process? 
    ps = sorted(set([z[2] for z in res]))    # Get a unique set of PIDs
    t_total = numpy.empty(np)
    for i in range(np):
        pass
        t_total[i] = sum([z[1] for z in res if z[2] == ps[i]])
        print("({})  t_total[{}] = {:12.4f} (seconds)".format(ps[i],i,t_total[i]))


    # total_time = sum([z[1] for z in res])        
    print("Total time in all tasks     : {:12.4f}".format(sum(sleep_times)))

    print("Done!!!")