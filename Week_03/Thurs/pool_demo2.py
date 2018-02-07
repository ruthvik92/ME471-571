
from multiprocessing import Pool
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
    use_callback = True

    print("Launching {} jobs on {} cores".format(njobs,np))

    pool = Pool(processes=np) 
    
    # launching multiple evaluations asynchronously *may* use more processes
    print("")
    sleep_times = 5*numpy.random.rand(njobs)
    pnum = range(njobs)
    z = zip(pnum,sleep_times)

    if use_callback:    
        
        results = []
        time_total = 0
        def cb(result):
            print("In callback {}".format(result))
            print("")
            time_total += result.get()
            results.append(result)    
        
        t0 = time.time()
        for j in range(njobs):
            # This launches jobs, but they only get called with a .get() or .join, etc. 
            pool.apply_async(zzz,args=((pnum[j],sleep_times[j]),),callback=cb)
        t1 = time.time()
        print("Time launching jobs : {:12.4f}".format(t1-t0))
    
        # This is where jobs are actually started. 
        t0 = time.time()
        pool.close()
        pool.join()
        t1 = time.time()
        print("Time joining jobs : {:12.4f}".format(t1-t0))    
    
        # Use results collected from callback.
        res = results
    else:
        results = []
        t0 = time.time()
        for j in range(njobs):
            # This launches jobs, but they only get called with a .get() or .join, etc. 
            r = pool.apply_async(zzz,args=((pnum[j],sleep_times[j]),))
            results.append(r)
        t1 = time.time()
        print("Time launching jobs : {:12.4f}".format(t1-t0))
    
        # This is where jobs are actually started. 
        t0 = time.time()
        pool.close()
        pool.join()
        t1 = time.time()
        print("Time joining jobs : {:12.4f}".format(t1-t0))    
    
        # Use results collected from callback.
        res = [r.get() for r in results]


            
    # how much time was spent in each process? 
    ps = sorted(set([z[2] for z in res]))    # Get a unique set of PIDs
    t_total = numpy.empty(np)
    for i in range(np):
        t_total[i] = sum([z[1] for z in res if z[2] == ps[i]])
        print("({})  t_total[{}] = {:12.4f} (seconds)".format(ps[i],i,t_total[i]))

    # total_time = sum([z[1] for z in res])    
    #total_time = sum(sleep_times)            
    print("Total time in all tasks     : {:12.4f}".format(total_time))

    print("Done!!!")