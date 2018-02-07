from multiprocessing import Pool
import time, os, random

def worker(z):
    jobnum, t = z    # Distribute tuple to variables.
    id = os.getpid()
    print("In process {} ({:2d}) is sleeping {:8.4f} seconds".format(id,jobnum,t))
    time.sleep(t)
    return (jobnum,t,os.getpid())

def print_pool_results(res,np):
    # how much time was spent in each process? 
    pids = sorted(set([z[2] for z in res]))    # Get a unique set of PIDs
    print("")
    print("Total time spent in each process")
    total_time = 0
    for i,p in enumerate(pids):
        proc_count = sum([1 for z in res if z[2] == p])
        proc_time  = sum([z[1] for z in res if z[2] == p])
        proc_jobs  = tuple([z[0] for z in res if z[2] == p])
        print("Process {:2d} ({})  {:8.4f}(s) {:4d} job(s) {}"
              .format(i+1,p,proc_time,proc_count,proc_jobs))
        total_time += proc_time
    print("")
    print("{:>25s} {:12.4f}".format("Total work done (s)",total_time))                
        