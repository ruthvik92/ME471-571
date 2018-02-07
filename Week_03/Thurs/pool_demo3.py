
from multiprocessing import Pool
import random

results = []

def f(t):
    return t

def cb_append_result(t):
    results += [t]        # local variable 'results' referenced before assignment
    # results.append(t)       # works
    
if __name__ == '__main__':
    pool = Pool() 
    t = random.random()
    
    pool.apply_async(f,args=(t,),callback=cb_append_result)    
    pool.close()
    pool.join()
    print("Result is {}".format(results))