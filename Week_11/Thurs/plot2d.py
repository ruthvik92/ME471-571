
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_pcolor(ax,ay,bx,by,N,u):
    
    dx = (bx-ax)/N
    dy = (by-ay)/N
    xe = np.linspace(ax,bx,N+1)
    ye = np.linspace(ay,by,N+1)
    xc = xe[:-1] + dx/2
    yc = ye[:-1] + dy/2
    xem, yem = np.meshgrid(xe,ye)
    xcm, ycm = np.meshgrid(xc,yc)

    plt.pcolor(xem, yem, u)
    plt.xlim([ax,bx])
    plt.ylim([ay,by])
    plt.axis('square')

    plt.show()