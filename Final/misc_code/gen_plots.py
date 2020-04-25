import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math


def make_sine_plot():
    f = np.vectorize(lambda x: math.sin(x))
    x = np.linspace(0,2*math.pi,500)
    y = f(x)

    fig,ax = plt.subplots()
    ax.plot(x,y)
    ax.set(xlabel='x',ylabel='y')

    plt.grid()
    plt.savefig('../pictures/simple_sine_function.png')

def make_complex_sine_plot():
    f1 = np.vectorize(lambda x: (0.5*math.sin(2*math.pi*4*x)) + (1.5*math.sin(2*math.pi*1.5*x)))
    x1 = np.linspace(0,2*math.pi,500)
    y1 = f1(x1)

    f2 = np.vectorize(lambda x: 0.5*math.sin(2*math.pi*4*x))
    x2 = np.linspace(0,2*math.pi,500)
    y2 = f2(x2)

    f3 = np.vectorize(lambda x: 1.5*math.sin(2*math.pi*1.5*x))
    x3 = np.linspace(0,2*math.pi,500)
    y3 = f3(x3)

    gs = gridspec.GridSpec(2,2)
    plt.figure(figsize=(16,8))

    ax = plt.subplot(gs[0,:])
    plt.plot(x1,y1)
    ax.set(xlabel='x',ylabel='y')
    plt.grid()

    ax = plt.subplot(gs[1,0])
    ax.set(xlabel='x',ylabel='y',ylim=(-2,2))
    plt.plot(x2,y2)
    plt.grid()

    ax = plt.subplot(gs[1,1])
    ax.set(xlabel='x',ylabel='y',ylim=(-2,2))
    plt.plot(x3,y3)
    plt.grid()

    plt.savefig('../pictures/complex_sine_function.png')

if __name__ == '__main__':
    make_sine_plot()
    make_complex_sine_plot()