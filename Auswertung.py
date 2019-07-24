import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import * 

params = {'legend.fontsize': 20,
		'figure.figsize': (12, 8),
		'axes.labelsize': 22,
		'axes.titlesize':2,
		'xtick.labelsize':20,
		'ytick.labelsize':20,}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.add_subplot(111)

def lin_func(B,x):
	return B[0]*x + B[1]
	
def quad_func(B,x):
	return B[0]*x*x + B[1]
	
### Lebensdauer ###

heights = np.array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  6. , 7. ,  8. ,  9. , 10. , 12. , 14. , 16. ])
jumps = np.load('all-results.npy')
mean = jumps.mean(axis=1)
std = jumps.std(axis=1)

lin_Model = Model(lin_func)
data = RealData(heights, np.log(mean), sy=std/mean, )
odr = ODR(data, lin_Model, beta0=[10.,8.])
out = odr.run()
out.pprint()

x_fit = np.linspace(heights[0],heights[-1],1000)
y_fit = lin_func(out.beta, x_fit)

daten = ax.errorbar(heights, np.log(mean), yerr=std/mean, linestyle='',marker = 'o',label = 'Data', linewidth = 3)
fit = ax.plot(x_fit,y_fit, label = 'Linear Fit', linewidth = 3,)
extra = ax.plot([], [], '',color = 'white',label = r'm$= (-0{,}279 \pm 0{,}003)$'
+'\n'+ r'b$= (7{,}545 \pm 0{,}019)$')
# ~ ax.set_yscale("log")
ax.set(xlabel=r'height / a.u.', ylabel=r'ln(N)')
ax.legend(loc = 'upper right')
fig.savefig("logDecayFit.pdf", bbox_inches='tight')

plt.show()
