import numpy as np 
import sympy as smp 
import scipy as sp 
import matplotlib.pyplot as plt 
from IterativeBacktest import * 
from IterativeBase import * 


def basic_calc(S0= 100, mu=0.05, sigma= 0.2, T= 1, steps= 252): 
    dt = T / steps 
    Z = np.random.normal(0, 1, steps)
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z 
    prices = S0 * np.exp(np.cumsum(increments))
    return prices 

for i in range(2): 
    plt.plot(basic_calc())
    plt.title("Multiple brownian motion charts")
    plt.show() 

prices = basic_calc() 

def monte_carlo_simulations(S0= 100, mu=0.05, sigma= 0.2, T= 1, steps= 252, N =100000):
    finals = np.zeros(N)
    for i in range(N):
        path = basic_calc(S0, mu, sigma, T, steps)
        finals[i] = path[-1]
        
    print(f" Expected final price : {np.mean(final_prices)}" )
    print(f" Expexted Standard deviation: {np.std(final_prices)} ") 
    

    return finals 

final_prices = monte_carlo_simulations()

print(f" Expected final price : {np.mean(final_prices)}" )
print(f" Expexted Standard deviation: {np.std(final_prices)} ") 

plt.hist(final_prices, bins=50)
plt.title("Normal Distribution with Browninan motion")
plt.xlabel("Final price")
plt.ylabel("freq")  
plt.show()


def brownian_motion_strat(): 
    brownian_price = basic_calc() 
    return brownian_price



