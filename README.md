# Simulate Interest Rates with Vasicek Model Tree Construction and Monte Carlo Simulation
 
### author: Yi Rong
### date: 03/21/21

---

### Problem

Use Vasicek model tree construction and implement an algorithm
to calculate the interest rates for a 30-month period (&Delta;t = 1 month). Plot the risk-neutral
distributions of the short rate for 1, 2, …, 30 months (expectation of short rate and the ±1
Standard Deviations). Repeat the same calculation using Monte Carlo Simulation and compare
the results.

### Solution

Firstly, a function should be created to solve the two equations based on expected rate and
standard deviation, where <em>&alpha;</em> is the down rate for current time, <em>dt</em> is 1/12.

<img src="media/image1.png" width = "450" align="center">

After transformation, the solution is:

<img src="media/image2.png" width = "200" align="center">

The code is shown as below:

```{python }
# solve two equations
def solver(alpha, exp, dw):
    # dw is σ√dt
    r = exp - (dw)**2 / (alpha - exp) 
    return r
```

Then, the tree can be build under 2 circumstances: 
in even period or odd period. The code is presented below:

```{python }
#  build tree and calculate the interest rates for a 30 periods
def VTree(period, sigma, r0, theta, k, dt):
    dx = k * (theta - r0) * dt
    dw = sigma * np.sqrt(dt)
    nodes = [0.0] * ((2 * period + 1) * (period + 1)) 
    nodes = np.reshape(nodes, (2 * period + 1, period + 1))
    nodes[period, 0] = r0
    
    # expected rate for current time
    for i in range(1, period + 1, 1):
        nodes[period, i] = nodes[period, i - 1] + k * (theta - nodes[period, i - 1]) * dt
    
    # use the expected rate to calculate the up and down rates
    for i in range(1, period + 1, 2):
        nodes[period - 1, i] = nodes[period, i - 1] + k * (theta - nodes[period, i - 1]) * dt + dw
        nodes[period + 1, i] = nodes[period, i - 1] + k * (theta - nodes[period, i - 1]) * dt - dw
        nodes[period, i] = 0
    
    # extend the recombinding tree with a solver
    for i in range(2, period + 1, 1):
        # in even period
        if i%2 == 0:
            for j in range(period - 2, period - 2 - i, -2):
                alpha = nodes[j + 2, i]
                exp = nodes[j + 1, i - 1] + k * (theta - nodes[j + 1, i - 1]) * dt
                nodes[j, i] = solver(alpha, exp, dw)
                
            for j in range(period + 2, period + 2 + i, 2):
                alpha = nodes[j - 2, i]
                exp = nodes[j - 1, i - 1] + k * (theta - nodes[j - 1, i - 1]) * dt
                nodes[j, i] = solver(alpha, exp, dw)      
        
        # in odd period
        else:
            for j in range(period - 3, period - 2 - i, -2):
                alpha = nodes[j + 2, i]
                exp = nodes[j + 1, i - 1] + k * (theta - nodes[j + 1, i - 1]) * dt
                nodes[j, i] = solver(alpha, exp, dw)
            
            for j in range(period + 3, period + 2 + i, 2):
                alpha = nodes[j - 2, i]
                exp = nodes[j - 1, i - 1] + k * (theta - nodes[j - 1, i - 1]) * dt
                nodes[j, i] = solver(alpha, exp, dw)                    
    
    return nodes
    
Tree = pd.DataFrame(VTree(30, 0.0126, 0.05121, 0.15339, 0.025, 1 / 12))
```

The tree can be visualized as below:

<img src="media/image3.png" align="center">

Both risk-neutral method and Monte Carlo are used to calculate short rate for each period. The
code is shown as below:

```{python }
period = 30
sigma = 0.0126
r0 = 0.05121
theta = 0.15339
k = 0.025
dt = 1 /12

# risk-neutral results
rndist = [0.0] * (3 * (period + 1))
rndist = np.reshape(rndist, (3, period + 1))
rndist[0, 0] = r0
rndist[1, 0] = r0
rndist[2, 0] = r0

# Monte Carlo results
mc = [0.0] * (3 * (period + 1))
mc = np.reshape(mc, (3, period + 1))
mc[0, 0] = r0
mc[1, 0] = r0
mc[2, 0] = r0
for i in range(1, period + 1, 1):
    # calculate according to risk-neutral equation
    rndist[1, i] = rndist[1, i - 1] + k * (theta - rndist[1, i - 1]) * dt
    rndist[0, i] = rndist[1, i] + sigma * np.sqrt(i * dt)
    rndist[2, i] = rndist[1, i] - sigma * np.sqrt(i * dt)
    
    # calculate according to Monte Carlo, simulate 10000 times
    mc[1, i] = mc[1, i - 1] + k * (theta - mc[1, i - 1]) * dt + sigma * np.mean(np.random.normal(0, 1, 10000))
    mc[0, i] = mc[1, i] + sigma * np.sqrt(i * dt)
    mc[2, i] = mc[1, i] - sigma * np.sqrt(i * dt)
rndist = np.transpose(rndist)
mc = np.transpose(mc)
```

The graph is drawn as below, where the lines represent risk-neutral results and the scatters
represent the Monte Carlo results:

```{python }
rndist = pd.DataFrame(rndist)
mc = pd.DataFrame(mc)
rndist.columns = ['exp+sd', 'exp', 'exp-sd']
mc.columns = ['exp+sd_mc', 'exp_mc', 'exp-sd_mc']
rndist.plot()
plt.plot(mc,'go', markersize=4)
plt.title('Risk-neutral and Monte Carlo')
plt.show()
```

<img src="media/image4.png" align="center">

### Conclusion
When the simulation times are large enough, the Monte Carlo results will be very close
to the risk-neutral results.
