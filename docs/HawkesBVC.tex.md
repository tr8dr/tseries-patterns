# HawkesBVC
This is a buy/sell imbalance indicator making use of the Bulk Volume Classifier as described by proposed by 
Easley, López de Prado and O'Hara [2013].  The bulk classification approach makes use of the return probability
distribution to classify unlabeled volume as buyer or seller originated.

The cdf of the volatility normalized return of each bar indicates whether the return is towards the middle (0.5),
left (0.0) or right (1.0) of the distribution.  For strong upward (downward) moves, we expect a cdf value approaching
1.0 (0.0).  We can calculate the signed volume for each bar as:

$$
signed\, volume_t = 2 (volume_t \times cdf(r_t / \sigma_{t})) - 1
$$   

To get an overall sense of signed buy/sell imbalance momentum we use a hawkes process.  Hawkes processes model the intensity 
of events and are applicable for events which tend to cluster in a self-exciting manner.  The form of the Hawkes process we use is:

$$
H(t) = \mu (t) + \sum_{i=0}^{t} N(i) e^{- \kappa \Delta t}
$$

where N(i) indicates the # or magnitude of events at time $$ t_i $$ and $$ \kappa $$ is the decay factor.  In the case of 
BVC, since is a 0 centered process, our mean $$ \mu (t) $$ is 0 and N(i) is the signed buy-sell change in
volume for a particular period.

As for the volatilty $$ \sigma_{t} $$, for the purposes of this implementation, simply computing a rolling standard deviation
over a window.  Other implementations may wish to use volume weighted standard deviation or other estimators of volatility.

Putting this together the indicator is implemented as:

$$
\begin{align*} 
signed\, volume_t = 2 (volume_t \times cdf(r_t / \sigma_{t})) - 1 \\
bvc(t) &= bvc(t-1) e^{-\kappa} + signed \, volume_t
\end{align*}
$$

This implementation of the BVC signal requires a market data source that includes volume.   The data frame should contain:

- timestamp { a column named, one of: 'stamp', 'time', 'date', 'datetime' }
- price { a column named either: 'price' or 'close' }
- volume or (buyvolume and sellvolume)


## Examples
Below are some examples of the same (intra-day) data series, parameterized with different decays.

### BVC (window = 30, kappa = 0.1)
```Python
obj = HawkesBVC(30,0.1)
obj.eval (df)
obj.plot()
```
![Graph of BVC](/docs/images/BVC1.png)

### BVC (window = 30, kappa = 0.05)
```Python
obj = HawkesBVC(30,0.05)
obj.eval (df)
obj.plot(title="variation with kappa = 0.05")
```

![Graph of labels](/docs/images/BVC2.png)

