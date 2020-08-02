# HawkesBVC
This is a buy/sell imbalance indicator making use of the Bulk Volume Classifier as described by proposed by 
Easley, López de Prado and O'Hara [2013].  The bulk classification approach makes use of the return probability
distribution to classify unlabeled volume as buyer or seller originated.

The cdf of the volatility normalized return of each bar indicates whether the return is towards the middle (0.5),
left (0.0) or right (1.0) of the distribution.  For strong upward (downward) moves, we expect a cdf value approaching
1.0 (0.0).  We can calculate the signed volume for each bar as:

<img src="https://render.githubusercontent.com/render/math?math=signed%5C%2C%20volume_t%20%3D%202%20%28volume_t%20%5Ctimes%20cdf%28r_t%20%2F%20%5Csigma_%7Bt%7D%29%29%20-%201"/>

To get an overall sense of signed buy/sell imbalance momentum we use a hawkes process.  Hawkes processes model the intensity 
of events and are applicable for events which tend to cluster in a self-exciting manner.  The form of the Hawkes process we use is:

<img src="https://render.githubusercontent.com/render/math?math=H%28t%29%20%3D%20%5Cmu%20%28t%29%20%2B%20%5Csum_%7Bi%3D0%7D%5E%7Bt%7D%20N%28i%29%20e%5E%7B-%20%5Ckappa%20%5CDelta%20t%7D">

where N(i) indicates the # or magnitude of events at time t_i and κ is the decay factor.  In the case of 
BVC, since is a 0 centered process, our mean μ(t) is 0 and N(i) is the signed buy-sell change in
volume for a particular period.

As for the volatilty σ(t), for the purposes of this implementation, simply computing a rolling standard deviation
over a window.  Other implementations may wish to use volume weighted standard deviation or other estimators of volatility.

Putting this together the indicator is implemented as:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign%2A%7D%20%0Asigned%5C%2C%20volume_t%20%26%3D%202%20%28volume_t%20%5Ctimes%20cdf%28r_t%20%2F%20%5Csigma_%7Bt%7D%29%29%20-%201%20%5C%5C%0Abvc%28t%29%20%26%3D%20bvc%28t-1%29%20e%5E%7B-%5Ckappa%7D%20%2B%20signed%20%5C%2C%20volume_t%0A%5Cend%7Balign%2A%7D"/>

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

