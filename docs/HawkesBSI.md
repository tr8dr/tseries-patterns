# HawkesBSI
This is a buy/sell imbalance indicator modeled as a Hawkes "self-exciting" process.  Given buyer / seller labelled
volume this gives an indication of buyer or seller momentum, identifying start and end of substantial price moves.

Hawkes processes model the intensity of events and are applicable for events which tend to cluster in a self-exciting
manner.  The form of the Hawkes process we use is:

<img src="https://render.githubusercontent.com/render/math?math=H%28t%29%20%3D%20%5Cmu%20%28t%29%20%2B%20%5Csum_%7Bi%3D0%7D%5E%7Bt%7D%20N%28i%29%20e%5E%7B-%20%5Ckappa%20%5CDelta%20t%7D">

where N(i) indicates the # or magnitude of events at time t_i and κ is the decay factor.  In the case of 
buy/sell imbalance, since is a 0 centered process, our mean μ(t) is 0 and N(i) is the signed buy-sell change in
volume for a particular period.

This implementation of the BSI signal requires a market data source that either provides buyer and seller attributed trades
or used a classifier to determine likely buyer or seller attribution.   The data frame should contain:

- timestamp { a column named, one of: 'stamp', 'time', 'date', 'datetime' }
- price { a column named either: 'price' or 'close' }
- buyvolume
- sellvolume


## Examples
Below are some examples of the same (intra-day) data series, parameterized with different decays.

### BSI with decay (kappa) = 0.1
```Python
df = pd.read_csv("csv/volumebars.csv", parse_dates=['stamp'])

obj = HawkesBSI(0.1)
obj.eval (df)
obj.plot()
```
![Graph of BSI](/docs/images/BSI1.png)

### BSI with decay (kappa) = 0.5
```Python
obj = HawkesBSI(0.5)
obj.eval (df)
obj.plot(title="variation with kappa = 0.5")
```
![Graph of BSI](/docs/images/BSI2.png)

