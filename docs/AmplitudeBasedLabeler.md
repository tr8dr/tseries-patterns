# AmplitudeBasedLabeler
A few years ago developed an algorithm to label momentum and trend patterns in intra-day or daily price data.  
In spite of its simplicity, has performed quite well as compared to a number of more complicated statistical approaches.  
As is not especially proprietary, hence thought to share this more broadly.

Please note that the labeler makes use of a forward window to achieve 0 lag labeling.  Hence if you want to use a technique like this in live trading, would only be useful as a way to identify prior price moves, but cannot indicate the direction of the current time period.  The tradeoff is between 0 lag + lookahead or lag + no lookahead.

The labeler behavior is defined by two parameters (which seem intuitive from a trading perspective):

- minimum trend / momentum amplitude of interest
   * this should be some multiple of volatility / noise
- maximum amount of noise allowed in move:
   * defined by maximum period where no new high (low) is achieved), as well as
   * no drawdown in move exceeding the minimum move amplitude

There are other ways to define noise or extension, but these choises resulted in a super-simple model, that works well.   In addition an incremental OLS is performed to determine which points best fit the move, discarding outliers around the edges.


## Examples
Below are some examples of the same (intra-day) data series, parameterized for more noise, less noise, higher or lower minimum amplitudes.

### Labeling (minamp = 20bps, Tinactive = 5mins)
```Python
labeler = AmplitudeBasedLabeler (minamp = 20, Tinactive = 10)
labels = labeler.label (df)
labeler.plot()
```
![Graph of labels](/docs/images/labeling.20.5.png)

### Labeling (minamp = 20bps, Tinactive = 15mins)
```Python
labeler = AmplitudeBasedLabeler (minamp = 20, Tinactive = 30)
labels = labeler.label (df)
labeler.plot()
```

![Graph of labels](/docs/images/labeling.20.15.png)

