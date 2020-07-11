# Financial Timeseries Patterns library
This package will contain a collection of price pattern detectors (online and offline).  I am starting this library by open sourcing one of the labeling algorithms I use.

A few years ago developed an algorithm to label momentum and trend patterns in intra-day or daily price data.  In spite of its simplicity, has performed quite well as compared to a number of more complicated statistical approaches.  As is not especially proprietary, hence thought to share this more broadly.  I will be adding other pattern related algorithms to this library over time.

I use these algorithms for:

- collecting price moves for pattern analysis
- comparing online trend or MR signal versus optimum behavior as identified ex-post by this labeler
- labels for supervised machine learning in learning momentum signals
- studying market microstructure around large moves

## AmplitudeBasedLabeler
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
![Graph of labels](/docs/labeling.20.5.png)

### Labeling (minamp = 20bps, Tinactive = 15mins)
```Python
labeler = AmplitudeBasedLabeler (minamp = 20, Tinactive = 30)
labels = labeler.label (df)
labeler.plot()
```

![Graph of labels](/docs/labeling.20.15.png)

