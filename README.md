# Financial Timeseries Patterns library
This package will contain a collection of price pattern detectors (online and offline).  I am starting this library by open sourcing one of the labeling algorithms I use.

A few years ago developed an algorithm to label momentum and trend patterns in intra-day or daily price data.  In spite of its simplicity, has performed quite well as compared to a number of more complicated statistical approaches.  As is not especially proprietary, hence thought to share this more broadly.  I will be adding other pattern related algorithms to this library over time.

I use these algorithms for:

- collecting price moves for pattern analysis
- comparing online trend or MR signal versus optimum behavior as identified ex-post by this labeler
- labels for supervised machine learning in learning momentum signals
- studying market microstructure around large moves

## Functionality
Here is a (growing) list of functionality provided by the library
- AmplitudeBasedLabeller ([doc](/docs/AmplitudeBasedLabeler.md))
- HawkesBSI ([doc](/docs/HawkesBSI.tex.md))
- HawkesBVC ([doc](/docs/HawkesBVC.tex.md))


