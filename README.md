# mv labeler (momentum / trend labelers)
This is a collection of labeling functions for trend / momentum in financial timeseries.   Given a price / return series, the labelers will label each sample in the price series with { -1, 0, +1 } indicating whether is part of a downward, neutral, or upwards trend respectively.   

I developed the initial labeler 5 or 6 years ago and have used since for precisely identifying momentum and trend patterns.   This is useful for:

- collecting price moves for pattern analysis
- comparing online trend or MR signal versus optimum behavior as identified ex-post by this labeler
- labels for supervised machine learning in learning momentum signals
- studying market microstructure around large moves


