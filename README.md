# mv labeler (momentum / trend labeler)
This package will contain a collection of price pattern detectors (online and offline).  A few years ago developed an algorithm to label momentum and trend patterns in intra-day or daily price data.  In spite of its simplicity, has performed astonishingly well. As is not especially proprietary, thought to share this more broadly.

I use this algorithms for:

- collecting price moves for pattern analysis
- comparing online trend or MR signal versus optimum behavior as identified ex-post by this labeler
- labels for supervised machine learning in learning momentum signals
- studying market microstructure around large moves

The labeler behavior is defined by two parameters (which seem intuitive from a trading perspective):

- minimum trend / momentum amplitude of interest
   * this should be some multiple of volatility / noise
- maximum amount of noise allowed in move:
   * defined by maximum period where no new high (low) is achieved), as well as
   * no drawdown in move exceeding the minimum move amplitude

There are other ways to define noise or extension, but these choises resulted in a super-simple model, that works well.   In addition an incremental OLS is performed to determine which points best fit the move, discarding outliers around the edges.


## Examples
The labeler uses a 
