# Both Pick

Both Pick is a CMSE 830 project about choosing a movie for two people. It uses MovieLens ratings to estimate individual preferences and compare different ways of finding a shared recommendation.

## Work so far

The data has been audited and prepared as a reproducible sample of 5,000 MovieLens users. The analysis covers rating patterns, movie coverage, and genre summaries. Global-mean and user/movie-bias baselines have been compared with matrix factorization using separate training, validation, and test ratings.

The Streamlit app lets two anonymized real MovieLens profiles compare three approaches: highest average predicted rating, highest lower predicted rating, or a minimum predicted rating required by both viewers. It includes genre filters, preference plots, and strategy comparisons. The current results and recommendation logic have passed reproducibility and app checks.

Predicted ratings are estimates, and combining them does not measure actual enjoyment together. Genre frequency describes what someone has rated, not their complete taste.

## Next steps

Review the results and their limitations, complete the analysis, and explore genre-based support for users with limited rating histories. That support is not implemented yet. This is a working version and will continue to change before the final submission.

Data: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/), GroupLens, University of Minnesota. See the [dataset conditions](https://files.grouplens.org/datasets/movielens/ml-32m-README.html). Reference: Harper and Konstan (2015), [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872).
