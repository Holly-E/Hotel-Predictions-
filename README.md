# Hotel Cluster Selection 
Predict which hotel cluster the user is likely to book, given search details.

In dealing with missing data I determined that ‘orig_destination_distance’ had over 13 million
NaN values, and ‘srch_ci’ and ‘srch_co’ had 47k missing values. I replaced the
‘orig_destination_distance’ missing values with the mean value of 1970 and added a feature
denoting whether or not the original data had a missing value for this field.

Originally I was going to use the most common values for the missing values of ‘srch_ci’ and
‘srch_co’, however the most common check-in month was December and the most common
check-out was August, as people with check-ins in Dec probably check-out after the new year in
many cases. A check-in of Dec and check-out of August wouldn’t make sense when looking at
the length of stay. In addition, the test set mainly took place in 2015 so imputing the year 2014
would not make sense, so instead I opted to impute for these a value of -1.

All columns were int or float type except for the ‘date_time’, ‘srch_ci’ and ‘srch_co’ columns
which were objects. I converted each of these objects into datetime values, then extracted the
year, month, day, day of the week (and hour for ‘date_time’) for each of these columns into
separate features.

I created a field for the length of stay and the number of days between booking the trip and the
start of the trip. I thought it was interesting that the most common length of stay was one day,
and trips were most commonly booked same-day as check-in.

Destinations file had 150 columns which would greatly increase the run time of machine
learning algorithms. I used PCA to reduce these down to a representative 5 columns and joined
this with the training and test data.

I used a combination of feature selection methods to reduce the number of features down from
41. I found the top 10 features for each of 5 methods: pearson correlation, chi-squared,
recursive feature elimination with logistic regression as the estimator, lasso regression and
random forest’s feature importance. I kept the features that were in the top 10 for a minimum of
2 of these methods, which whittled it down to the top 14 features. ‘Orig_destination_distance’
was the only feature the was in the top 10 for all 5 feature selection methods.

Finally, to make the predictions I used K Nearest Neighbor, Random Forest, Logistic Regression
and a Multi-layer Perceptron classifier. I used sci-kit learn’s Gridsearch CV for hyperparameter
tuning. KNN, random forest, and the MLP classifer performed the best on my cross-validation
set, so I fed the results for these three into a voting classifer, which uses predicted class labels
for majority rule voting
