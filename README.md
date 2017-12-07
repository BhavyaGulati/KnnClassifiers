# KnnClassifiers
Iris Flower dataset: iris classification.  The problem is comprised of 150 observations of iris flowers from three different species. There are 4 measurements of given flowers: sepal length, sepal width, petal length and petal width, all in the same unit of centimeters. The predicted attribute is the species, which is one of setosa, versicolor or virginica.

## Follow the following steps:

Handle Data: Open the dataset from CSV and split into test/train datasets.
Similarity: Calculate the distance between two data instances.
Neighbors: Locate k most similar data instances.
Response: Generate a response from a set of data instances.
Accuracy: Summarize the accuracy of predictions.
Main: Tie it all together.

1. Handle Data

The first thing we need to do is load our data file. The data is in CSV format without a header line or any quotes. We can open the file with the open function and read the data lines using the reader function in the csv module.

Next we need to split the data into a training dataset that kNN can use to make predictions and a test dataset that we can use to evaluate the accuracy of the model.
We first need to convert the flower measures that were loaded as strings into numbers that we can work with. Next we need to split the data set randomly into train and datasets. A ratio of 67/33 for train/test is a standard ratio used.
Pulling it all together, we can define a function called loadDataset that loads a CSV with the provided filename and splits it randomly into train and test datasets using the provided split ratio.

2. Similarity

In order to make predictions we need to calculate the similarity between any two given data instances. This is needed so that we can locate the k most similar data instances in the training dataset for a given member of the test dataset and in turn make a prediction.

Given that all four flower measurements are numeric and have the same units, we can directly use the Euclidean distance measure. This is defined as the square root of the sum of the squared differences between the two arrays of numbers (read that again a few times and let it sink in).
Additionally, we want to control which fields to include in the distance calculation. Specifically, we only want to include the first 4 attributes. One approach is to limit the euclidean distance to a fixed length, ignoring the final dimension.
Putting all of this together we can define the euclideanDistance function.

3. Neighbors

Now that we have a similarity measure, we can use it collect the k most similar instances for a given unseen instance.

This is a straight forward process of calculating the distance for all instances and selecting a subset with the smallest distance values.
Implement getNeighbors function that returns k most similar neighbors from the training set for a given test instance (using the already defined euclideanDistance function)

4. Response

Once we have located the most similar neighbors for a test instance, the next task is to devise a predicted response based on those neighbors.
We can do this by allowing each neighbor to vote for their class attribute, and take the majority vote as the prediction.
getResponse function for getting the majority voted response from a number of neighbors. It assumes the class is the last attribute for each neighbor.


## Accuracy
Running the example, you will see the results of each prediction compared to the actual class value in the test set. At the end of the run, you will see the accuracy of the model. In this case, a little over 98%.

