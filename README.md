## Fake_News_Detection

# Fake News Classification - Pattern Recognition

Fake news and hoaxes have been there since before the advent of the Internet. The widely accepted definition of Internet fake news is: fictitious articles deliberately fabricated to deceive readers”. Social media and news outlets publish fake news to increase readership or as part of psychological warfare. Ingeneral, the goal is profiting through clickbaits. Clickbaits lure users and entice curiosity with flashy headlines or designs to click links to increase advertisements revenues. This exposition analyzes the prevalence of fake news in light of the advances in communication made possible by the emergence of social networking sites. The purpose of the work is to come up with a solution that can be utilized by users to detect and filter out sites containing false and misleading information. We use simple and carefully selected features of the title and post to accurately identify fake posts. The experimental results show a 99.4% accuracy using logistic classifier.



This project applies various machine learning classifiers to the task of **fake news classification**. The goal is to determine whether a given news article is real or fake based on its textual content. The models in this project utilize **Doc2Vec** embeddings for feature extraction and apply multiple classification algorithms including K-Nearest Neighbors, Bayesian Classifier, Random Forest, and more.


Required Libraries:
```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
scikit-plot
keras
opencv-python
gensim
nltk
tensorflow
```
├── datasets/
│   ├── train.csv            # CSV file with training data (text and labels)
├── xtr.npy                  # Training features (Embeddings)
├── xte.npy                  # Test features (Embeddings)
├── ytr.npy                  # Training labels (0: Fake, 1: Real)
├── yte.npy                  # Test labels (0: Fake, 1: Real)
├── main.py                  # Main script to run the classifiers
How to Use
Preprocessing and Embedding: The first step involves reading the dataset (train.csv) and preprocessing the text data. The text is cleaned, stopwords are removed, and Doc2Vec embeddings are generated.

Training the Model: After the embeddings are created, the dataset is split into training and testing sets. The machine learning models are trained on the training set and tested on the testing set.

Choose a Classifier: You can select from the following classifiers by entering the corresponding number:
```bash
K-Nearest Neighbors (KNN)
Bayesian Classifier (GaussianNB)
Random Forest Classifier
Multilayer Perceptron (MLP)
AdaBoost Classifier
Decision Tree Classifier
Gaussian Process Classifier
Support Vector Machine (SVM)
Stochastic Gradient Descent (SGD)
Neural Network (TensorFlow)
Confusion Matrix and Evaluation: After training, a confusion matrix will be displayed, along with other evaluation metrics such as accuracy, precision, recall, and F1-score.
```
Running the Code
To run the code, execute the main.py script. Here is an example of how to use it:

```bash
python main.py
```
You will be prompted to select one of the classification models. Enter the corresponding number (e.g., 1 for K-Nearest Neighbors).

The main function PRProject_Main(n) is responsible for calling the appropriate classifier based on the user's selection.

Handwritten Digit Recognition Project-Pattern Recognition

To Choose your Method  Run, First You Need To Install Required Libraries

1. Knn Classifier
2. Bayesian Classifier
3. Random Forest Classifier
4. Multilayer Perceptron Classifier
5. Ada Boost Classifier
6. Decision Tree Classifier
7. Gaussian Process Classifier
8. Support Vector Machine
9. Stochastic Gradient Descent
10. Neural Network (TensorFlow)

Choosing The Number Of Method : 

After selecting the classifier, the script will:

Train the model on the training set.
Output the classification report.
Display a confusion matrix to evaluate the performance.
Classifiers Included
This project includes implementations for the following classification algorithms:

1. K-Nearest Neighbors (KNN)
KNN is a simple and widely-used classification algorithm that classifies data points based on the majority class among their nearest neighbors.

2. Bayesian Classifier (GaussianNB)
This model uses Bayes' Theorem with the assumption of normality for classifying the news articles.

3. Random Forest Classifier
Random Forest is an ensemble method that creates a forest of decision trees, where each tree is trained on a random subset of the features.

4. Multilayer Perceptron (MLP)
A neural network-based classifier that uses multiple hidden layers to learn the decision boundaries.

5. AdaBoost Classifier
AdaBoost is an ensemble technique that combines weak classifiers into a strong classifier by giving more weight to misclassified samples.

6. Decision Tree Classifier
A tree-like model where each node represents a decision based on one of the input features, and the leaves represent the final classification.

7. Gaussian Process Classifier
A probabilistic classifier that uses a Gaussian process as a prior over functions.

8. Support Vector Machine (SVM)
SVM finds the hyperplane that best separates the data into different classes by maximizing the margin between the classes.

9. Stochastic Gradient Descent (SGD)
SGD is an optimization technique that updates the model parameters based on random subsets of the training data.

10. Neural Network (TensorFlow)
This deep learning model uses fully connected layers and ReLU activation functions to classify the news articles.

Results and Metrics
Each classifier outputs the following evaluation metrics:

Accuracy: The percentage of correctly classified instances.
Precision: The ratio of true positive results to all predicted positives.
Recall: The ratio of true positive results to all actual positives.
F1-Score: The harmonic mean of precision and recall.
Confusion Matrix: A matrix showing the true vs predicted classifications, which is visualized using a heatmap.
Example:
For K-Nearest Neighbors, after training and evaluation, the following might be output:
```bash
KNN Accuracy Is: 92.45 %
Confusion matrix, without normalization
[[ 250   35]
 [  30  215]]
```
Conclusion
This project provides a robust framework for experimenting with various classifiers to detect fake news. The use of Doc2Vec embeddings enables the model to handle textual data efficiently, while different classification algorithms give you the flexibility to experiment with various techniques for optimal performance.

Feel free to modify the code and explore additional classification models or techniques!
