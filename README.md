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

Running the Code
To run the code, execute the main.py script. Here is an example of how to use it:

```bash
python main.py
```
You will be prompted to select one of the classification models. Enter the corresponding number (e.g., 1 for K-Nearest Neighbors).

The main function PRProject_Main(n) is responsible for calling the appropriate classifier based on the user's selection.
