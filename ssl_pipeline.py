
###Imporing the required packages:
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.base import clone
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
###Text Preprocessing and Cleaning:
def clean_text(text):
if not isinstance(text, str):
return '' # Return an empty string for non-string inputs
text = re.sub(r'http\S+', '', text) # Remove URLs
text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
text = text.lower() # Lowercase text
text = re.sub(r'\b\w{1,2}\b', '', text) # Remove words with 1 or 2 letters
text = re.sub(r'[^a-z\s]', '', text) # Keep text with letters and spaces
tokens = text.split() # Tokenize
# Remove stopwords
tokens = [word for word in tokens if word not in stopwords.words('english')]
# Lemmatize
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]
return ' '.join(tokens)
# Apply the clean_text function to the 'Review Text' column
df['Review Text'] = df['Review Text'].apply(clean_text)
print(df)
###Split data into labeled and unlabeled
# Define labeled data as data where "Sentiment" is not missing
labeled_data = df[df['Emotions'].notna() & (df['Emotions'] != 'NaN')]
###Extract X and y from labeled_data
# Extract labels from labeled_data
y_labeled = labeled_data['Emotions']
y_unlabeled = unlabeled_data['Emotions']
X_labeled = labeled_data['Review Text']
X_unlabeled = unlabeled_data['Review Text']
#print("y_labeled ",y_labeled )
#print("y_unlabeled",y_unlabeled)
#print("X_unlabeled",X_unlabeled)
#print("X_labeled",X_labeled)
#print(df)
###Pipeline
# Parameters
svm_params = dict(C=1.0, kernel='linear', gamma='auto', probability=True)
vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)
# Supervised Pipeline with SVM
svm_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", SVC(**svm_params)),
]
)
# SelfTraining Pipeline
st_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", SelfTrainingClassifier(SVC(**svm_params), verbose=True)),
]
)
###Define a function for a classification report
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
print("Number of training samples:", len(X_train))
print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1)) #if x == 'NaN'
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print("y Train", y_train)
#print("y Predict",y_pred)
#print("y Test",y_test)
print(
"Micro-averaged F1 score on test set: %0.3f"
% f1_score(y_test, y_pred, average="micro")
)
print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred,zero_division=1))
print("\n\n")
###Split the data:
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2,
stratify=y_labeled, random_state=42)
### Self-Training Classifier on the Labeled Data
print("Supervised SVMClassifier on the labeled data:")
eval_and_print_metrics(svm_pipeline, X_train, y_train, X_test, y_test)
### Self-training on labeled and unlabelled data
test_indices = X_test.index
#print("TEST INDICES",test_indices)
# Exclude test data from X_labeled and y_labeled based on the identified indices
X_labeled_filtered = X_labeled.drop(index=test_indices, errors='ignore')
y_labeled_filtered = y_labeled.drop(index=test_indices, errors='ignore')
# Concatenate the filtered labeled data with the unlabeled data
X=X_combined = pd.concat([X_labeled_filtered, X_unlabeled])
y=y_combined = pd.concat([y_labeled_filtered, y_unlabeled])
# Print the concatenated dataframes X and y
print("Concatenated X:")
print(X)
print("\nConcatenated y:")
print(y)
# Print "Self Training Classifier on the labeled and unlabeled data:"
print("Self Training Classifier on the labeled and unlabeled data:")
# Evaluate and print metrics for the self-training classifier on the combined data
eval_and_print_metrics(st_pipeline, X_combined, y_combined, X_test, y_test)
###Creating Confusion Matrix and Heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
# Assuming st_pipeline is your trained Self-Training classifier
# and eval_and_print_metrics is a custom function used for evaluation
# We'll manually generate and plot the confusion matrix here
# Get predictions on the test set
y_pred = st_pipeline.predict(X_test)
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#: Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=True,
yticklabels=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
###Co Training using Random Forest and Logistic Regression
# Prepare labeled and unlabeled data
unlabeled_data = df[df['Emotions'] == 'NaN'][['Review Text']]
unlabeled_data['Emotions'] = -1
labeled_data = df[df['Emotions'].notna() & (df['Emotions'] != 'NaN')]
y_labeled = labeled_data['Emotions']
y_unlabeled = unlabeled_data['Emotions']
X_labeled = labeled_data['Review Text']
X_unlabeled = unlabeled_data['Review Text']
# Parameters
vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)
logistic_params = dict(C=1.0, max_iter=1000)
rf_params = dict(n_estimators=100, max_depth=5, random_state=42)
# Supervised Pipelines for Logistic Regression and Random Forest
logistic_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", LogisticRegression(**logistic_params)),
]
)
rf_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", RandomForestClassifier(**rf_params)),
]
)
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
print("Number of training samples:", len(X_train))
print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(
"Micro-averaged F1 score on test set: %0.3f"
% f1_score(y_test, y_pred, average="micro")
)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\n\n")
def co_training_step(logistic_clf, rf_clf, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
num_iterations=5):
for i in range(num_iterations):
print(f"\nIteration {i+1}/{num_iterations}...")
# Check if there are enough samples to continue training
if len(X_labeled) == 0 or len(X_unlabeled) == 0:
print("Stopping co-training as there are no samples left to process.")
break
# Train both classifiers on the labeled data
logistic_clf.fit(X_labeled, y_labeled)
rf_clf.fit(X_labeled, y_labeled)
# Get predictions and probabilities for the unlabeled data
logistic_preds = logistic_clf.predict(X_unlabeled)
logistic_probs = logistic_clf.predict_proba(X_unlabeled)
rf_preds = rf_clf.predict(X_unlabeled)
rf_probs = rf_clf.predict_proba(X_unlabeled)
# Identify high-confidence predictions
logistic_confident = (logistic_probs.max(axis=1) > 0.9)
rf_confident = (rf_probs.max(axis=1) > 0.9)
# Ensure there are confident predictions before proceeding
if logistic_confident.sum() == 0 and rf_confident.sum() == 0:
print("No confident predictions found, stopping co-training.")
break
# Add confident predictions to the labeled data
if logistic_confident.sum() > 0:
new_labels_logistic = logistic_preds[logistic_confident]
X_labeled = pd.concat([X_labeled, X_unlabeled[logistic_confident]])
y_labeled = pd.concat([y_labeled, pd.Series(new_labels_logistic)])
if rf_confident.sum() > 0:
new_labels_rf = rf_preds[rf_confident]
X_labeled = pd.concat([X_labeled, X_unlabeled[rf_confident]])
y_labeled = pd.concat([y_labeled, pd.Series(new_labels_rf)])
# Remove the newly labeled data from the unlabeled set
X_unlabeled = X_unlabeled[~logistic_confident & ~rf_confident]
y_unlabeled = y_unlabeled[~logistic_confident & ~rf_confident]
# Print the progress
print(f"Labeled data size: {len(X_labeled)}")
print(f"Unlabeled data size: {len(X_unlabeled)}")
return logistic_clf, rf_clf, X_labeled, y_labeled
# Split labeled data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2,
stratify=y_labeled, random_state=42)
# Clone the pipelines to ensure the co-training starts with the original models
logistic_pipeline_clone = clone(logistic_pipeline)
rf_pipeline_clone = clone(rf_pipeline)
# Perform co-training
logistic_clf, rf_clf, X_final, y_final = co_training_step(
logistic_pipeline_clone, rf_pipeline_clone, X_train, y_train, X_unlabeled, y_unlabeled
)
# Evaluate the classifiers
print("\nEvaluating Logistic Regression on test set:")
eval_and_print_metrics(logistic_clf, X_final, y_final, X_test, y_test)
print("\nEvaluating Random Forest on test set:")
eval_and_print_metrics(rf_clf, X_final, y_final, X_test, y_test)
####S3VM Model Construction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
# Prepare labeled and unlabeled data
unlabeled_data = df[df['Emotions'] == 'NaN'][['Review Text']]
unlabeled_data['Emotions'] = -1
labeled_data = df[df['Emotions'].notna() & (df['Emotions'] != 'NaN')]
y_labeled = labeled_data['Emotions']
y_unlabeled = unlabeled_data['Emotions']
X_labeled = labeled_data['Review Text']
X_unlabeled = unlabeled_data['Review Text']
# Parameters
vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)
svm_params = dict(C=1.0, kernel='linear', gamma='auto', probability=True)
# Create a pipeline for S3VM using SelfTrainingClassifier
s3vm_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", SelfTrainingClassifier(SVC(**svm_params), verbose=True, criterion='k_best',
k_best=10)),
]
)
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
print("Number of training samples:", len(X_train))
print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(
"Micro-averaged F1 score on test set: %0.3f"
% f1_score(y_test, y_pred, average="micro")
)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\n\n")
# Combine labeled and unlabeled data
X_combined = pd.concat([X_labeled, X_unlabeled])
y_combined = pd.concat([y_labeled, y_unlabeled])
# Split labeled data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2,
stratify=y_labeled, random_state=42)
# Train the S3VM model using the combined dataset
print("Training S3VM on labeled and unlabeled data:")
eval_and_print_metrics(s3vm_pipeline, X_combined, y_combined, X_test, y_test)
###plotting confusion matrix for S3VM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
print("Number of training samples:", len(X_train))
print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Printing the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print(
"Micro-averaged F1 score on test set: %0.3f"
% f1_score(y_test, y_pred, average="micro")
)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test),
yticklabels=set(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Combine labeled and unlabeled data
X_combined = pd.concat([X_labeled, X_unlabeled])
y_combined = pd.concat([y_labeled, y_unlabeled])
# Split labeled data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2,
stratify=y_labeled, random_state=42)
# Train the S3VM model using the combined dataset
print("Training S3VM on labeled and unlabeled data:")
eval_and_print_metrics(s3vm_pipeline, X_combined, y_combined, X_test, y_test)
###Label Spreading and Label Propagation - Graph Based Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Preprocess the 'Review Text' column
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['Review Text'].fillna(''))
# Convert the 'Emotions' column to a format suitable for semi-supervised learning
# Labeled data remains the same, and missing labels are filled with -1
y = df['Emotions'].factorize()[0] # Convert emotions to numerical labels
unlabeled_mask = df['Emotions'].isna()
y[unlabeled_mask] = -1 # Mark unlabeled data with -1
# Convert the sparse matrix to a dense format
X_dense = X.toarray()
# Define Label Spreading and Label Propagation models
label_spreading = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.8)
label_propagation = LabelPropagation(kernel='knn', n_neighbors=5)
# Fit the models on the dense data
label_spreading.fit(X_dense, y)
label_propagation.fit(X_dense, y)
# Predict the labels for the unlabeled data
spreading_predictions = label_spreading.transduction_[unlabeled_mask]
propagation_predictions = label_propagation.transduction_[unlabeled_mask]
# Get the unique labels from the original 'Emotions' column
unique_labels = df['Emotions'].factorize()[1]
# Convert predictions back to original emotion labels
spreading_emotions = unique_labels[spreading_predictions]
propagation_emotions = unique_labels[propagation_predictions]
# Add predictions back to the original dataframes
df_spreading = df.copy()
df_propagation = df.copy()
df_spreading.loc[unlabeled_mask, 'Emotions'] = spreading_emotions
df_propagation.loc[unlabeled_mask, 'Emotions'] = propagation_emotions
# Create classification reports and confusion matrices for both methods
y_true = y[~unlabeled_mask] # True labels for labeled data
# Label Spreading predictions
y_pred_spreading = label_spreading.transduction_[~unlabeled_mask]
report_spreading = classification_report(y_true, y_pred_spreading,
target_names=unique_labels)
cm_spreading = confusion_matrix(y_true, y_pred_spreading)
# Label Propagation predictions
y_pred_propagation = label_propagation.transduction_[~unlabeled_mask]
report_propagation = classification_report(y_true, y_pred_propagation,
target_names=unique_labels)
cm_propagation = confusion_matrix(y_true, y_pred_propagation)
# Print classification reports
print("Classification Report - Label Spreading")
print(report_spreading)
print("\nClassification Report - Label Propagation")
print(report_propagation)
# Plot the confusion matrices for both methods
# Label Spreading Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_spreading, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels,
yticklabels=unique_labels)
plt.title('Confusion Matrix - Label Spreading')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# Label Propagation Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_propagation, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels,
yticklabels=unique_labels)
plt.title('Confusion Matrix - Label Propagation')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
###Topic Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
# Preprocess the data
# We'll use CountVectorizer here,
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Review Text'].fillna(''))
# Apply LDA
n_topics = 5 # You can adjust this number to get more or fewer topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
# Analyze the Topics
# Get the words associated with each topic
def display_topics(model, feature_names, no_top_words):
for topic_idx, topic in enumerate(model.components_):
print(f"Topic {topic_idx + 1}:")
print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
print()
no_top_words = 10 # Number of top words to display for each topic
feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, no_top_words)
###Word Cloud Library Install
pip install wordcloud
###Generating Word Cloud
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#Preprocess the data
# We'll use CountVectorizer here, vectorizer = CountVectorizer(max_df=0.95, min_df=2,
stop_words='english')
X = vectorizer.fit_transform(df['Review Text'].fillna(''))
#Apply LDA
n_topics = 5 # You can adjust this number to get more or fewer topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
# Analyze the Topics and Create Word Clouds
def display_wordclouds(model, feature_names, no_top_words):
for topic_idx, topic in enumerate(model.components_):
print(f"Topic {topic_idx + 1}:")
# Generate word cloud
wordcloud = WordCloud(
background_color='white',
max_words=no_top_words,
contour_width=3,
contour_color='steelblue'
).generate_from_frequencies({feature_names[i]: topic[i] for i in
topic.argsort()[:-no_top_words - 1:-1]})
# Display the word cloud
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title(f"Word Cloud for Topic {topic_idx + 1}")
plt.show()
no_top_words = 10 # Number of top words to display for each topic
feature_names = vectorizer.get_feature_names_out()
display_wordclouds(lda, feature_names, no_top_words)
###Sentiment Analysis
from textblob import TextBlob
def analyze_sentiment(text):
blob = TextBlob(text)
polarity = blob.sentiment.polarity
if polarity > 0:
sentiment = "Positive"
elif polarity < 0:
sentiment = "Negative"
else:
sentiment = "Neutral"
return sentiment, polarity
# Correcting the way we pass the 'Review Text' column
text = df['Review Text'].fillna('') # Handle NaN values
# Applying the function to each review
df['Sentiment'], df['Polarity'] = zip(*text.apply(analyze_sentiment))
# Display the result
df[['Review Text', 'Sentiment', 'Polarity']].head()
###Creating bar plots for sentiment distribution
import matplotlib.pyplot as plt
# Calculate the frequency of each sentiment
sentiment_counts = df['Sentiment'].value_counts()
# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
# Add titles and labels
plt.title('Sentiment Analysis of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.xticks(rotation=45) # Rotate x-axis labels if necessary
# Show the plot
plt.show()
###Sentiment Distribution for Each Product Category
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
# Function to calculate sentiment using TextBlob
def get_sentiment_textblob(review):
analysis = TextBlob(review)
if analysis.sentiment.polarity > 0:
return 'Positive'
elif analysis.sentiment.polarity < 0:
return 'Negative'
else:
return 'Neutral'
# Apply sentiment analysis to the review text using TextBlob
df['Sentiment'] = df['Review Text'].fillna('').apply(get_sentiment_textblob)
# Group the data by the 'Class Name' (product category) and count sentiment occurrences
grouped_df = df.groupby('Class Name')['Sentiment'].value_counts().unstack().fillna(0)
# Define the correct color mapping for the sentiment categories
corrected_colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
# Generate a stacked bar chart for each product category with the correct colors
grouped_df.plot(kind='bar', stacked=True, figsize=(12, 8), color=[corrected_colors[col] for col in
grouped_df.columns])
# Add titles and labels
plt.title('Sentiment Analysis by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Frequency of Sentiments')
plt.xticks(rotation=45)
# Show the plot
plt.show()
###Alternate way of creating a combined co-training model report for RF + LR
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.base import clone
import pandas as pd
import numpy as np
# Assuming df is already loaded as your DataFrame
# Prepare labeled and unlabeled data
unlabeled_data = df[df['Emotions'] == 'NaN'][['Review Text']]
unlabeled_data['Emotions'] = -1
labeled_data = df[df['Emotions'].notna() & (df['Emotions'] != 'NaN')]
y_labeled = labeled_data['Emotions']
y_unlabeled = unlabeled_data['Emotions']
X_labeled = labeled_data['Review Text']
X_unlabeled = unlabeled_data['Review Text']
# Parameters
vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)
logistic_params = dict(C=1.0, max_iter=1000)
rf_params = dict(n_estimators=100, max_depth=5, random_state=42)
# Supervised Pipelines for Logistic Regression and Random Forest
logistic_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", LogisticRegression(**logistic_params)),
]
)
rf_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", RandomForestClassifier(**rf_params)),
]
)
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
print("Number of training samples:", len(X_train))
print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(
"Micro-averaged F1 score on test set: %0.3f"
% f1_score(y_test, y_pred, average="micro")
)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\n\n")
def co_training_step(logistic_clf, rf_clf, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
num_iterations=5):
for i in range(num_iterations):
print(f"\nIteration {i+1}/{num_iterations}...")
# Check if there are enough samples to continue training
if len(X_labeled) == 0 or len(X_unlabeled) == 0:
print("Stopping co-training as there are no samples left to process.")
break
# Train both classifiers on the labeled data
logistic_clf.fit(X_labeled, y_labeled)
rf_clf.fit(X_labeled, y_labeled)
# Get predictions and probabilities for the unlabeled data
logistic_preds = logistic_clf.predict(X_unlabeled)
logistic_probs = logistic_clf.predict_proba(X_unlabeled)
rf_preds = rf_clf.predict(X_unlabeled)
rf_probs = rf_clf.predict_proba(X_unlabeled)
# Identify high-confidence predictions
logistic_confident = (logistic_probs.max(axis=1) > 0.9)
rf_confident = (rf_probs.max(axis=1) > 0.9)
# Ensure there are confident predictions before proceeding
if logistic_confident.sum() == 0 and rf_confident.sum() == 0:
print("No confident predictions found, stopping co-training.")
break
# Add confident predictions to the labeled data
if logistic_confident.sum() > 0:
new_labels_logistic = logistic_preds[logistic_confident]
X_labeled = pd.concat([X_labeled, X_unlabeled[logistic_confident]])
y_labeled = pd.concat([y_labeled, pd.Series(new_labels_logistic)])
if rf_confident.sum() > 0:
new_labels_rf = rf_preds[rf_confident]
X_labeled = pd.concat([X_labeled, X_unlabeled[rf_confident]])
y_labeled = pd.concat([y_labeled, pd.Series(new_labels_rf)])
# Remove the newly labeled data from the unlabeled set
X_unlabeled = X_unlabeled[~logistic_confident & ~rf_confident]
y_unlabeled = y_unlabeled[~logistic_confident & ~rf_confident]
# Print the progress
print(f"Labeled data size: {len(X_labeled)}")
print(f"Unlabeled data size: {len(X_unlabeled)}")
return logistic_clf, rf_clf, X_labeled, y_labeled
# Split labeled data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2,
stratify=y_labeled, random_state=42)
# Clone the pipelines to ensure the co-training starts with the original models
logistic_pipeline_clone = clone(logistic_pipeline)
rf_pipeline_clone = clone(rf_pipeline)
# Perform co-training
logistic_clf, rf_clf, X_final, y_final = co_training_step(
logistic_pipeline_clone, rf_pipeline_clone, X_train, y_train, X_unlabeled, y_unlabeled
)
# Combine predictions from both classifiers
def combined_predictions(logistic_clf, rf_clf, X):
# Fit vectorizers within the pipeline on labeled data before predictions
logistic_clf.fit(X_labeled, y_labeled)
rf_clf.fit(X_labeled, y_labeled)
# Get predictions and probabilities
logistic_probs = logistic_clf.predict_proba(X)
rf_probs = rf_clf.predict_proba(X)
# Use majority voting or highest confidence approach
combined_preds = []
for i in range(len(X)):
if np.max(logistic_probs[i]) > np.max(rf_probs[i]):
combined_preds.append(logistic_clf.predict(X[i:i+1])[0])
else:
combined_preds.append(rf_clf.predict(X[i:i+1])[0])
return np.array(combined_preds)
# Evaluate combined predictions on the test set
combined_preds = combined_predictions(logistic_clf, rf_clf, X_test)
# Print the combined classification report
print("\nCombined Co-Training Model Evaluation:")
print("Micro-averaged F1 score on test set: %0.3f" % f1_score(y_test, combined_preds,
average="micro"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, combined_preds))
print("\nClassification Report:\n", classification_report(y_test, combined_preds,
zero_division=1))
####Co Training model using Gradient Boosting + S3VM
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
# Prepare labeled and unlabeled data
unlabeled_data = df[df['Emotions'] == 'NaN'][['Review Text']]
unlabeled_data['Emotions'] = -1
labeled_data = df[df['Emotions'].notna() & (df['Emotions'] != 'NaN')]
y_labeled = labeled_data['Emotions']
y_unlabeled = unlabeled_data['Emotions']
X_labeled = labeled_data['Review Text']
X_unlabeled = unlabeled_data['Review Text']
# Ensure consistent label types by encoding all labels as integers
label_encoder = LabelEncoder()
y_labeled = label_encoder.fit_transform(y_labeled)
y_unlabeled = np.full_like(X_unlabeled, -1) # Ensure unlabeled data has consistent -1 labels
# Retrieve class names for proper output in evaluation
class_names = label_encoder.classes_
# Parameters
vectorizer_params = dict(ngram_range=(1, 2), min_df=1, max_df=0.8)
svm_params = dict(C=1.0, kernel='linear', probability=True)
gbc_params = dict(n_estimators=100, max_depth=3, random_state=42)
# Supervised Pipelines for SVM and Gradient Boosting
svm_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", SVC(**svm_params)),
]
)
gbc_pipeline = Pipeline(
[
("vect", CountVectorizer(**vectorizer_params)),
("tfidf", TfidfTransformer()),
("clf", GradientBoostingClassifier(**gbc_params)),
]
)
def eval_combined_metrics(svm_clf, gbc_clf, X_test, y_test, class_names):
try:
# Predict probabilities with both classifiers
svm_probs = svm_clf.predict_proba(X_test)
gbc_probs = gbc_clf.predict_proba(X_test)
# Combine probabilities by averaging (ensemble approach)
combined_probs = (svm_probs + gbc_probs) / 2
# Predict the final labels based on the combined probabilities
combined_preds = np.argmax(combined_probs, axis=1)
# Evaluate the combined predictions
print("\nCombined Model Evaluation:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, combined_preds))
print("\nClassification Report:\n", classification_report(y_test, combined_preds,
target_names=class_names, zero_division=1))
print("\nMicro-averaged F1 score on test set: %0.3f" % f1_score(y_test, combined_preds,
average="micro"))
except NotFittedError as e:
print(f"Model fitting error: {e}")
def co_training_step(svm_clf, gbc_clf, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
num_iterations=5):
for i in range(num_iterations):
print(f"\nIteration {i+1}/{num_iterations}...")
# Check if there are enough samples to continue training
if len(X_labeled) == 0 or len(X_unlabeled) == 0:
print("Stopping co-training as there are no samples left to process.")
break
# Train both classifiers on the labeled data
svm_clf.fit(X_labeled, y_labeled)
gbc_clf.fit(X_labeled, y_labeled)
# Get predictions and probabilities for the unlabeled data
svm_preds = svm_clf.predict(X_unlabeled)
svm_probs = svm_clf.predict_proba(X_unlabeled)
gbc_preds = gbc_clf.predict(X_unlabeled)
gbc_probs = gbc_clf.predict_proba(X_unlabeled)
# Identify high-confidence predictions
svm_confident = (svm_probs.max(axis=1) > 0.9)
gbc_confident = (gbc_probs.max(axis=1) > 0.9)
# Ensure there are confident predictions before proceeding
if svm_confident.sum() == 0 and gbc_confident.sum() == 0:
print("No confident predictions found, stopping co-training.")
break
# Add confident predictions to the labeled data
if svm_confident.sum() > 0:
new_labels_svm = svm_preds[svm_confident]
X_labeled = pd.concat([X_labeled, X_unlabeled[svm_confident]])
y_labeled = np.concatenate([y_labeled, new_labels_svm])
if gbc_confident.sum() > 0:
new_labels_gbc = gbc_preds[gbc_confident]
X_labeled = pd.concat([X_labeled, X_unlabeled[gbc_confident]])
y_labeled = np.concatenate([y_labeled, new_labels_gbc])
# Remove the newly labeled data from the unlabeled set
X_unlabeled = X_unlabeled[~svm_confident & ~gbc_confident]
y_unlabeled = y_unlabeled[~svm_confident & ~gbc_confident]
# Print the progress
print(f"Labeled data size: {len(X_labeled)}")
print(f"Unlabeled data size: {len(X_unlabeled)}")
# If co-training stops early, ensure the pipelines are still fitted on the current labeled data
if len(X_labeled) > 0:
svm_clf.fit(X_labeled, y_labeled)
gbc_clf.fit(X_labeled, y_labeled)
return svm_clf, gbc_clf, X_labeled, y_labeled
# Split labeled data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2,
stratify=y_labeled, random_state=42)
# Clone the pipelines to ensure the co-training starts with the original models
svm_pipeline_clone = clone(svm_pipeline)
gbc_pipeline_clone = clone(gbc_pipeline)
# Perform co-training
svm_clf, gbc_clf, X_final, y_final = co_training_step(
svm_pipeline_clone, gbc_pipeline_clone, X_train, y_train, X_unlabeled, y_unlabeled
)
# Evaluate combined predictions with emotion class names
eval_combined_metrics(svm_clf, gbc_clf, X_test, y_test, class_names)
###Plotting the Confusion Matrix in a Heatmap
import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Create a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
yticklabels=class_names)
# Add titles and labels
plt.title(title)
plt.xlabel('Predicted')
plt.ylabel('True')
# Display the plot
plt.show()
# Example usage with the combined model predictions
try:
# Evaluate combined predictions with emotion class names
eval_combined_metrics(svm_clf, gbc_clf, X_test, y_test, class_names)
# Combined predictions (re-run the function to get combined predictions)
svm_probs = svm_clf.predict_proba(X_test)
gbc_probs = gbc_clf.predict_proba(X_test)
combined_probs = (svm_probs + gbc_probs) / 2
combined_preds = np.argmax(combined_probs, axis=1)
# Plotting the confusion matrix
plot_confusion_matrix(y_test, combined_preds, class_names, title="Confusion Matrix -
Combined Model")
except NotFittedError as e:
print(f"Model fitting error: {e}")
###Grouping topics from topic modeling with sentiments extracted
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
# Load your DataFrame and preprocess the text data
# Vectorize the text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Review Text'].fillna(''))
# Perform Topic Modeling with LDA
n_topics = 5 # You can adjust the number of topics as needed
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
# Get the dominant topic for each document
topic_assignments = lda.transform(X).argmax(axis=1)
# Add the topic assignments to the DataFrame
df['Topic'] = topic_assignments
# Extract sentiments
#Using the ones already generated in the previous steps
# Group Topics by Sentiment
# Create a cross-tabulation of sentiments and topics
topic_sentiment_counts = pd.crosstab(df['Topic'], df['Sentiment'])
# Plot the Topics against Sentiments
plt.figure(figsize=(10, 6))
sns.heatmap(topic_sentiment_counts, annot=True, fmt='d', cmap='Blues')
plt.title('Distribution of Topics across Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Topic')
plt.show()
# Using the barplot to show the distribution
topic_sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
plt.title('Distribution of Topics across Sentiments')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()
###Topic distribution for emotions classes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
# Preprocess the data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Review Text'].fillna(''))
# Apply LDA
n_topics = 5 # Adjust the number of topics as needed
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
# Get the topic distribution for each document
topic_distribution = lda.transform(X)
# Assign each review its dominant topic
df['Dominant Topic'] = np.argmax(topic_distribution, axis=1)
# Categorize topics by emotions
# Since there is an 'Emotion' column in the dataframe
emotion_topic_distribution = df.groupby('Emotions')['Dominant
Topic'].value_counts(normalize=True).unstack().fillna(0)
# Display the distribution of topics for each emotion
print(emotion_topic_distribution)
###Plotting a Barchart
import matplotlib.pyplot as plt
import seaborn as sns
# Plot proportions
plt.figure(figsize=(12, 6))
emotion_topic_proportions.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
plt.title('Proportion of Topics Across Emotions')
plt.xlabel('Emotions')
plt.ylabel('Proportion of Topic')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
###Creating a Gantt Chart for the Logbook
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
# Data for the Gantt chart based on the logbook entries
tasks = [
"Project Initiation and Data Collection",
"Data Cleaning and Preprocessing",
"Exploratory Data Analysis (EDA)",
"Model Development - Supervised and Semi-Supervised Learning",
"Sentiment Analysis and Emotion Extraction Using Transformers",
"Topic Modeling",
"Model Evaluation and Comparison",
"Finalization and Reflection"
]
# Start and end dates for each task
start_dates = pd.to_datetime([
"2024-07-15", "2024-07-22", "2024-07-29",
"2024-08-05", "2024-08-12", "2024-08-19",
"2024-08-26", "2024-09-02"
])
end_dates = pd.to_datetime([
"2024-07-21", "2024-07-28", "2024-08-04",
"2024-08-11", "2024-08-18", "2024-08-25",
"2024-09-01", "2024-09-07"
])
# Creating a DataFrame for plotting
df = pd.DataFrame({
'Task': tasks,
'Start': start_dates,
'End': end_dates
})
# Plotting the Gantt chart
plt.figure(figsize=(12, 6))
for i, task in enumerate(df.itertuples()):
plt.barh(task.Task, (task.End - task.Start).days, left=task.Start, color='skyblue',
edgecolor='black')
plt.xlabel('Date')
plt.ylabel('Tasks')
plt.title('Gantt Chart for Project Logbook')
plt.grid(axis='x', linestyle='--', alpha=0.7)
# Formatting the x-axis to display dates properly
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
