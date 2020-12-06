#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease using Machine Learning

# In[1]:


import numpy as np # np is short for numpy
import pandas as pd # pandas is so commonly used, it's shortened to pd
import matplotlib.pyplot as plt
import seaborn as sns # seaborn gets shortened to sns

# to make plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## step 1: Load Data

# In[2]:


df = pd.read_csv("heart-disease.csv") # 'DataFrame' shortened to 'df'


# In[3]:


df.head()


# In[4]:


# count samples with positive (1) and negative (0) 'target' (output) values in the dataset
df.target.value_counts()


# In[5]:


# Normalize value counts
df.target.value_counts(normalize=True)


# In[6]:


# Plot the value counts with a bar graph
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);


# In[7]:


df.sex.value_counts()


# In[8]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[9]:


# Create a plot
pd.crosstab(df.target, df.sex).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);


# In[10]:


# Create a plot
pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])

# Add some attributes to it
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0); # keep the labels on the x-axis vertical


# In[11]:


# Create another figure
plt.figure(figsize=(10,6))

# Start with positve examples
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="salmon") # define it as a scatter figure

# Now for negative examples, we want them on the same plot, so we call plt again
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="lightblue") # axis always come as (x, y)

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");


# In[12]:


df.age.plot.hist();


# In[13]:


pd.crosstab(df.cp, df.target)


# In[14]:


# Create a new crosstab and base plot
pd.crosstab(df.cp, df.target).plot(kind="bar", 
                                   figsize=(10,6), 
                                   color=["lightblue", "salmon"])

# Add attributes to the plot to make it more readable
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation = 0);


# In[15]:


# Finding the correlation between independent variables
corr_matrix = df.corr()
corr_matrix 


# In[16]:


# plotting
corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");


# ## 5. Modeling

# In[17]:


df.head()


# In[18]:


# separating features and labels
X = df.drop("target", axis=1)

# Target variable
y = df.target.values


# In[19]:


X.head()


# In[20]:


# Targets
y


# In[21]:


# Random seed for reproducibility
np.random.seed(42)

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[22]:


X_train.head()


# In[23]:


y_train, len(y_train)


# ### Model choices
# 
# using the following models and comparing their results.
# 
# 1. Logistic Regression
# 2. K-Nearest Neighbors
# 3. RandomForest

# In[24]:


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[25]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores


# ## Model Comparison
# 
# Since we've saved our models scores to a dictionary, we can plot them by first converting them to a DataFrame.

# In[26]:


model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();


# ### Hand-tuning KNeighborsClassifier (K-Nearest Neighbors or KNN)
# 
# Hand tuning number of neighbours, main hyperparameter for the K-Nearest Neighbors (KNN) algorithm. The default is 5 (`n_neigbors=5`).

# In[27]:


# Create a list of train scores
train_scores = []

# Create a list of test scores
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 21) # 1 to 20

# Setup algorithm
knn = KNeighborsClassifier()

# Loop through different neighbors values
for i in neighbors:
    knn.set_params(n_neighbors = i) # set neighbors value
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores
    test_scores.append(knn.score(X_test, y_test))


# In[28]:


train_scores


# In[29]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ### Tuning Logistic-regression and Random forest models

# In[30]:


# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# In[31]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model
rs_log_reg.fit(X_train, y_train);


# In[32]:


rs_log_reg.best_params_


# In[33]:


rs_log_reg.score(X_test, y_test)


# In[34]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);


# In[35]:


# Find the best parameters
rs_rf.best_params_


# In[36]:


# Evaluate the randomized search random forest model
rs_rf.score(X_test, y_test)


# ### Tuning a model with GridSearchCV

# In[37]:


# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);


# In[38]:


# Check the best parameters
gs_log_reg.best_params_


# In[39]:


# Evaluate the model
gs_log_reg.score(X_test, y_test)


# ## Evaluating Metrics for a classification model, beyond accuracy
# 
# We explore:
# * ROC curve and AUC score - [`plot_roc_curve()`]
# * Confusion matrix - [`confusion_matrix()`]
# * Classification report - [`classification_report()`]
# * Precision - [`precision_score()`]
# * Recall - [`recall_score()`]
# * F1-score - [`f1_score()`]

# In[40]:


# Make preidctions on test data
y_preds = gs_log_reg.predict(X_test)


# In[41]:


y_preds


# In[42]:


y_test


# In[43]:


# Import ROC curve function from metrics module
from sklearn.metrics import plot_roc_curve

# Plot ROC curve and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test);


# In[44]:


# Display confusion matrix
print(confusion_matrix(y_test, y_preds))


# In[45]:


# Import Seaborn
import seaborn as sns
sns.set(font_scale=1.5) # Increase font size

def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    
plot_conf_mat(y_test, y_preds)


# In[46]:


# Show classification report
print(classification_report(y_test, y_preds))


# In[47]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[48]:


# Import cross_val_score
from sklearn.model_selection import cross_val_score

# Instantiate best model with best hyperparameters (found with GridSearchCV)
clf = LogisticRegression(C=0.23357214690901212,
                         solver="liblinear")


# In[49]:


# Cross-validated accuracy score
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5, # 5-fold cross-validation
                         scoring="accuracy") # accuracy as scoring
cv_acc


# Since there are 5 metrics here, we'll take the average.

# In[50]:


cv_acc = np.mean(cv_acc)
cv_acc


# Now we'll do the same for other classification metrics.

# In[51]:


# Cross-validated precision score
cv_precision = np.mean(cross_val_score(clf,
                                       X,
                                       y,
                                       cv=5, # 5-fold cross-validation
                                       scoring="precision")) # precision as scoring
cv_precision


# In[52]:


# Cross-validated recall score
cv_recall = np.mean(cross_val_score(clf,
                                    X,
                                    y,
                                    cv=5, # 5-fold cross-validation
                                    scoring="recall")) # recall as scoring
cv_recall


# In[53]:


# Cross-validated F1 score
cv_f1 = np.mean(cross_val_score(clf,
                                X,
                                y,
                                cv=5, # 5-fold cross-validation
                                scoring="f1")) # f1 as scoring
cv_f1


# In[54]:


# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_precision,
                            "Recall": cv_recall,
                            "F1": cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);


# In[55]:


# Fit an instance of LogisticRegression (taken from above)
clf.fit(X_train, y_train);


# In[56]:


# Check coef_
clf.coef_


# In[57]:


# Match features to columns
features_dict = dict(zip(df.columns, list(clf.coef_[0])))
features_dict


# In[58]:


# Visualize feature importance
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False);


# In[59]:


pd.crosstab(df["sex"], df["target"])


# In[60]:


# Contrast slope (positive coefficient) with target
pd.crosstab(df["slope"], df["target"])

