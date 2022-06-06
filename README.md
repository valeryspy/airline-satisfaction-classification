# Airline Satisfaction Classification

Simple machine learning `classifier` for customer satisfaction with Python and Scikit.

Data taken from: [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

Decision Tree classifying airline satisfaction based on the initial model:
```python
plt.figure(figsize=(40,20))
tree.plot_tree(clf,
          filled=True,
          rounded=True,
          class_names=["Not Satidfied", "Satidfied"])
plt.show()
```
![download (3)](https://user-images.githubusercontent.com/103786666/172252451-ed424a40-8e7c-4911-a96a-df03d11bf129.png)

Validating the best model on the entire dataset:
```python
X=df2.iloc[:,:-1] # extract dataframe to predictor variables
Y=df2.iloc[:,-1:] # extract dataframe to response variables
   
  # validating the model
y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = confusion_matrix(Y, y_pred, labels=clf.classes_, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                        display_labels=clf.classes_)
disp.plot()
plt.title("Confusion matrix, fraction")
plt.show()
```

![download (4)](https://user-images.githubusercontent.com/103786666/172253375-51946d7e-908a-4575-b357-d0a41dc80610.png)



