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


