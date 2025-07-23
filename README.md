DECISION-TREE-IMPLEMENTATION
COMPANY : CODTECH IT SOLUTIONS

NAME : Mule Lakshmi Narayanamma

INTERN ID : CT04DH236

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

DESCRIPTION : This little project walks you through building a Decision Tree Classifier with the Iris flower dataset in Python, using familiar data science libraries. The goal is to train a simple model that sorts Iris samples by their sepal and petal measurements: sepal length, sepal width, petal length, and petal width.

A Decision Tree is an easy-to-grasp yet powerful method for sorting data or even making numeric forecasts. Here we stick to the sorting part. The tree learns which measurements group together and, in the end, names each sample with its flower species. It does this by slicing the dataset at feature value thresholds, jotting down the rules, and handing out a final label.

To keep things tidy, we rely on a handful of trusted Python tools:

Pandas lets us load the CSV and keep it in a neat DataFrame that behaves like a spreadsheet.

Scikit-learn, often called sklearn, supplies functions for splitting the set, training the tree, and checking how well it does; plus, it comes with the Iris data right out of the box.

Matplotlib serves up charts that let us peek inside the tree structure and spot the decision boundaries.

All these libraries are free, open-source, and show up in just about every machine-learning notebook you find online.

The project was built in Visual Studio Code (VSCode), a lightweight, free code editor with helpful extensions that simplify debugging and file management. provided code can also works on other platforms like Jupyter Notebook , Pycharm etc.

The Iris dataset is a classic example often pulled into early tutorials on classification. Inside it you will find

150 flowers, each a separate observation,

4 measurements: sepal length, sepal width, petal length, petal width,

3 species, each represented by 50 samples-Setosa, Versicolor, Virginica.

Model Process and Evaluation
steps involved are following :

The data was loaded directly with load_iris() from the sklearn library.

We split the set with train_test_split, keeping 60 percent for training and 40 percent for testing, while stratifying to preserve class ratios.

A DecisionTreeClassifier was fitted to the training portion.

Predictions were then generated for the hold-out test set.

Performance was reviewed through accuracy, a confusion matrix, and the full classification report, revealing strengths and weaknesses in each species.

Finally the tree itself was drawn with plot_tree(), clearly showing the rules that govern each make-or-break split.

Decision trees are used to help people and systems in making intelligent decisions in a variety of real-world scenarios, not just in computer programs or school projects. Here are a few basic examples:

Medical care To determine what illness a patient may have, doctors use decision trees. The system may indicate potential health issues based on symptoms like fever, cough, or pain. This facilitates quicker and more informed decision-making for physicians.

Banking Decision trees are used by banks to determine whether a person is a good fit when they apply for a credit card or loan. To determine whether or not to approve someone, the system considers factors like income, credit history, and prior behavior.

Decision trees are used in retail (shopping) establishments and internet retailers.

OUTPUT :
