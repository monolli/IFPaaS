'''
This is a minimalistic Iris flower classifier.
'''

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def main() -> None:
    """Executes the program.

    Returns
    -------
    None
        Description of returned object.

    """
    # Load the dataset from the sklearn lib.
    iris = load_iris()
    # Break it in two separate dataset, data and target.
    X = iris.data
    y = iris.target
    # Split the dataset in two, training and test.
    # The classes are balanced and with a reasonable population, so it is ok to shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=997, shuffle=True)
    # Define a random forest classifier
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=997, verbose=1)
    # Fit the model
    clf.fit(X_train, y_train)
    # Predict using the test datasets
    Pred = clf.predict(X_test)
    print(Pred)
    print(iris.target_names)
    print([iris.target_names[x] for x in Pred])
    # Print the confusion matrix
    print(confusion_matrix(y_test, Pred))
    # Print accuracy
    print("Accuracy: ", round(accuracy_score(y_test, Pred), 2))


if __name__ == "__main__":
    main()
