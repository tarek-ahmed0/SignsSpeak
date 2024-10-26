# Model used libraries
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset from a pickle file
with open('dataset.pickle', 'rb') as dataset:
    loaded_dataset = pickle.load(dataset)
    data = np.asarray(loaded_dataset['data'])
    labels = np.asarray(loaded_dataset['labels'])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size = 0.2, 
    shuffle = True,  
    stratify = labels
)

# Train a RandomForestClassifier
classifier_model = RandomForestClassifier()
classifier_model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = classifier_model.predict(x_test)

# Calculate and print the accuracy score
score = accuracy_score(y_test, y_predict)  # Corrected order here
print(f"Model Accuracy Score: {score * 100:.2f}%")


# Model Saving 
saved_model = open('model.pickle', 'wb')
pickle.dump({
    'model' : classifier_model,
}, saved_model)
saved_model.close()