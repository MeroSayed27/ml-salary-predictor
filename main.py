import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Split features (X) and target (y)
X = data[["YearsExperience"]]
y = data["Salary"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
prediction = model.predict(pd.DataFrame([[5]], columns=["YearsExperience"]))

print("Predicted salary for 5 years experience:", prediction[0])