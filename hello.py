import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense

# Read data
data = pd.read_csv('titanic3.csv')

# Fill up unknown data
data.replace('?', np.nan, inplace= True)
data = data.astype({"age": np.float64, "fare": np.float64})

#Plot Data
# fig, axs = plt.subplots(ncols=5, figsize=(30,5))
# sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
# sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
# sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
# sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
# sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])

# determine relationships
data.replace({'male': 1, 'female': 0}, inplace=True)
data.corr().abs()[["survived"]]
data['relatives'] = data.apply (lambda row: int((row['sibsp'] + row['parch']) > 0), axis=1)
data.corr().abs()[["survived"]]

#drop NA
data = data[['sex', 'pclass','age','relatives','fare','survived']].dropna()

# train
x_train, x_test, y_train, y_test = train_test_split(data[['sex','pclass','age','relatives','fare']], data.survived, test_size=0.2, random_state=0)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Fit
model = GaussianNB()
model.fit(X_train, y_train)

# test prediction
predict_test = model.predict(X_test)
print(metrics.accuracy_score(y_test, predict_test))

# another model
model = Sequential()
model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.summary()

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=200)

y_pred = np.argmax(model.predict(X_test), axis=-1)
print(metrics.accuracy_score(y_test, y_pred))

