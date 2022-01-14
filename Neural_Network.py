import itertools as it
import tensorflow as ts
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
import numpy as np


# define baseline model
def baseline_model(features, classes):
    # create model
    model = Sequential()

    # Rectified Linear Unit Activation Function
    model.add(Dense(features * 2, input_dim=features, activation='sigmoid'))
    model.add(Dense(features * 2, activation='sigmoid'))  # Softmax for multi-class classification
    model.add(Dense(classes, activation='softmax'))  # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    seed = 42069
    np.random.seed(seed)
    ts.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


    df = pd.read_pickle("DataFrame.pkl")

    # for col in df.columns:
    # print(col)

    le = LabelEncoder()
    Y = le.fit_transform(df["experiment"])
    Y = np.sort(np.array(Y))
    X = np.array([df["x"], df["y"], df["z"]])
    X = np.transpose(X)

    val = np.array([])
    for lst in X:
        toms = list(it.chain.from_iterable([lst[0], lst[1], lst[2]]))
        # print(np.shape(toms))

        val = np.append(val, toms)
    val.shape = (1600, 300)
    X = val

    print(f"Shape of X: {np.shape(X)}")
    print(f"Datatype X: {X.dtype}")
    print(f"Shape of Y: {np.shape(Y)}")
    print(f"Datatype Y: {Y.dtype}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=seed)

    samples, features = X_train.shape
    classes = np.unique(Y_train).size

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    print(samples, features, classes)

    yhot = np_utils.to_categorical(Y)
    yhot_train = np_utils.to_categorical(Y_train)
    yhot_test = np_utils.to_categorical(Y_test)

    cmodel = KerasClassifier(build_fn=baseline_model, epochs=2000, batch_size=100, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)

    result = cross_val_score(cmodel, X, yhot, cv=kfold)
    print("Result: %.2f%% (%.2f%%)" % (result.mean() * 100, result.std() * 100))

    model = baseline_model(features, classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, yhot_train, validation_split=0.33,
                        epochs=200, batch_size=100, verbose=0)

    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())  # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()  # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, yhot_test)
    print('Accuracy from evaluate: %.2f' % (accuracy * 100))

    predict_x = model.predict(X_test)
    pred = np.argmax(predict_x, axis=1)
    print(f'Prediction Accuracy: {(pred == Y_test).mean() * 100:f}')





if __name__ == "__main__":
    main()
