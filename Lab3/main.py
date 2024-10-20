import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


def download():
    df = pd.read_csv('titanic.csv')
    # print(df.head())
    X, y = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df['Survived']
    return X, y

def analysisNan(X):
    missing = X.isnull().sum()
    print(missing[missing > 0].index.tolist())
    for i in missing[missing > 0].index.tolist():
        if pd.api.types.is_numeric_dtype(X[i]):
            # print(X[i].dtype)
            X[i] = pd.Series(X[i].fillna(X[i].mean(skipna=True)))
            # print(X[i].fillna(X[i].mean(skipna=True)))
            # print(X[i])
        else:
            # print(X[i].dtype)
            # print(X[i].iloc[61])
            # print(X[i].index[X[i].isnull()])
            X[i] = pd.Series(X[i].fillna("None"))
            # print(X[i].iloc[61])
    # print(X.info())
    return X

def srez(X, y):
    X_train, X_test, Y_train, Y_test = X.iloc[:int(len(X) * 0.7)], X.iloc[int(len(X) * 0.7):], y.iloc[:int(len(y) * 0.7)], y.iloc[int(len(y) * 0.7):]
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    return X_train, X_test, Y_train, Y_test

def mashtab(X_train, Y_train, X_test):
    scaler = StandardScaler()
    print(X_train)
    scaler.fit(X=X_train, y=Y_train)
    scaler_train = scaler.transform(X=X_train)
    scaler_test = scaler.transform(X=X_test)
    # for i in X_train.columns:
    #     print(i)
    #     # scaler.fit(X=X_train[i], y=Y_train)
    print(scaler)
    print(scaler_train)
    print(scaler_test)

def regression(X_train, Y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    print(f"Коэффициент w{logreg.coef_},\nКоэффициент w0 {logreg.intercept_}")
    return logreg

def sortirovka(logreg):
    sorted_weights = sorted(zip(logreg.coef_.ravel(), X.columns), reverse=True)
    print(sorted_weights)
    df = pd.DataFrame(sorted_weights)
    df = df.rename(columns={0: "weights", 1: "features"})
    print(df)
    plt.barh(df["features"], df["weights"])
    plt.show()
    return df

def logistic_function(x):
    try:
        return (1 / (1 + np.exp(-x)))
    except Exception as e:
        print(e)
    # return np.dot(X_test, logreg.coef_.T) + logreg.intercept_

def distribution(Y_test, a, b, bins):
    # Построение гистограммы
    plt.hist(Y_test, bins=bins, edgecolor='black', lw=2)

    # Добавление названий осей
    plt.xlabel(f'{a}')
    plt.ylabel('Частота')
    plt.title(f'{b}')

    # Показать график
    plt.show()



if __name__ == "__main__":
    X, y = download()
    X = analysisNan(X)
    # print(X.head())
    # print(y.head())
    # missing = X.isnull().sum()
    # print(missing[missing > 0].index.tolist())
    # for i in missing[missing > 0].index.tolist():
    #     if pd.api.types.is_numeric_dtype(X[i]):
    #         # print(X[i].dtype)
    #         X[i] = pd.Series(X[i].fillna(X[i].mean(skipna=True)))
    #         # print(X[i].fillna(X[i].mean(skipna=True)))
    #         # print(X[i])
    #     else:
    #         # print(X[i].dtype)
    #         # print(X[i].iloc[61])
    #         # print(X[i].index[X[i].isnull()])
    #         X[i] = pd.Series(X[i].fillna("None"))
    #         # print(X[i].iloc[61])
    print(X.info())
    print(X['Sex'].unique())
    X['Sex'] = X['Sex'].map(lambda x: 1 if x == 'male' else 0)
    print(X['Sex'].unique())
    print(X['Embarked'].unique())
    X = pd.get_dummies(X)
    print(X)
    X_train, X_test, Y_train, Y_test = srez(X, y)
    mashtab(X_train, Y_train, X_test)
    logreg = regression(X_train, Y_train)
    df = sortirovka(logreg)
    # print(np.dot(X_test, logreg.coef_.T) + logreg.intercept_)
    pred_prob = np.ravel(logistic_function(np.asarray(np.dot(X_test, logreg.coef_.T) + logreg.intercept_, dtype=float)))
    # print(pred_prob)
    pred_predict_proba = logreg.predict_proba(X_test)[:, 1]
    print(f"pred_prob равна pred_predict_proba? {np.all([pred_prob, pred_predict_proba])}")
    pred_bin = (pred_prob >= 0.5).astype(int)
    print(f'Точность predict_proba: {np.mean(pred_bin == Y_test[:].values)}')
    pred_class = logreg.predict(X_test)
    print(f'Точность predict: {np.mean(pred_class == Y_test[:].values)}')
    distribution(Y_test, "Значение", "Распределение целевого значения", 2)
    distribution(pred_prob, "Значение", "Распределение предсказываемого значения", 10)
    selected_indices = np.where((pred_prob >= 0.75) & (pred_prob <= 0.85))[0]
    pred_selected = logreg.predict(X_test.iloc[selected_indices])
    print(f'Точность на отрезке от 0.75 до 0.85: {np.mean(pred_selected == Y_test.iloc[selected_indices].values)}')



