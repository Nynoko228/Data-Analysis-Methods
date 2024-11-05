import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

def make_seed():
    plt.rcParams['figure.figsize'] = (11, 6.5)
    np.random.seed(13)
    n = 500
    X = np.zeros(shape=(n, 2))
    X[:, 0] = np.linspace(-5, 5, 500)
    X[:, 1] = X[:, 0] + 0.5 * np.random.normal(size=n)
    y = (X[:, 1] > X[:, 0]).astype(int)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap='winter')
    plt.show()
    return X, y

def logreg(X_train, y_train, X_test):
    lr = LogisticRegression(random_state=13)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    return lr, y_pred_lr

def decision_regions(X_test, y_test, lr):
    from mlxtend.plotting import plot_decision_regions
    plot_decision_regions(X_test, y_test, lr)
    plt.show()

def make_DT(X_train, y_train, X_test):
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(random_state=13)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    return dt, y_pred_dt

def make_seed2():
    np.random.seed(13)
    X = np.random.randn(500, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap='winter')
    plt.show()
    return X, y

def overtraining():
    np.random.seed(13)
    n = 100
    X = np.random.normal(size=(n, 2))
    X[:50, :] += 0.25
    X[50:, :] -= 0.25
    y = np.array([1] * 50 + [0] * 50)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap='winter')
    plt.show()

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    from sklearn.tree import DecisionTreeClassifier
    from mlxtend.plotting import plot_decision_regions
    for i, max_depth in enumerate([3, 5, None]):
        for j, min_samples_leaf in enumerate([15, 5, 1]):
            dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=13)
            dt.fit(X, y)
            ax[i][j].set_title('max_depth = {} | min_samples_leaf = {}'.format(max_depth, min_samples_leaf))
            ax[i][j].axis('off')
            plot_decision_regions(X, y, dt, ax=ax[i][j])
    plt.show()

    dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=13)
    dt.fit(X, y)
    plot_decision_regions(X, y, dt)
    plt.show()

    print(accuracy_score(y, dt.predict(X)))

def instability():
    np.random.seed(13)
    n = 100
    X = np.random.normal(size=(n, 2))
    X[:50, :] += 0.25
    X[50:, :] -= 0.25
    y = np.array([1] * 50 + [0] * 50)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap='winter')
    plt.show()

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    from sklearn.tree import DecisionTreeClassifier
    from mlxtend.plotting import plot_decision_regions

    for i in range(3):
        for j in range(3):
            seed_idx = 3 * i + j
            np.random.seed(seed_idx)
            dt = DecisionTreeClassifier(random_state=13)
            idx_part = np.random.choice(len(X), replace=False, size=int(0.9 * len(X)))
            X_part, y_part = X[idx_part, :], y[idx_part]
            dt.fit(X_part, y_part)
            ax[i][j].set_title('sample #{}'.format(seed_idx))
            ax[i][j].axis('off')
            plot_decision_regions(X_part, y_part, dt, ax=ax[i][j])
    plt.show()

def H(R, y):
    return y[R.index].var(ddof=0)


def split_node(R_m, feature, t):
    left = R_m[R_m[feature] <= t]
    right = R_m[R_m[feature] > t]
    return left, right


def q_error(R_m, feature, t, y):
    left, right = split_node(R_m, feature, t)
    # print(left, right)
    return (len(left)/len(R_m)*H(left, y)+len(right)/len(R_m)*H(right, y))

def get_optimal_split(R_m, feature, y):
    Q_array = []
    feature_values = np.unique(R_m[feature])
    for t in feature_values:
        a = q_error(R_m, feature, t, y)
        not np.isnan(a) and Q_array.append(a)
    opt_threshold = feature_values[np.argmin(Q_array)]
    return opt_threshold, Q_array

if __name__ == "__main__":
    # X, y = make_seed()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
    # lr, y_pred_lr = logreg(X_train, y_train< X_test)
    # print(accuracy_score(y_pred_lr, y_test))
    # decision_regions(X_test, y_test, lr)
    # dt, y_pred_dt = make_DT(X_train, y_train, X_test)
    # print(accuracy_score(y_pred_dt, y_test))
    # decision_regions(X_test, y_test, dt)
    # X, y = make_seed2()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
    # lr, y_pred_lr = logreg(X_train, y_train, X_test)
    # print(accuracy_score(y_pred_lr, y_test))
    # decision_regions(X_test, y_test, lr)
    # dt, y_pred_dt = make_DT(X_train, y_train, X_test)
    # print(accuracy_score(y_pred_dt, y_test))
    # decision_regions(X_test, y_test, dt)
    # overtraining()
    # instability()
    # linnerud = load_linnerud(as_frame=True)
    linnerud = load_linnerud()
    print(linnerud.DESCR)
    print(f"Ключи датасета: {linnerud.keys()}")
    print(f"D.keys() -> a set-like object providing a view on D's keys")
    print(f"Признаки датасета: {linnerud.feature_names, linnerud.target_names}")
    # X: pd.DataFrame = linnerud.data
    X = pd.DataFrame(data=linnerud.data, columns=linnerud.feature_names)
    print(X)
    print(f"Последние 5 значений X:\n{X[-5:]}")
    print(f"Размер X: {len(X)}")
    y = linnerud.target
    print(f"Последние 5 значений y:\n{y[-5:]}")
    print(f"Размер y: {len(y)}")
    plt.title('... distribution')
    plt.xlabel('...')
    plt.ylabel('# samples')
    plt.hist(y, bins=20)
    plt.show()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=13)
    Q_array = [q_error(X_train, "Chins", 5, y)]
    print(f'Ошибка: {Q_array[0]}')
    feature = 'Chins'
    feature_values = np.unique(X_train[feature])
    print(feature_values, len(feature_values))
    Q_array = list(map(lambda x: q_error(X_train, 'Chins', x, y), feature_values))
    print(Q_array)
    nan_value = feature_values[np.where(np.isnan(Q_array))]
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, Q_array, marker='o', linestyle='-')
    plt.xlabel('Порог')
    plt.ylabel('Значение ошибки')
    plt.title(f'Feature {feature}')
    plt.grid(True)
    plt.show()
    # Задание 2.4
    results = []
    for f in X_train.columns:
        t, Q_array = get_optimal_split(X_train, f, y)
        print(t, Q_array, Q_array[np.argmin(Q_array)])
        results.append((f, t, Q_array[np.argmin(Q_array)]))
    results = sorted(results, key=lambda x: x[2])
    print(f"Результаты 2.4: {results}")
    results_df = pd.DataFrame(results, columns=['feature', 'optimal t', 'min Q error'])
    optimal_feature, optimal_t, optimal_error = results[0]
    print(results_df)
    _, optimal_Q_array = get_optimal_split(X_train, optimal_feature, y)
    plt.figure(figsize=(10, 6))
    print(X_train[optimal_feature], optimal_Q_array)
    print(np.unique(X_train[optimal_feature]))
    print(nan_value)
    plt.plot(np.delete(np.unique(X_train[optimal_feature]), np.where(np.unique(X_train[optimal_feature]) == nan_value)[0][0]), optimal_Q_array, marker='o', linestyle='-')
    plt.xlabel('Порог')
    plt.ylabel('Значение ошибки')
    plt.title(f'Feature {feature}')
    plt.grid(True)
    plt.show()
    # Задание 2.5
    print(len(X[optimal_feature]), len(y))
    print(X[optimal_feature], y)
    print(type(X[optimal_feature]), type(y))
    plt.scatter(X[optimal_feature], y)
    plt.axvline(x=optimal_t, color="red")
    plt.xlabel(optimal_feature)
    plt.ylabel('target')
    plt.title('Feature: {} | optimal t: {} | Q error: {:.2f}'.format(optimal_feature, optimal_t, optimal_error))
    plt.show()
