from sklearn.datasets import fetch_california_housing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


def practice(data):
    print(data.keys())
    print(data['DESCR'])
    print(data['target'])
    X, y = data['data'], data['target']
    print("Размер матрицы объектов: ", X.shape)
    print("Рaзмер вектора y: ", y.shape)
    # plt.scatter(X[:, 2], y)
    # plt.xlabel('AveRooms')
    # plt.ylabel('Price')
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(mean_squared_error(y_test, predictions))
    grid_searcher = GridSearchCV(KNeighborsRegressor(),
                                 param_grid={'n_neighbors': range(1, 40, 2),
                                             'weights': ['uniform', 'distance'],
                                             'p': [1, 2, 3]},
                                 cv=5)
    # grid_searcher.fit(X_train, y_train)
    # best_predictions = grid_searcher.predict(X_test)
    # print(mean_squared_error(y_test, best_predictions))
    # print(grid_searcher.best_params_)
    metrics = []
    for n in range(1, 40, 2):
        knn = KNeighborsRegressor(n_neighbors=n)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        metrics.append(np.mean(scores))
    # plt.plot(range(1, 40, 2), metrics)
    # plt.ylabel('Negative mean squared error')
    # plt.xlabel('Number of neightbors')
    # plt.show()
    X, y = make_moons(n_samples=200, noise=0.2)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X, y)
    x_grid, y_grid = np.meshgrid(np.linspace(-2.0, 3.0, 100), np.linspace(-2.0, 2.0, 100))
    xy = np.stack([x_grid, y_grid], axis=2).reshape(-1, 2)
    predicted = knn_clf.predict(xy)
    plt.scatter(xy[:, 0], xy[:, 1], c=predicted, alpha=0.2, s=1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def not_normilize(df):
    plt.hist(df['trip_duration'], bins=30, edgecolor='red')
    plt.show()

def log_normilize(df):
    df_tr = np.log1p(df['trip_duration'])
    plt.hist(df_tr, bins=30, edgecolor='red')
    plt.show()

def add_log_trip_duration(df):
    df_log = np.log1p(df['trip_duration'])
    return df_log

def datetime(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    return df['pickup_datetime']

def dates_apply(df):
    dates = df['pickup_datetime'].apply(lambda x: x.date())
    print(dates)
    plt.figure(figsize=(25, 5))
    data_count_plot = sns.countplot(x=dates)
    data_count_plot.set_xticklabels(data_count_plot.get_xticklabels(), rotation=90)
    plt.show()

def group(df):
    # print(df.tail())
    dates = df['pickup_datetime'].apply(lambda x: x.date())
    group_by_date = df.groupby(dates)
    sns.relplot(data=group_by_date.log_trip_duration.aggregate('mean'), kind='line')
    # plt.show()
    # print(df.tail())
    # print(group_by_date.tail())

def create_features(data_frame):
    X = pd.concat(
        [
         data_frame.pickup_datetime.apply(lambda x: x.timetuple().tm_yday),
         data_frame.pickup_datetime.apply(lambda x: x.hour)
        ], axis=1, keys=['day', 'hour']
    )
    return X, data_frame.log_trip_duration

def LinePredict(X_train, y_train, X_test, y_test):
    line = LinearRegression()
    line.fit(X=X_train, y=y_train)
    predict = line.predict(X_test)
    print(f"Предсказание у линейной регрессии: {predict}")
    print(f"MSE: {mean_squared_error(y_test, predict)}")

def Neighbors(X_train, y_train, X_test, y_test):
    knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(f"Предсказание у методов основанных на ближайших соседях: {predictions}")
    print(f"MSE: {mean_squared_error(y_test, predictions)}")

if __name__ == "__main__":
    df = pd.read_csv('train/train.csv')
    df = df.drop(columns=['dropoff_datetime'])
    # print(df.head(7), "\n")
    df = df.sort_values(by="pickup_datetime")
    # print(df.head(7), "\n")
    train_df, test_df = df.iloc[:1000000], df.iloc[1000000:]
    # print(train_df.tail())
    # print(test_df.tail())
    # print(f"Size test_df: {len(test_df)}")
    # not_normilize(train_df)
    # log_normilize(train_df)
    train_df['log_trip_duration'], test_df['log_trip_duration'] = add_log_trip_duration(train_df), add_log_trip_duration(test_df)
    train_df['pickup_datetime'], test_df['pickup_datetime'] = datetime(train_df), datetime(test_df)
    # dates_apply(train_df)
    # dates_apply(test_df)
    group(train_df)
    X_train, y_train = create_features(train_df)
    X_test, y_test = create_features(test_df)
    # print(X_train.tail(5))
    ohe = ColumnTransformer([("One Hot", OneHotEncoder(sparse_output=False), [1])], remainder='passthrough')
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.fit_transform(X_test)
    print(X_train)
    print(X_train.shape)
    LinePredict(X_train, y_train, X_test, y_test)
    Neighbors(X_train, y_train, X_test, y_test)
    # print(train_df.tail())
    # print(test_df.tail())
