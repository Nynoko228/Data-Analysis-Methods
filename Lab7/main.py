import zipfile
from collections import defaultdict, Counter
import datetime
import math
from scipy import linalg
import numpy as np

# read data
movies = {} # id
users = {} # id
ratings = defaultdict(list) # user-id

# Ссылка для загрузки зипника: http://files.grouplens.org/datasets/movielens/ml-1m.zip

with zipfile.ZipFile("ml-1m.zip", "r") as z:
    # parse movies
    with z.open("ml-1m/movies.dat") as m:
        for line in m:
            MovieID, Title, Genres = line.decode('iso-8859-1').strip().split("::")
            MovieID = int(MovieID)
            Genres = Genres.split("|")
            if Title == "Star Wars: Episode V - The Empire Strikes Back (1980)":
                StarWarsID = MovieID
                print(f"MovieID у фильма Звёздные воины 5: {StarWarsID}")
            movies[MovieID] = {"Title": Title, "Genres": Genres}

    # parse users
    with z.open("ml-1m/users.dat") as m:
        fields = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
        for line in m:
            row = list(zip(fields, line.decode('iso-8859-1').strip().split("::")))
            data = dict(row[1:])
            data["Occupation"] = int(data["Occupation"])
            users[int(row[0][1])] = data

    # parse ratings
    with z.open("ml-1m/ratings.dat") as m:
        for line in m:
            UserID, MovieID, Rating, Timestamp = line.decode('iso-8859-1').strip().split("::")
            UserID = int(UserID)
            MovieID = int(MovieID)
            Rating = int(Rating)
            Timestamp = int(Timestamp)
            ratings[UserID].append((MovieID, Rating, datetime.datetime.fromtimestamp(Timestamp)))



# train-test split
times = []
for user_ratings in ratings.values():
  times.extend([x[2] for x in user_ratings])
times = sorted(times)
threshold_time = times[int(0.8 * len(times))]

train = []
test = []
for user_id, user_ratings in ratings.items():
    train.extend((user_id, rating[0], rating[1] / 5.0) for rating in user_ratings if rating[2] <= threshold_time)
    test.extend((user_id, rating[0], rating[1] / 5.0) for rating in user_ratings if rating[2] > threshold_time)
print("ratings in train:", len(train))
print("ratings in test:", len(test))

train_by_user = defaultdict(list)
test_by_user = defaultdict(list)
for u, i, r in train:
    train_by_user[u].append((i, r))

for u, i, r in test:
    test_by_user[u].append((i, r))

train_by_item = defaultdict(list)
for u, i, r in train:
    train_by_item[i].append((u, r))

n_users = max([e[0] for e in train]) + 1
n_items = max([e[1] for e in train]) + 1

# Реализация ALS
np.random.seed(0)
LATENT_SIZE = 10
N_ITER = 20

# регуляризаторы
lambda_p = 0.2
lambda_q = 0.001

# латентные представления
p = 0.1 * np.random.random((n_users, LATENT_SIZE))
q = 0.1 * np.random.random((n_items, LATENT_SIZE))


def compute_p(p, q, train_by_user):
    for u, rated in train_by_user.items():
        rated_items = [i for i, _ in rated]
        rated_scores = np.array([r for _, r in rated])
        Q = q[rated_items, :]
        A = (Q.T).dot(Q)
        d = (Q.T).dot(rated_scores)
        p[u, :] = np.linalg.solve(lambda_p * len(rated_items) * np.eye(LATENT_SIZE) + A, d)
    return p

def compute_q(p, q, train_by_item):
    for i, rated in train_by_item.items():
        rated_users = [j for j, _ in rated]
        rated_scores = np.array([s for _, s in rated])
        P = p[rated_users, :]
        A = (P.T).dot(P)
        d = (P.T).dot(rated_scores)
        q[i, :] = np.linalg.solve(lambda_q * len(rated_users) * np.eye(LATENT_SIZE) + A, d)
    return q

def train_error_mse(predictions):
    return np.mean([(predictions[u, i] - r) ** 2 for u, i, r in train])

def test_error_mse(predictions):
    return np.mean([(predictions[u, i] - r) ** 2 for u, i, r in test])


for iter in range(N_ITER):
    p = compute_p(p, q, train_by_user)
    q = compute_q(p, q, train_by_item)

    predictions = p.dot(q.T)

    print(iter, train_error_mse(predictions), test_error_mse(predictions))


# print(q[StarWarsID])
skal = {}
for i in range(len(q)):
    if i != StarWarsID:
        skal[i] = np.dot(q[i], q[StarWarsID])

top_3 = [key for key, value in sorted(skal.items(), key=lambda x: x[1], reverse=True)[:3]]
print(top_3)
print(f"Имена самых похожих фильмов: {movies[top_3[0]]['Title'], movies[top_3[1]]['Title'], movies[top_3[2]]['Title']}")
print(f"Сумма id: {sum(top_3)}")


skalP = {}
for i in range(len(p)):
    if i != 5472:
        skalP[i] = np.dot(p[i], p[5472])

top = [key for key, value in sorted(skalP.items(), key=lambda x: x[1], reverse=True)]
# print(top)
print(f"Имя самого похожего человека по просмотренным фильмам с 5472: {users[top[0]]}")
films1 = ratings[5472]
films2 = ratings[top[0]]
cnt = 0
for i in films1:
    for j in films2:
        if i[0] == j[0]:
            cnt += 1
print(cnt)

def DCG_k(ratings_list, k):
    '''
      ratings_list: np.array(n_items,)
      k: int
    '''
    summa = 0
    for i in range(k):
        summa += ((2**(ratings_list[i])-1)/math.log2(i+2))
    print(f"DCG_k = {summa}")
    return summa

def iDCG_k(ratings_list, k):
    print("iDCG_k")
    sorted_list = np.sort(ratings_list)[::-1]
    print(sorted_list)
    return DCG_k(sorted_list, k)

def NDCG_k(r, k):
    '''
      ratings_list: np.array(n_items,)
      k: int
    '''
    print(f"NDCG_k: {DCG_k(r, k) / iDCG_k(r, k)}")

NDCG_k([5, 5, 4, 5, 2, 4, 5, 3, 5, 5, 2, 3, 0, 0, 1, 2, 2, 3, 0], 5)


