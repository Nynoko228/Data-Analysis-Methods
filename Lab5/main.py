import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
# nltk.download("stopwords", download_dir="C:\\Users\\GUEST\\Desktop\\IVM-24\\Lab5\\nltk_data")
# nltk.download("punkt", download_dir="C:\\Users\\GUEST\\Desktop\\IVM-24\\Lab5\\nltk_data")
# nltk.download("snowball_data", download_dir="C:\\Users\\GUEST\\Desktop\\IVM-24\\Lab5\\nltk_data")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("snowball_data")
# nltk.data.path.append("C:\\Users\\GUEST\\Desktop\\IVM-24\\Lab5\\nltk_data")
import threading

def preprocess_text(texts):
    stop_words = set(stopwords.words('russian'))
    regex = re.compile('[^а-я А-Я]')
    preprocess_texts = []
    for i in  tqdm.tqdm(range(len(texts))):
        text = texts[i].lower()
        text = regex.sub(' ', text)
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        preprocess_texts.append(' '.join(filtered_sentence))

    return preprocess_texts

def stemming_texts(texts):
    # st = LancasterStemmer()
    st = SnowballStemmer("russian")
    stem_text = []
    for text in tqdm.tqdm(texts):
        word_tokens = word_tokenize(text)
        stem_text.append(' '.join([st.stem(word) for word in word_tokens]))
    return stem_text

def bow(vectorizer, train, test):
    train_bow = vectorizer.fit_transform(train)
    test_bow = vectorizer.transform(test)
    return train_bow, test_bow

from sklearn.feature_extraction.text import CountVectorizer

def prepare_data(df, text_column, target_column, test_size=0.3, random_state=42):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def learning(X_train, y_train, num):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    RFC, GBC = RandomForestClassifier(), GradientBoostingClassifier()
    param_grid_rf = {'max_depth': [5, 6, 7], 'max_features': ['sqrt', 'log2']}
    grid_rf = GridSearchCV(RFC, param_grid_rf, cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    grid_rf.fit(X_train, y_train)

    print(f"Лучшие параметры: {grid_rf.best_params_}, {num}")

    param_grid_gb = {'max_depth': [5, 6, 7], 'max_features': ['sqrt', 'log2']}
    grid_gb = GridSearchCV(GBC, param_grid_gb, cv=3, n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    grid_gb.fit(X_train, y_train)

    print(f"Лучшие параметры: {grid_gb.best_params_}, {num}")


if __name__ == "__main__":

    # vectorizer = CountVectorizer()
    # vectorizer.fit(['порядок слов в документе не важен', 'мешок слов'])
    # print(vectorizer)
    # print(vectorizer.get_feature_names_out())
    # print(vectorizer.transform(['важен порядок', 'не мешок не порядок']).toarray())
    # # newsgroups_train = pd.read_csv('negative.csv', delimiter=';')
    # newsgroups_train = fetch_20newsgroups(subset='train')
    # newsgroups_test = fetch_20newsgroups(subset='test')
    # print(newsgroups_train.keys())
    # print(newsgroups_train.data[0])
    # newsgroups_train['preprocess_data'] = preprocess_text(newsgroups_train.data)
    # newsgroups_test['preprocess_data'] = preprocess_text(newsgroups_test.data)
    # print(newsgroups_train['preprocess_data'][0])
    #
    # newsgroups_train['data_stemming'] = stemming_texts(newsgroups_train.preprocess_data)
    # newsgroups_test['data_stemming'] = stemming_texts(newsgroups_test.preprocess_data)
    # print(newsgroups_train.data_stemming[0])
    # print(newsgroups_train.preprocess_data[0])
    #
    # X_train_bow, X_test_bow = bow(vectorizer,
    #                               newsgroups_train.data,
    #                               newsgroups_test.data)
    #
    # print(X_train_bow.shape, X_test_bow.shape)

    dfN = pd.read_csv('negative.csv', header=None, sep=';', quotechar='"')
    dfP = pd.read_csv('positive.csv', header=None, sep=';', quotechar='"')

    df = pd.concat([dfN, dfP])
    df = df.sample(frac=1).reset_index(drop=True)  # перемешали данные
    print(f"df:\n{df}")
    df.columns = ["id", "date", "name", "message", "sentiment"] + ["undefined" for i in range(df.shape[1] - 5)]
    print(f"df:\n{df}")
    preprocess_data = df
    preprocess_data["message"] = preprocess_text(df.message)
    # print(f"preprocess_data:\n{preprocess_data}")

    # print(preprocess_data[0])
    data_stemming = preprocess_data
    data_stemming["message"] = stemming_texts(preprocess_data["message"])
    print(f"data_stemming:\n{data_stemming}")
    # print(data_stemming[0])

    X_train, X_test, y_train, y_test = prepare_data(df, "message", "sentiment")
    X_train_preprocess, X_test_preprocess, y_train_preprocess, y_test_preprocess = prepare_data(preprocess_data, "message", "sentiment")
    X_train_stemming, X_test_stemming, y_train_stemming, y_test_stemming = prepare_data(data_stemming, "message", "sentiment")
    threading.Thread(target=learning, args=(X_train, y_train, 1)).run()
    threading.Thread(target=learning, args=(X_train_preprocess, y_train_preprocess, 2)).run()
    threading.Thread(target=learning, args=(X_train_stemming, y_train_stemming, 3)).run()
    # learning(X_train_preprocess, y_train_preprocess, 2)
    # learning(X_train_stemming, y_train_stemming, 3)