import os
import sys
import warnings

from nltk.data import ZipFilePathPointer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import nltk
# nltk.data.path.append(r'C:\Users\Alexander\Desktop\Магистр\Data-Analysis-Methods\Lab5\nltk_data')
import re
import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
import threading

def preprocess_text(texts):
    stop_words = set(stopwords.words('russian'))
    regex = re.compile('[^а-я А-Я]')
    preprocess_texts = []
    for i in  tqdm.tqdm(range(len(texts))):
        text = texts[i].lower()
        text = regex.sub(' ', text)
        word_tokens = word_tokenize(text, language="russian")
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        preprocess_texts.append(' '.join(filtered_sentence))
    return preprocess_texts

def stemming_texts(texts):
    st = SnowballStemmer("russian")
    stem_text = []
    for text in tqdm.tqdm(texts):
        word_tokens = word_tokenize(text, language="russian")
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

def prepare_data_TF(df, text_column, target_column, test_size=0.3, random_state=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def learning(X_train, y_train, X_test, y_test, num, bag):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    RFC, GBC = RandomForestClassifier(), GradientBoostingClassifier()
    param_grid_rf = {'max_depth': [5, 6, 7], 'max_features': ['sqrt', 'log2']}
    grid_rf = GridSearchCV(RFC, param_grid_rf, cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    y_predicted = grid_rf.predict(X_test)
    accuracy_score_n = accuracy_score(y_predicted, y_test)
    print(f"Лучшие параметры у дерева: {grid_rf.best_params_}, {num}, {bag}, {accuracy_score_n}")

    param_grid_gb = {'max_depth': [5, 6, 7], 'max_features': ['sqrt', 'log2']}
    # param_grid_gb = {'max_depth': [10, 20, 30], 'max_features': ['sqrt', 'log2']}
    grid_gb = GridSearchCV(GBC, param_grid_gb, cv=3, n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    y_predicted = grid_gb.predict(X_test)
    accuracy_score_n = accuracy_score(y_predicted, y_test)
    print(f"Лучшие параметры у градиентного спуска: {grid_gb.best_params_}, {num}, {bag}, {accuracy_score_n}")

def Bag_of_Words():
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(['порядок слов в документе не важен', 'мешок слов'])
    print(vectorizer.get_feature_names_out())
    vectorizer.transform(['важен порядок', 'не мешок не порядок']).toarray()
    X_train_bow, X_test_bow = bow(vectorizer,
                                  X_train.data,
                                  X_test.data)
    print(X_train_bow.shape)
    print(X_test_bow.shape)
    X_train_bow, X_test_bow = bow(vectorizer,
                                  X_train.data_stemming,
                                  X_test.data_stemming)
    print(X_train_bow.shape)
    print(X_test_bow.shape)


def Bag_of_WordsTF():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer_tf_idf = TfidfVectorizer()
    X_train_tfidf, X_test_tfidf = bow(vectorizer_tf_idf,
                                      X_train.data,
                                      X_test.data)
    X_train_tfidf_preprocess, X_test_tfidf_preprocess = bow(vectorizer_tf_idf,
                                                            X_train.preprocess_data,
                                                            X_test.preprocess_data)

if __name__ == "__main__":
    # if not os.path.exists(nltk_data_dir):
    #     os.makedirs(nltk_data_dir)
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


    data_stemming = preprocess_data
    data_stemming["message"] = stemming_texts(preprocess_data["message"])
    print(f"data_stemming:\n{data_stemming}")

    logs = "output.txt"
    with open(logs, "w", encoding="utf-8") as file:
        # Перенаправляем вывод
        sys.stdout = file
        X_train, X_test, y_train, y_test = prepare_data(df, "message", "sentiment")
        X_train_preprocess, X_test_preprocess, y_train_preprocess, y_test_preprocess = prepare_data(preprocess_data, "message", "sentiment")
        X_train_stemming, X_test_stemming, y_train_stemming, y_test_stemming = prepare_data(data_stemming, "message", "sentiment")
        threading.Thread(target=learning, args=(X_train, y_train, X_test, y_test, "исходные тексты", "Обычный мешок слов")).run()
        threading.Thread(target=learning, args=(X_train_preprocess, y_train_preprocess, X_test_preprocess, y_test_preprocess, "предварительно обработанные тексты", "Обычный мешок слов")).run()
        threading.Thread(target=learning, args=(X_train_stemming, y_train_stemming, X_test_stemming,  y_test_stemming, "тексты после стемминга", "Обычный мешок слов")).run()

        X_trainTF, X_testTF, y_trainTF, y_testTF = prepare_data_TF(df, "message", "sentiment")
        X_train_preprocessTF, X_test_preprocessTF, y_train_preprocessTF, y_test_preprocessTF = prepare_data_TF(preprocess_data, "message", "sentiment")
        X_train_stemmingTF, X_test_stemmingTF, y_train_stemmingTF, y_test_stemmingTF = prepare_data_TF(data_stemming, "message", "sentiment")
        threading.Thread(target=learning, args=(X_trainTF, y_trainTF, X_testTF, y_testTF, "исходные тексты", "Взвешенный мешок слов")).run()
        threading.Thread(target=learning, args=(X_train_preprocessTF, y_train_preprocessTF, X_test_preprocessTF, y_test_preprocessTF, "предварительно обработанные тексты", "Взвешенный мешок слов")).run()
        threading.Thread(target=learning, args=(X_train_stemmingTF, y_train_stemmingTF, X_test_stemmingTF, y_test_stemmingTF, "тексты после стемминга", "Взвешенный мешок слов")).run()
    sys.stdout = sys.__stdout__

    # learning(X_train_preprocess, y_train_preprocess, 2)
    # learning(X_train_stemming, y_train_stemming, 3)