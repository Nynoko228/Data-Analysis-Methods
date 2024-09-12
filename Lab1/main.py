import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('math_students.csv', delimiter=',')


def reasons(data):
    lib = {"home": "близко к дому", "reputation": "репутация школы", "course": "предпочтение некоторым предметам",
           "other": "другое"}
    print("Самая частая причина выбора была", lib[data['reason'].value_counts().index[0]])


def MFedu(data):
    print("Нету образования у отца", data[(data['Fedu'] == 0)].shape[0])
    print("Нету образования у матери", data[(data['Medu'] == 0)].shape[0])
    print("Нету образования у обоих родителей", data[((data['Medu'] == 0) & (data['Fedu'] == 0))].shape[0])


def age(data):
    print("Минимальный возраст ученика в школе Mousinho da Silveira составляет",
          data[(data['school'] == "MS")]['age'].min(), "лет")


def cntofabsences(data):
    print(f"Количество учеников с нечётным числом пропусков составляет {len([i for i in data['absences'] if i % 2 == 1])}")


def raznosti(data):
    d = data[["G3", "romantic"]]
    NoRomantic = d.query("romantic == 'no'")['G3'].mean()
    YesRomantic = d.query("romantic == 'yes'")['G3'].mean()
    print(
        f"Разность между средними итоговыми оценками студентов, состоящих и не состоящих в романтических отношениях составляет {round(abs(NoRomantic - YesRomantic), 2)} баллов")


def six(data):
    activities = data['activities'].value_counts().index[0]
    d = data[['absences', 'activities']]
    absences = d.query(f"activities == '{activities}'")['absences'].value_counts()
    print(
        f"Чаще всего студенты с внеклассными занятиями имели {absences.index[0]} пропусков и количество таких студентов составляет {absences[0]}")


def histogramma(data):
    import numpy as np
    d = data['G1'].value_counts()
    print(d)
    plt.figure(figsize=(10, 7))
    plt.title('Оценки за первый семестр')
    plt.bar(d.index, d.values, edgecolor='black', linewidth=2, width=1)
    x_ticks = np.linspace(1, 20, 20)
    plt.xticks(x_ticks)
    plt.xlabel('Оценка')
    plt.ylabel('Кол-во студентов')
    plt.show()


reasons(data)
MFedu(data)
age(data)
cntofabsences(data)
raznosti(data)
six(data)
histogramma(data)

# функция .head(n) выводит первые n строк таблицы (по умолчанию n=5)
# print(data.head()) # Вывод первых n строк
# print(data.tail()) # Вывод последних n строк
# print(data.shape) # Вывод количества x строк и y столбцов
# print(data.columns) # Вывод названий y столбцов
# print(data[data.columns[:-1]].head())
# print(data.iloc[:, :-1].head())
# print(data.loc[:, data.columns[:-1]].head())
# print(data.drop(['G3'], axis=1).head())
# print(data.isnull().any().any())
# print(data.describe())
# print(data.info())
# print(data['guardian'].unique())
# print(data['guardian'].nunique())
# print(data['guardian'].value_counts())
# print(data[(data['guardian'] == 'mother') & ((data['Mjob'] == 'teacher') | (data['Mjob'] == 'at_home'))].head())
# print(data[(data['guardian'] == 'mother') & ((data['Mjob'] == 'teacher') | (data['Mjob'] == 'at_home'))].shape)
# data['alc'] = (5 * data['Dalc'] + 2 * data['Walc']) / 7
# print(data[['Walc', 'Dalc', 'alc']].head(10))
# print(data.shape) # Вывод количества x строк и y столбцов
# print(data.columns) # Вывод названий y столбцов
# print(data['absences'].mean())
# mean_absences = data['absences'].mean()
# stud_few_absences = data[data['absences'] < mean_absences]
# stud_many_absences = data[data['absences'] >= mean_absences]
# print('Students with few absences:', stud_few_absences.shape[0])
# print('Students with many absences:', stud_many_absences.shape[0])
# stud_few_absences_g3 = stud_few_absences['G3'].mean()
# stud_many_absences_g3 = stud_many_absences['G3'].mean()
# print('Students with few absences, mean G3:', stud_few_absences_g3)
# print('Students with many absences, mean G3:', stud_many_absences_g3)
# data_by_school = data.groupby('school')
# # print(data_by_school.describe())
# print(data_by_school.mean(numeric_only=True))

# plt.figure(figsize=(10,7))
# plt.title('Распределение кол-во пропусков')
# data['absences'].hist()
# plt.xlabel('Кол-во пропусков')
# plt.ylabel('Кол-во студентов')
# plt.show()
