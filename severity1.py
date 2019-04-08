# this script runs at every call

# import speech_recognition as sr
import pandas as pd
import numpy as np
from sklearn import preprocessing,neighbors

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

city_freq = pd.read_csv('data.csv')

city_list = list(city_freq.City) # list of all the cities present in the database
food_words = ['food']
medical_words = ['medicine']
rescue_words = ['rescue']

r = sr.Recognizer()
r.energy_threshold = 1000
mic = sr.Microphone()
with mic as source:
    print("Say something...")
    audio = r.listen(source)

isaid = r.recognize_google(audio)
# isaid = "Im from mumbai i need food medicine food rescue"
list_of_words = isaid.split(" ")
said_cities = []
print("Call converted to text : "+isaid)
print()
# print("list of word",list_of_words)
# print("city list",city_list)
for word in list_of_words:
    if word.lower() in city_list:
        said_cities.append(word)
        #print("mil gayi")
#
# # we have said_cities like ghaziabad ghaziabad delhi delhi delhi ...
# print(said_cities)
said_cities = set(said_cities)
# print(said_cities)
# print(said_cities_set)


# updating city numbers in csv file to calculate severity 2
for city in said_cities:
    index_city = city_list.index(city.lower()) # index of the city in the csv file data.csv
    city_freq.loc[index_city,'number'] = city_freq.loc[index_city,'number'] + 1
city_freq.to_csv('data.csv',index=False)

# calculating severity1 for every call and adding in another dataset , for that we have to train for severity 1

help_dataset = pd.read_csv("sev1_training.csv")
help_dataset.replace('?',-99999,inplace=True)
X = np.array(help_dataset.drop(['Severity1'],1))
y = np.array(help_dataset['Severity1'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy in finding the Severity of the call is : " + str(accuracy*100) + "%")
f = 0
m = 0
r = 0
n = 1
for word in list_of_words:
    if word.lower() in food_words:
        # print(word)
        f = 1
    elif word.lower() in medical_words:
        # print(word)
        m = 1
    elif word.lower() in rescue_words:
        # print(word)
        r = 1
# print(f,m,r,n)
parameters = np.array([[f,m,r,n]])
parameters = parameters.reshape(len(parameters),-1)
severity1 = (clf.predict(parameters))[0]
print("Severity of this call : " + str(severity1))
# severity1 = 4 # assumed

city_sev1_df = pd.read_csv('city_sev1.csv')
index = len(city_sev1_df)

city_sev1_df.loc[index, 'City'] = list(said_cities)[0]
city_sev1_df.loc[index, 'Severity1'] = severity1
city_sev1_df.to_csv('city_sev1.csv',index=False)



