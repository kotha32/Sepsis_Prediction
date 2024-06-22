# Libraries
from django.shortcuts import render,redirect
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from .models import User




################ Home #################
def home(request):
	return render(request,'home1.html')
def login(request):
	return render(request,'loginform.html')
def loginCheck(request):
	if request.method=="POST":
		print('printtttttttttttttttttttttttttttttttt')
		username= request.POST.get('username')
		password= request.POST.get('email')
		try:
			user_object = User.objects.get(firstname=username,password=password)
			print(user_object)
		except:
			#user_object = None
			print('hello')
		if user_object is not None:
			print('hiiiiiiii')
			request.session['useremail'] = user_object.email
			return redirect('home')
			print('hiiiiiiii')
	return render(request,'home.html')	
def logout(request):
	return render(request,'index.html')	
def reg(request):
	return render(request,'register.html')

######## SVM ######
def save(request):
	if request.method == 'POST':
		print('printtttttttttttttttttttttttttttttttt')
		print('checkkkkkkkkkkkkkkkkk')
		username= request.POST.get('username')
		password= request.POST.get('password')
		address= request.POST.get('address')
		email= request.POST.get('email')
		age= request.POST.get('age')
		gender= request.POST.get('gender')
		phone= request.POST.get('phone')
		user=User()
		user.firstname= request.POST.get('username')
		user.password= request.POST.get('password')
		user.address= request.POST.get('address')
		user.email= request.POST.get('email')
		user.age= request.POST.get('age')
		user.gender= request.POST.get('gender')
		user.phone= request.POST.get('phone')
		user.save()		
		return render(request,'loginform.html')
	return render(request,'loginform.html')	

######## SVM ######
def nvb(request):
	return render(request, 'pacweb1.html')


def pac(request):
	if request.method == 'POST':
		if request.method == 'POST':
			headline1 = int(request.POST.get('headline1'))
			headline2 = int(request.POST.get('headline2'))
			headline3 = float(request.POST.get('headline3'))
			headline4 = int(request.POST.get('headline4'))
			headline5 = int(request.POST.get('headline5'))
			headline6 = int(request.POST.get('headline6'))
			headline7 = int(request.POST.get('headline7'))
			from django.shortcuts import render
			from django.http import HttpResponse
			import pandas as pd
			import numpy as np
			import matplotlib.pyplot as plt
			from sklearn.model_selection import train_test_split
			from sklearn.feature_extraction.text import TfidfVectorizer
			import itertools
			from sklearn import metrics
			import os
			import seaborn as sns
			from sklearn.model_selection import train_test_split
			from sklearn.metrics import confusion_matrix

			df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\sepsis_prediction\sepsis_prediction\sepsis\sepsis.csv')
			df.fillna(0, inplace=True)
			# printing df variable it will print total records in the dataset
			print(df)

			# df.iloc[] means index location of columns
			# here X stores 0 to 12 columns records
			X = df.iloc[:, 2:9].values

			# df.target means dataset has target column, we have taken target column as output or labled column
			y = df.SepsisLabel

			# atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
			# atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
			# train_test separation
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
			input = [[headline1, headline2, headline3, headline4, headline5, headline6, headline7]]
			# Applying Naive Bayes
			# Applying tfidf to the data set

			from sklearn.linear_model import PassiveAggressiveClassifier
			clf = PassiveAggressiveClassifier()
			clf.fit(X_train, y_train)  # Fit Naive Bayes classifier according to X, y
			pred = clf.predict(X_test)  # Perform classification on an array of test vectors X.
			pred1 = clf.predict(input)  # Perform classification on an array of test vectors X.
			print(pred)
			print(pred1)
			fakefalse = ''
			if pred1 == 1:
				fakefalse = 'sepsis'
			else:
				fakefalse = 'not a sepsis'
			score = metrics.accuracy_score(y_test, pred)
			print("accuracy:   %0.3f" % score)
			cm = metrics.confusion_matrix(y_test, pred)
			print(cm)
			d = {'a': score, 'b': cm, 'c': pred1, 'e': fakefalse}
	return render(request, 'result.html', d)
def svm(request):	
	return render(request,'acc1.html')		
def dec(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix

	df=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\sepsis_prediction\sepsis_prediction\sepsis\sepsis.csv')
	df.fillna(0, inplace=True)
	#printing df variable it will print total records in the dataset
	print(df)

	#df.iloc[] means index location of columns
	#here X stores 0 to 12 columns records
	X = df.iloc[:, 1:43].values

	#df.target means dataset has target column, we have taken target column as output or labled column
	y = df.SepsisLabel

	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
	from sklearn.linear_model import PassiveAggressiveClassifier
	linear_clf = PassiveAggressiveClassifier()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def randomf(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix

	df=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\sepsis_prediction\sepsis_prediction\sepsis\sepsis.csv')
	df.fillna(0, inplace=True)
	#printing df variable it will print total records in the dataset
	print(df)

	#df.iloc[] means index location of columns
	#here X stores 0 to 12 columns records
	X = df.iloc[:, 1:43].values

	#df.target means dataset has target column, we have taken target column as output or labled column
	y = df.SepsisLabel
	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	from sklearn.neighbors import KNeighborsClassifier
	linear_clf = KNeighborsClassifier(n_neighbors=3)
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	scored = round(score, 2)
	print(metrics.accuracy_score(y_test, pred))
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x = ['KNN ALGORITHM']
	y = [scored]

	plt.bar(x, y)
	plt.title('Top Word\n')
	
	plt.ylabel('Accuracy')
	plt.show()	  
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def mnb(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix

	df=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\sepsis_prediction\sepsis_prediction\sepsis\sepsis.csv')
	df.fillna(0, inplace=True)
	#printing df variable it will print total records in the dataset
	print(df)

	#df.iloc[] means index location of columns
	#here X stores 0 to 12 columns records
	X = df.iloc[:, 1:43].values

	#df.target means dataset has target column, we have taken target column as output or labled column
	y = df.SepsisLabel

	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	linear_clf = SVC()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def graph(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	from sklearn.linear_model import LogisticRegression

	df=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\sepsis_prediction\sepsis_prediction\sepsis\sepsis.csv')
	df.fillna(0, inplace=True)
	#printing df variable it will print total records in the dataset
	print(df)

	#df.iloc[] means index location of columns
	#here X stores 0 to 12 columns records
	X = df.iloc[:, 1:43].values

	#df.target means dataset has target column, we have taken target column as output or labled column
	y = df.SepsisLabel
	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	linear_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=8, min_samples_leaf=10),n_estimators=500, random_state=3, learning_rate=0.001	)
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def accuracy(request):
	return render(request,'index.html')
def graph2(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	from sklearn.linear_model import LogisticRegression

	df=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\sepsis_prediction\sepsis_prediction\sepsis\sepsis.csv')
	#printing df variable it will print total records in the dataset
	print(df)

	#df.iloc[] means index location of columns
	#here X stores 0 to 12 columns records
	X = df.iloc[:, 1:43].values

	#df.target means dataset has target column, we have taken target column as output or labled column
	y = df.SepsisLabel
	#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
	#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	#train_test separation
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	linear_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=8, min_samples_leaf=10),n_estimators=500, random_state=3, learning_rate=0.001	)
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	scored = round(score, 2)
	linear_clf = SVC()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score1 = metrics.accuracy_score(y_test, pred)
	scorer = round(score1, 2)	 
	linear_clf = KNeighborsClassifier(n_neighbors=3)
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score2 = metrics.accuracy_score(y_test, pred) 
	scoregnbpred = round(score2, 2)	 
	from sklearn.linear_model import PassiveAggressiveClassifier
	linear_clf = PassiveAggressiveClassifier()
	linear_clf.fit(X_train, y_train)
	pred = linear_clf.predict(X_test)
	print('=====================================================================')
	#pred2 = linear_clf.predict(atest1)
	print(pred)
	score3 = metrics.accuracy_score(y_test, pred)	 
	scoresvcpred= round(score3, 2)
	meanse=[scored,scorer,scoregnbpred,scoresvcpred]
	#bargraph for meansquarederror
	'''fig = plt.figure()
	ax = fig.add_subplot(111)
	langs = ['AdaBoost','SVC','KNeighbors','PassiveAggressive']
	students = [meanse[0],meanse[1],meanse[2],meanse[3],meanse[4]]
	ax.bar(langs,students)
	plt.show()'''
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'home1.html')	   		