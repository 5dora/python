#%%
import sqlite3 #데이터베이스
import re #정규식
import numpy as np #숫자 라이브러리
import pandas as pd #데이터 처리 라이브러리
import matplotlib.pyplot as plt # 그래프 라이브러
import matplotlib
import seaborn as sns # 그래프 고도화
# %%
# scikit-learn
from sklearn import datasets as data
print(data)
dir(data) # 디렉토리 확인

iris=data.load_iris()
irdata=iris.data # 데이터
irtgt=iris.target # 라벨링
feature=iris.feature_names # 데이터 컬럼명
tgtname = iris.target_names # 
#%%
print(irdata)
print(irtgt)
print(feature)
print(tgtname)
#%%
feature=['sl', 'sw','pl','pw']
df=pd.DataFrame(irdata,columns=feature)
df
# %%
df.plot() #기초 그래프
df.plot(style='.')
# %%
df.describe() # 기초 통계량 요약
# %%
df.info() # 데이터 타입 요약
# %%
# 카테고리별 갯수 히스토그램
plt.hist(irtgt) # 데이터가 잘 분포되어있는지 확인/ 비대칭 데이터는 아닌가 확인하기
# %%
df['tgt']=irtgt
df
# %%
sns.pairplot(df, hue='tgt')
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(irdata, irtgt,test_size=0.3, shuffle=True, random_state=1)
print(X_train.shape, X_test.shape)
#%%
######################## 최근접 이웃 ############################
from sklearn.neighbors import KNeighborsClassifier
# 하이퍼파라미터 튜닝
for i in range(3,20,2):
    print('knn: ', i)
    knn3 = KNeighborsClassifier(n_neighbors=i) # 모델 저장하기
    knn3.fit(X_train, Y_train) # 학습 시키기
    pred=knn3.predict(X_test) #시험보기
    print(pred) # 시험본답
    print(Y_test) # 실제 답
    from sklearn.metrics import accuracy_score # 정확도 계산하기
    acc = accuracy_score(pred, Y_test)
    print("점수[",i,"]: ", acc)
# %%
#################### 서포트 백터 머신 ########################
from sklearn.svm import SVC
for i in range(1,10):
    svc = SVC(C=i)
    svc.fit(X_train, Y_train)
    pred= svc.predict(X_test)
    print(pred)
    print(Y_test)
    acc=accuracy_score(pred, Y_test)
    print('SVM[',i,'] acc: ', acc)
# %%
############# 의사결정 나무 ##################
from sklearn.tree import DecisionTreeClassifier as DT
for j in range(2, 10):   
    for i in range(2, 10):   
        dt = DT(max_depth=i, min_samples_leaf=j)
        dt.fit(X_train, Y_train)
        pred=dt.predict(X_test)
        print(pred)
        print(Y_test)
        acc = accuracy_score(pred, Y_test)
        print('DT[',i,',',j,'] acc:', acc)
# %%
############## 앙상블 모델 랜덤 포레스트 #############
from sklearn.ensemble import RandomForestClassifier as RF
rf=RF()
rf.fit(X_train, Y_train)
pred=dt.predict(X_test)
print(pred)
print(Y_test)
acc=accuracy_score(pred, Y_test)
print('RF[',i,'] acc: ', acc)
