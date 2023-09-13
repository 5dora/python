#%%
############ import ###############
import sqlite3 # 데이터베이스
import re # 정규식
import numpy as np # 숫자 라이브러리
import pandas as pd # 데이터 처리 라이브러리
import matplotlib.pyplot as plt # 그래프 라이브러
import matplotlib
import seaborn as sns # 그래프 고도화
#%%
########### 데이터 불러오기 ###########
# scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data['data']
Y = data['target']
fname = data['feature_names']
print(data.DESCR)  # Description 속성을 이용해서 데이터셋의 정보를 확인
df=pd.DataFrame(X, columns=fname)
df
#%%
############### EDA #################
#%%
df.describe() # 기초 통계량 요약
#%%
df.info() # 데이터 타입 요약
#%%
#### 기초 시각화 ###
plt.hist(Y) # 분포확인
#%%
plt.plot(df['mean radius'],'.')
#%%
sns.scatterplot(df.iloc[:,:3]) # 행은 전체, 열은 3개
#%%
tdf= df.copy()
tdf['tgt'] = Y
sns.pairplot(tdf.iloc[:,-5:], hue='tgt')
#%%
########## 데이터 전처리 ##########
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=True, random_state=1, stratify=Y) 
# stratify 비율 분포를 맞춰서 나누기
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
plt.hist(Y_train)
plt.hist(Y_test)
#%%
################ ML ############
# 앙상블 모델 RandomForest
from sklearn.ensemble import RandomForestClassifier as RF
def makeRF(i,j):
    rf = RF(max_depth=i, max_leaf_nodes=j)
    rf.fit(X_train, Y_train)
    pred=rf.predict(X_test)
    acc = accuracy_score(pred, Y_test)
    print('RF[', i,j,'] acc: ', acc)
    return acc
accs = []
beforeACC=0
bestACC=[]
for i in range(2,10):
    for j in range(2,10):
        acc=makeRF(i,j)
        if(acc>beforeACC):
            bestACC = [i,j,acc]   
        beforeACC = acc 
        accs.append(acc)
#%%
plt.plot(accs)
print(bestACC)
#%%
# 랜덤포레스트 매개변수 고정
# from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier(n_estimators=5, random_state=2) 
# forest.fit(X_train, Y_train)
#%%
# DecisionTree
from sklearn.tree import DecisionTreeClassifier as DT
def makeRF(i,j):
    rf = RF(max_depth=i, max_leaf_nodes=j)
    rf.fit(X_train, Y_train)
    pred=rf.predict(X_test)
    acc = accuracy_score(pred, Y_test)
    print('RF[', i,j,'] acc: ', acc)
    return acc
accs = []
beforeACC=0
bestACC=[]
for i in range(2,10):
    for j in range(2,10):
        acc=makeRF(i,j)
        if(acc>beforeACC):
            bestACC = [i,j,acc]   
        beforeACC = acc 
        accs.append(acc)
plt.plot(accs)
print(bestACC)

# %%
# 그래디언트 부스팅 - 무겁다
from sklearn.ensemble import GradientBoostingClassifier as GB
def makeGB(i,j):
    gb = GB(min_samples_split=j, n_estimators=i*50)
    gb.fit(X_train, Y_train)
    pred=gb.predict(X_test)
    acc = accuracy_score(pred, Y_test)
    print('GB[', i,j,'] acc: ', acc)
    return acc
accs = []
beforeACC=0
bestACC=[]
for i in range(1,10):
    for j in range(2,10):
        acc=makeGB(i,j)
        if(acc>beforeACC):
            bestACC = [i,j,acc]   
        beforeACC = acc 
        accs.append(acc)
plt.plot(accs)
print(bestACC)
# %%
# Adaboosting - 모멘텀 주면서
from sklearn.ensemble import AdaBoostClassifier as AB
def makeAB(j):
    ab = AB(n_estimators=j*20)
    ab.fit(X_train, Y_train)
    pred=ab.predict(X_test)
    acc = accuracy_score(pred, Y_test)
    print('AB[', j,'] acc: ', acc)
    return acc
accs = []
beforeACC=0
bestACC=[]
for j in range(1,10):
    acc=makeAB(j)
    if(acc>beforeACC):
        bestACC = [j,acc]   
    beforeACC = acc 
    accs.append(acc)
plt.plot(accs)
print(bestACC)
# %%
