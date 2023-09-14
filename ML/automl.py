########### AUTOML - Pycaret #############
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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
cancer = load_breast_cancer()
cancer
# %%
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['tgt'] = cancer.target
df

# %%
############## Pycaret 사용하기 ################
from pycaret.classification import * # 분류 or 회귀 선택
# setup() pycaret을 사용하기 위한 data setting
cres = setup(df, target='tgt',train_size=0.8, session_id=0) 
# train_size 설정을 통해 train과 test 별도로 분리하지 않아도 됨
# session_id: Random seed 고정
cres
# %%
models() # 어떤 모델 사용할 수 있는지 확인
# %%
# compare_models(): 여러 모델의 성능 비교
bestmodel = compare_models(sort="Accuracy") # sort 기준 설정하기 
# fold: cross_validation의 fold를 지정 (default = 10)
# sort: 정렬기준 지표 설정 (default = Accuracy)
# n_select: 상위 n개의 모델 결과만 출력
bestmodel
# %%
# finalize_model(): 최종 모델로 설정 후 마지막 학습 진행
fmodel=finalize_model(bestmodel)
fmodel
# %%
# predict_model(): 예측 결과를 'Label' 변수에 저장
pred=predict_model(bestmodel, data=df.iloc[-100:])
pmean = pred['prediction_score'].mean()
pmean
# %%
