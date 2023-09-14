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