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
# 형태소 분석기
from konlpy.tag import Kkma
# %%
kkma = Kkma()
# %%
res=kkma.pos('안녕하세요 그런데 여러분 만나서 반갑습니다.')
res

# %%
def getPOS(txt="안녕하세요 그런데 여러분 만나서 반갑습니다."):
    res=kkma.pos(txt)
    reqPos=['NNG','NNP','NP','VV','VA','VCN','JC','MAC','EFA','EFQ','EFO','EFA','EFI','EFR']#,'VCP','EFN'
    wset=[]
    for r in res:
        if(r[1] in reqPos):
            wset.append(r[0])
            print(r)
    return(' '.join(wset))
getPOS()
# %%
# 파일 읽기
fname='src\현진건-운수_좋은_날+B3356-개벽.txt'
with open(fname, encoding='utf8') as f:
    r=f.readlines()
print(r)
# %%
lucky = ''.join(r)
lucky[:100]
# %%
lucky= lucky.replace('\n\n','{nn}')
lucky = lucky.replace('\n', '')
print(lucky)
lucky= lucky.replace('{nn}','.')
print(lucky)
# %%
lucks = lucky.split('.')
lucks[:10]
# %%
copus=[]
for luck in lucks:
    ltxt=getPOS(luck)
    copus.append(ltxt)
copus[:10]

# %%
# t='새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추적추적 내리는 날이었다'
# getPOS(t)

# %%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 몇 번 나왔는지 세서 CountVectorizer
# 행렬에서 많이 나오는 거 제거 TfidfVectorizer
cvect = CountVectorizer()
cvfit = cvect.fit_transform(copus)
cvfit
cvtable = cvfit.toarray()
print(cvtable.shape)
print(cvect.vocabulary_)
# %%
plt.imshow(cvtable[:100,:100])
# %%
# TfidVectorized
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 빈도-역빈도 몇 번 나왔는지 세서 CountVectorizer
# 행렬에서 많이 나오는 거 제거 TfidfVectorizer
tvect = TfidfVectorizer()
tvfit = tvect.fit_transform(copus)
tvfit
tvtable = tvfit.toarray()
print(tvtable.shape)
print(tvect.vocabulary_)
# %%
plt.imshow(tvtable[:100,:100])
# %%
cdf = pd.DataFrame(cvtable[:100,:100])
vdf = pd.DataFrame(tvtable[:100,:100])
print(cdf)
print(vdf)
# %%
