# 블록별로 실행가능
#%%
print("hello world")
#%%
a=3
b=5
c=a+b
print(c)
#%%
# 단열 주석
'''문자열이지만 다열주석으로 사용가능'''
print('나머지연산 :', b%a)
'몫연산: ', b//a #몫연산 print 생략해도 주피터상에서는 출력
#%%
# 데이터 셋: array
    # 리스트: [] - 키 값없는 리스트
    # 튜플: () - 수정 불가능한 리스트
    # 딕셔너리: {키:밸류} - 연관배열: 객체의 개념
    # 셋: set() - 집합
#%%
# def 함수명():
def add(a=2, b=4):
    return a+b

print(add(2,3))
print(add())

# %%
# 클래스 선언
class person():
    def sayHello(self): #클래스의 메서드이다. self로
        print('Hello')
        # 일반함수는 self없고
        # 메서드 self
        # 캡슐화 하고 싶다? _넣기 def __sayHello__(self)
anna=person()
anna.sayHello()

# %%
class person():
    def __init__(self, name, age):
        self.name=name #클래스 속성
        self.age=age
    def sayHello(self): #클래스의 메서드
        print('Hello '+self.name)
        print("I'm "+self.age+"years old")
anna=person('anna','19')
anna.sayHello()

# %%
# 구구단 만들기
def make99():
    for i in range(1,10,1):
        print('')
        for j in range(1,10):
            ans=i*j
            if((i*j)%2==1):
                print(ans, '*', end='\t')
            else:
                print(ans, end='\t')
