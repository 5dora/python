mp1='모듈프로퍼티'

# 모듈 내 함수
def make99():
    for i in range(1,10,1):
        print('')
        for j in range(1,10):
            ans=i*j
            if((i*j)%2==1):
                print(ans, '*', end='\t')
            else:
                print(ans, end='\t')

# 모듈 내 클래스
class person():
    def __init__(self, name, age):
        self.name=name #클래스 속성
        self.age=age
    def sayHello(self): #클래스의 메서드
        print('Hello '+self.name)
        print("I'm "+self.age+"years old")
anna=person('anna','19')
anna.sayHello()