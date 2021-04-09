class learningclass():

    villege ='vethanakipuram'  #class variable 1
    District ='Thanjavur'  # class variable 2

    def __init__(self,age,height):
        self.age=age   #instance variable 1
        self.height=height # instance variable 2
    def print_age(self):  #instance method
        print(self.age)
    def print_color(self,color):
        print(self.color)
    def samemethod(self):
        print('i am from method A')
    def samemethod(self):
        print('i am from method A 2nd')
    @classmethod
    def class_info(cls): # class method
        print('This class is about people name ang age')

class B(learningclass): # class inheritance
    def samemethod(self): #poly morphism
        print('i am from method b')


#creating objects

suganthi = learningclass(23,5.1)
senthil =  learningclass(32,5.4)
suganthi.print_age()
suganthi.print_color('green')
#learningclass.print_age('suganthi',age)
senthil.print_age()

#class inheritance

var1=B(23,2)
var1.samemethod()

'''Since class variables become a part of every instance, you can just call  self to access them. Alternatively, you can explicitly call self.__class__.__var2 to make it clear where the intended variable is actually stored.'''