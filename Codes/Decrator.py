#Python Decorator function is a function that adds functionality to another, but does not modify it.

# Normal implemention

def dec(fun):
  def wrap():
     print('wrapper starging')
     fun()
     print('wrapper ending')
  return wrap
   
def hello():
    print('printed from function')
    
newfu=dec(hello)
newfu()

# With pie syntax implemention

def dec(fun):
  def wrap():
     print('wrapper starging')
     fun()
     print('wrapper ending')
  return wrap
@dec   
def hello():
    print('printed from function')
    
hello()
