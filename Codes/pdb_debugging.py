#adding breakpoint staring from python 3.7

s =[]
for i in range(10):
    i_temp = i+1
    #breakpoint()
    s.append(i_temp)

#print(s)

#pdb debugging

import pdb

def addition(a, b):
    answer = a ** b
    return answer

pdb.set_trace()
x = input("Enter first number : ")
y = input("Enter second number : ")
sum = addition(x, y)
print(sum)