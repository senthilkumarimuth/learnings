def add(a,b):
    return a+b

dispatcher={'add':add}
w='add'
try:
    function=dispatcher[w]
except KeyError:
    raise ValueError('invalid input')
a=1
b=2
function(a,b)