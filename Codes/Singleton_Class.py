class OnlyOne:
    class __OnlyOne:
        def __init__(self, arg):
            self.val = arg
        def __str__(self):
            return repr(self) + self.val
    instance = None
    def __init__(self, arg):
        if not OnlyOne.instance:
            OnlyOne.instance = OnlyOne.__OnlyOne(arg)
            print(OnlyOne.instance,'instance')

        else:
            OnlyOne.instance.val = arg
            print(OnlyOne.instance,'instance')
            print(OnlyOne.instance.val,'instance.val')

    def __getattr__(self, name):
        return getattr(self.instance, name)

x = OnlyOne('sausage')
#print(x)
y = OnlyOne('eggs')
#print(y)
z = OnlyOne('spam')
#print(z)
