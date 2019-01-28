class SingletonDecorator:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance


class foo:
    pass

foo = SingletonDecorator(foo)

x = foo()
y = foo()
z = foo()
x.val = 1
y.val = 2
z.val = 3

m = Singleton()
m.x = 100
n = Singleton()
n.x = 99

print(m.x, n.x)
