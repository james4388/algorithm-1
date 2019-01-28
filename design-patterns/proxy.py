

# Proxy object acts like real object, delegate all calls to real object
# Pefer object initialization until you need it, lazy initialization
# Use cases:
# - control access to another object
# - log all calls to subject
# - connect to subject which located on remote machine
# - instantiate a heavy object only when it's needed
# - temporarily store some calculation
# - count reference to an object
from abc import ABCMeta, abstractmethod
import random


class AbstractSubject(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def sort(self, reverse=False):
        pass


class RealSubject(AbstractSubject):
    def __init__(self):
        self.digits = []

        for i in xrange(1000000):
            self.digits.append(random.random())

    def sort(self, reverse=False):
        self.digits.sort()
        if reverse:
            self.digits.reverse()


class Proxy(AbstractSubject):

    reference_count = 0

    def __init__(self):
        if not getattr(self.__class__, 'cached_object', None):
            self.__class__.cached_object = RealSubject()
            print("Created new object")
        else:
            print("Use cached object")
        self.__class__.reference_count += 1
        print('Count reference....', self.__class__.reference_count)

    def sort(self, reverse=False):
        self.__class__.cached_object.sort(reverse=reverse)

    def __del__(self):
        self.__class__.reference_count -= 1

        if self.__class__.reference_count == 0:
            del self.__class__.cached_object

        print('Deleted object. Count of objects = ',
              self.__class__.reference_count)


if __name__ == '__main__':
    p1 = Proxy()

    p2 = Proxy()

    p3 = Proxy()
    p1.sort(reverse=True)

    print("deleting proxy 2")
    del p2
