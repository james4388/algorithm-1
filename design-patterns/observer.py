

# Use case: change in one object leads to change in other objects
# Maintain loose coupling between subject and observer, subject only knows
# list of observer and their interfaces
# Broadcast messages between subject and observers
# Subject can keep any number of observers
import time
import datetime
from abc import ABCMeta, abstractmethod


class Subject(object):
    def __init__(self):
        self.observers = []
        self.cur_time = None

    def register_observer(self, observer):
        if observer in self.observers:
            print(observer, ' already in subscription')
        else:
            self.observers.append(observer)

    def unregister_observer(self, observer):
        try:
            self.observers.remove(observer)
        except ValueError:
            print('No such observer in subject')

    def notify_observers(self):
        self.cur_time = time.time()
        for observer in self.observers:
            observer.notify(self.cur_time)


class Observer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def notify(self, unix_timestamp):
        pass


class USATimeObserver(Observer):
    def __init__(self, name):
        self.name = name

    def notify(self, unix_timestamp):
        t = (datetime.datetime.fromtimestamp(int(unix_timestamp))
             .strftime('%Y-%m-%d %I:%M:%S%p'))
        print('Observer', self.name, ' says:', t)


class EUTimeObserver(Observer):
    def __init__(self, name):
        self.name = name

    def notify(self, unix_timestamp):
        t = (datetime.datetime.fromtimestamp(int(unix_timestamp))
             .strftime('%Y-%m-%d %H:%M:%S'))
        print('Observer', self.name, ' says:', t)


if __name__ == '__main__':
    subject = Subject()

    print('Adding usa time observer...')
    ob1 = USATimeObserver('usa_time_observer')
    subject.register_observer(ob1)
    subject.notify_observers()

    time.sleep(2)
    print('Adding EU time observer...')
    ob2 = EUTimeObserver('eu_time_observer')
    subject.register_observer(ob2)
    subject.notify_observers()

    time.sleep(2)
    print("Removing us time observer....")
    subject.unregister_observer(ob1)
    subject.notify_observers()
