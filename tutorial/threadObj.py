import threading
import time
from random import randint, random
from collections import deque, defaultdict


class KeyValueStore(object):
    def __init__(self, max_connection=None, barrier=None):
        self.max_connection = max_connection or 1028
        self.store = {}
        self.connections = deque([])
        self.lock = threading.Lock()
        self.barrier = barrier
        self.sema_pool = threading.BoundedSemaphore(value=self.max_connection)

    def accept_connection(self, conn):
        with self.lock:
            time.sleep(random()*2)
            self.connections.append(conn)

    def close_connection(self, conn):
        with self.lock:
            if conn in self.connections:
                self.connections.remove(conn)

    def setKey(self, key, val):
        with self.sema_pool:
            time.sleep(random()*4)
            self.store[key] = val
            # print("Current store...", self.store)

    def getKey(self, key):
        val = 'No value'
        with self.sema_pool:
            val = self.store[key]
        return val

    def shutdown(self):
        self.connections = None


class Connection(object):
    def __init__(self, client_id=None):
        self.client_id = client_id
        self.db = None

    def connect(self, db):
        db.accept_connection(self)
        self.db = db

    def close(self):
        print("Closing connection...")
        self.db.close_connection(self)
        self.db = None

    def setKey(self, key, val):
        self.db.setKey(key, val)

    def getKey(self, key):
        return self.db.getKey(key)


def make_connection(db, congest):
    client_id = randint(0, 10)
    print("Making connection to store using id = %s" % client_id)
    conn = Connection(client_id)
    conn.connect(db)
    while congest.is_set():
        print("Sleep and wait for database less load...")
        time.sleep(random()*2)
    conn.setKey(client_id, "hello from {}".format(client_id))
    print("key set...", conn.getKey(client_id))
    conn.close()


def monitor_server(db, event):
    while True:
        with db.lock:
            if len(db.connections) >= db.max_connection:
                event.set()
            else:
                event.clear()
        time.sleep(2)


if __name__ == '__main__':
    event = threading.Event()
    barrier = threading.Barrier(parties=10)
    db = KeyValueStore(max_connection=4, barrier=barrier)
    monitor = threading.Thread(target=monitor_server, args=(db, event))
    monitor.start()

    for i in range(10):
        thread = threading.Thread(target=make_connection, args=(db, event))
        thread.start()
