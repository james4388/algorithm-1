'Demonstrate the Adapter Design Pattern using context managers, sequence methods, repr and properties'

import jnettool.tools.elements.NetworkElement
import jnettool.tools.Routing
import jnettool.tools.RouteInspector
import logging

class NetworkElement(object):
    'Adapter for jnet network element'

    def __init__(self, ipaddr):
        self.ipaddr = ipaddr
        self.oldne = jnettool.tools.elements.NetworkElement(ipaddr)

    def __enter__(self):
        return self

    def __exit__(self, exctype, excinst, exctb):
        if exctype == jnettool.tools.elements.MissingVar:
            logging.exception('No routing table found')
            self.oldne.cleanup('rollback')
            self.oldne.disconnect()
            return True
        elif exctype is not None:
            self.oldne.disconnect()
            return False
        else:
            self.oldne.cleanup('commit')
            self.oldne.disconnect()

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.ipaddr)

    @property
    def routing_table(self):
        return self.oldne.getRoutingTable()

class RoutingTable(object):

    def __init__(self, old_table):
        self.old_table = old_table

    def __len__(self):
        return self.old_table.getSize()

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError('Routing table offset is out of range')
        return self.old_table.getRouteByIndex(i)

    @property
    def name(self):
        return self.old_table.getName()

    @property
    def ipaddr(self):
        return self.old_table.getIPAddr()
