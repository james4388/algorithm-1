import jnettool.tools.elements.NetworkElement
import jnettool.tools.Routing
import jnettool.tools.RouteInspector
import logging

ne = jnettool.tools.elements.NetworkElement('171.0.2.45')
try:
    routing_table = ne.getRoutingTable()

except jnettool.tools.elements.MissingVar:
    logging.exception('No routing table found')
    ne.cleanup('rollback')

else:
    num_routes = routing_table.getSize()
    for i in range(num_routes):
        route = routing_table.getRouteByIndex(i)
        name = route.getName()
        ipaddr = route.getIPAddr()
        print '%15s -> %s' % (ipaddr, name)
    ne.cleanup('commit') 

finally:
    ne.disconnect()











