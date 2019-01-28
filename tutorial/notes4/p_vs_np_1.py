# XXX -- Top level review comments:
#
# * Nice exception recovery and logging.
#
# * Please cleanup code formatting.
#   This is a little rough on my eyes.
#
# * Should we use this as template for other
#   short network element scripts?
#
# -- Thanks.   The Boss :-)

import jnettool.tools.elements.NetworkElement, \
       jnettool.tools.Routing, \
       jnettool.tools.RouteInspector
import    logging

ne=jnettool.tools.elements.NetworkElement( '171.0.2.45' )
try:
    routing_table=ne.getRoutingTable()  # fetch table

except jnettool.tools.elements.MissingVar:
  # Record table fault
  logging.exception( '''No routing table found''' )
  # Undo partial changes
  ne.cleanup( '''rollback''' )

else:
   num_routes=routing_table.getSize()  # determine table size
   for RToffset in range( num_routes ):
          route=routing_table.getRouteByIndex( RToffset )
          name=route.getName()       # route name
          ipaddr=route.getIPAddr()          # ip address
          print "%15s -> %s"% (ipaddr, name) # format nicely
   ne.cleanup( '''commit''' ) # lockin changes
finally:
    ne.disconnect()











