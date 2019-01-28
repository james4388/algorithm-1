

# Advatanges of using facade pattern in subsystem
# - maintains loose coupling between client and subsystem
# - provides interface to set of interfaces in subsystem
# - wraps complicated subsystem with simpler interface
# - subsystem gains flexibility and clients gain simplicity
# Problems solved:
# - make software easier to test and use
# - reduce dependency of using external code
# - provide better and clearer API for client code
# Implementation: create facade class to provide simple method that call 
# subsystem's API
