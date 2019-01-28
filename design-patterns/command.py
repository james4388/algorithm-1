

# Client -> Invoker -> Command, ConcreteCommand -> Receiver
# Use case:
# - Keep history of request, implement callback function, handle request
# at different time, decouple object handling, undo functionality
# Advantage:
# - help in extensibility
# - create macro, sequence of command
