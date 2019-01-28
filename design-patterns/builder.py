

# Use case: create complex object should be independent of the part make up
# object. Construction process allows different representation of the object
# Director: construct_process() -> Builder -> ConcreteBuilder: buildPart()
# e.g: RTFReader -> TextConverter -> ASCIIConverter, TeXText, TextWidget
