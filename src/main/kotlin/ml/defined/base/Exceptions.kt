package ml.defined.base

class UnfulfilledExpectationsException(msg: String) : Exception(msg)

class DataNotFound : Exception(
    "Data could not be loaded automatically! You have to set it manually by arguments. Add --help for more instructions."
)

class ParsingException(msg: String, cause: Throwable? = null) : java.lang.Exception(msg, cause)