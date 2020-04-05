package ml.defined.base

import java.io.File

data class Config(
    val inputs: List<File>,
    val separator: String
)