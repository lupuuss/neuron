package ml.defined.base

import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.output.ProgressFormatter
import java.io.File

data class Config(
    val inputs: List<File>,
    val teacherMode: NetworkTeacher.Mode,
    val printMode: NetworkProgressPrinter.Mode,
    val printType: NetworkProgressPrinter.Type,
    val printFormatter: ProgressFormatter,
    val separator: String
)