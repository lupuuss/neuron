package ml.defined

import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.output.ProgressFormatter
import java.io.File

data class Config(
    val input: File,
    val alwaysFresh: Boolean,
    val freeze: Boolean,
    val alpha: Double,
    val beta: Double,
    val teacherMode: NetworkTeacher.Mode,
    val printMode: NetworkProgressPrinter.Mode,
    val printType: NetworkProgressPrinter.Type,
    val printFormatter: ProgressFormatter,
    val errorGoal: Double,
    val separator: String
)