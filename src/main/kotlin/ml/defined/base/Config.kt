package ml.defined.base

import ml.learn.NetworkTeacher
import java.io.File

data class Config(
    val inputs: List<File>,
    val teacherMode: NetworkTeacher.Mode,
    val separator: String
)