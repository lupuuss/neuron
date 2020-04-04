package ml.input

import ml.defined.base.Learning
import ml.defined.base.DataNotFound
import java.io.File

class AutoDataPicker(private val searchDir: File) {

    private fun fileExistsAndReadableOrNull(dir: File, name: String): File? {
        val file = File(dir, name)

        return if (file.exists() && file.canRead()) {
            file
        } else {
            null
        }
    }

    @Throws(DataNotFound::class)
    fun pickData(taskType: Learning.Type): List<File> = when (taskType) {

        Learning.Type.Exercise3 -> {
            fileExistsAndReadableOrNull(searchDir, "in.txt")?.let { listOf(it) }
        }

        Learning.Type.Transformation, Learning.Type.Transformation100, Learning.Type.TransformationHidden -> {
            fileExistsAndReadableOrNull(searchDir, "trans_in.txt")?.let { listOf(it) }
        }

        Learning.Type.Approximation -> {
            val file1 = fileExistsAndReadableOrNull(searchDir, "approx_1.txt")
            val file2 = fileExistsAndReadableOrNull(searchDir, "approx_2.txt")
            val testFile = fileExistsAndReadableOrNull(searchDir, "approx_test.txt")

            if (testFile != null && file1 != null && file2 != null) {
                listOf(file1, file2, testFile)
            } else {
                null
            }
        }

        Learning.Type.Iris -> {
            fileExistsAndReadableOrNull(searchDir, "iris.data")?.let { listOf(it) }
        }
    } ?: throw DataNotFound()

    fun pickSeparator(taskType: Learning.Type): String = when (taskType) {

        Learning.Type.Exercise3 -> ";"

        Learning.Type.Transformation,
        Learning.Type.Transformation100,
        Learning.Type.TransformationHidden,
        Learning.Type.Approximation -> " "

        Learning.Type.Iris -> ","
    }
}