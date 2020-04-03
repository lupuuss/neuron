package ml.input

import ml.defined.DefinedLearning
import ml.defined.exceptions.DataNotFound
import java.io.File
import java.lang.IllegalStateException

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
    fun pickData(taskType: DefinedLearning.Type): List<File> = when (taskType) {

        DefinedLearning.Type.Exercise3 -> {
            fileExistsAndReadableOrNull(searchDir, "in.txt")?.let { listOf(it) }
        }

        DefinedLearning.Type.Transformation -> {
            fileExistsAndReadableOrNull(searchDir, "trans_in.txt")?.let { listOf(it) }
        }

        DefinedLearning.Type.Approximation -> {
            val file1 = fileExistsAndReadableOrNull(searchDir, "approx_1.txt")
            val file2 = fileExistsAndReadableOrNull(searchDir, "approx_2.txt")
            val testFile = fileExistsAndReadableOrNull(searchDir, "approx_test.txt")

            if (testFile != null && (file1 != null || file2 != null)) {
                listOf(testFile, file1 ?: file2!!)
            } else {
                null
            }
        }

        DefinedLearning.Type.Iris -> {
            fileExistsAndReadableOrNull(searchDir, "iris.data")?.let { listOf(it) }
        }
    } ?: throw DataNotFound()
}