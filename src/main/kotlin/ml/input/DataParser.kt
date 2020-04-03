package ml.input

import ml.defined.base.ParsingException
import java.io.File
import java.lang.Exception
import java.util.*

class DataParser(
    var separator: String,
    var lineTransformer: (String) -> String = { it }
) {

    @Throws(ParsingException::class)
    fun parse(
        file: File,
        inputs: Int,
        expected: Int
    ): List<Pair<List<Double>, List<Double>>> {

        val result = mutableListOf<Pair<List<Double>, List<Double>>>()
        val scanner = Scanner(file)

        try {

            while (scanner.hasNextLine()) {

                val line = lineTransformer(scanner.nextLine())

                if (line.isEmpty()) continue

                val values = line.split(separator).map { it.toDouble() }

                result.add(values.subList(0, inputs) to values.subList(inputs, inputs + expected))
            }

            return result

        } catch (e: Exception) {

            throw ParsingException(
                "Parsing failed! You probably picked wrong separator or" +
                        " this neural network requires more inputs!",
                e
            )
        }
    }
}