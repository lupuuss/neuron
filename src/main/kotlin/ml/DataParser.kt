package ml

import java.lang.Exception
import java.util.*

class ParsingException(msg: String, cause: Throwable? = null) : Exception(msg, cause)

class DataParser(
    private var separator: String,
    private var inputs: Int,
    private var expected: Int
) {

    var lineTransformer: (String) -> String = { it }

    @Throws(ParsingException::class)
    fun parse(scanner: Scanner): List<Pair<List<Double>, List<Double>>> {

        val result = mutableListOf<Pair<List<Double>, List<Double>>>()

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