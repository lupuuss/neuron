package ml.output

import java.io.OutputStream

typealias ProgressFormatter = (String?, Int, String?) -> String

class NetworkProgressPrinter(private val outputStream: OutputStream) {

    enum class Type {
        Sequential, InPlace
    }

    enum class Mode {
        Sync, Async
    }


    var type: Type = Type.InPlace
    var mode: Mode = Mode.Async
        set(value) {

            if (value != field) {
                this.consolePrinter.close()
                this.consolePrinter = resolveMode(value)
            }
            field = value
        }

    var formatter: ProgressFormatter =
        { error, steps, metric -> "Error: $error | $metric: $steps" }

    private var consolePrinter: ConsolePrinter = resolveMode(mode)

    var stepMetric: String? = null
        set(value) {
            field = value
            updateOutput()
        }

    private var stepCounter: Int = 0

    private var currentError: List<Double>? = null

    private var output: String? = null
        set(value) {
            field = value

            when (type) {
                Type.Sequential -> consolePrinter.println(value)
                Type.InPlace -> consolePrinter.print("\r$value")
            }
        }

    private fun resolveMode(mode: Mode): ConsolePrinter = when (mode) {
        Mode.Sync -> SyncConsolePrinter(outputStream)
        Mode.Async -> AsyncConsolePrinter(outputStream)
    }

    private fun updateOutput() {
        output = formatter(currentError?.joinToString(separator = " "), stepCounter, stepMetric)
    }

    fun updateData(error: List<Double>?, steps: Int) {
        this.currentError = error
        this.stepCounter = steps

        updateOutput()
    }

    fun close() {
        consolePrinter.close()

        if (type == Type.InPlace) {
            outputStream.write('\n'.toInt())
        }
    }
}