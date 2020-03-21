package ml.output

import java.io.OutputStream

typealias ProgressFormatter = (Double?, Int, String?) -> String

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

    var stepCounter: Int = 0

    var currentError: Double? = null

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
        output = formatter(currentError, stepCounter, stepMetric)
    }

    fun updateData(error: Double?, steps: Int) {
        this.currentError = error
        this.stepCounter = steps

        updateOutput()
    }

    fun close() {
        consolePrinter.close()
    }
}