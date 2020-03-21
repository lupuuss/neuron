package ml.output

import java.io.OutputStream

abstract class ConsolePrinter(protected val outputStream: OutputStream) {

    abstract fun print(printable: Any?)

    abstract fun println(printable: Any?)

    open fun close() {}
}