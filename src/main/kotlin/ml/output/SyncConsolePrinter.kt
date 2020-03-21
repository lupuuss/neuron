package ml.output

import java.io.OutputStream

class SyncConsolePrinter(outputStream: OutputStream) : ConsolePrinter(outputStream) {
    override fun print(printable: Any?) {
        outputStream.write(printable.toString().toByteArray())
        outputStream.flush()
    }

    override fun println(printable: Any?) {
        outputStream.write("$printable\n".toByteArray())
        outputStream.flush()
    }
}