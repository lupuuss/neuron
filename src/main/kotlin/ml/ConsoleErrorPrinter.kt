package ml

import java.io.OutputStream
import java.nio.charset.Charset

class ConsoleErrorPrinter(private val stream: OutputStream) {

    private val errorMsg = "Error:"
    private var previousMsgSize = 0
    private var currentErrorString = "$errorMsg -"

    fun update(error: Double?) {
        currentErrorString = "$errorMsg ${error ?: '-'}"
    }

    private fun erasingLastMsgString(size: Int) =
        if (size == 0) "" else "\r" + generateSequence { ' ' }.take(size).joinToString() + "\r"

    fun print() {
        val erasingString = erasingLastMsgString(previousMsgSize)

        erasingString.toByteArray(Charset.defaultCharset()).forEach { stream.write(it.toInt()) }

        previousMsgSize = currentErrorString.length
        currentErrorString.chars().forEach(stream::write)

        stream.flush()
    }
}