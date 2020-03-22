package ml.output

import kotlinx.coroutines.*
import java.io.OutputStream
import java.util.concurrent.atomic.AtomicReference

class AsyncConsolePrinter(output: OutputStream) : ConsolePrinter(output) {

    private var printable: AtomicReference<Any?> = AtomicReference(null)
    private val delay = 10L


    private val coroutine = GlobalScope.launch {

        var lastReference: Any? = null

        withContext(Dispatchers.IO) {
            outputStream.write((printable.get().toString()).toByteArray())
        }

        while (true) {
            delay(delay)

            val ref = printable.get()

            if (ref == lastReference) {
                continue
            }

            lastReference = ref

            withContext(Dispatchers.IO) {
                outputStream.write(ref.toString().toByteArray())
            }
        }
    }

    override fun print(printable: Any?) {
        this.printable.set(printable)
    }

    override fun println(printable: Any?) {
        this.printable.set("$printable\n")
    }

    override fun close() {
        coroutine.cancel()
    }
}