package ml

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.default
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.enum
import com.github.ajalt.clikt.parameters.types.file
import ml.defined.Config
import ml.defined.DefinedLearning
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.output.ProgressFormatter

class Main : CliktCommand() {

    private val input by argument(help = Help.input).file(mustExist = true, mustBeReadable = true)

    private val mode by argument(help = Help.teacherMode)
        .enum<NetworkTeacher.Mode>(ignoreCase = true)
        .default(NetworkTeacher.Mode.Online)

    private val definedNetworks by argument(help = Help.definedNetworks)
        .enum<DefinedLearning.Type>()
        .default(DefinedLearning.Type.Exercise3)

    private val separator: String by option("--separator", "-s", help = Help.separator).default(";")

    private val freeze by option("--freeze", "-f", help = Help.freeze).flag(default = false)

    private val unfreeze by option("--unfreeze", "-u", help = Help.unfreeze).flag(default = false)

    private val async by option(help = Help.async).flag("--sync", default = false)

    private val inPlace by option("--in-place", help = Help.inPlace).flag("--seq", default = false)

    private val fullFormatter by option("--full", help = Help.formatter).flag("--error-only", default = false)

    override fun run() {

        val printFormatter: ProgressFormatter = if (fullFormatter) {
            { errorStr, steps, metric -> "Error: $errorStr | $metric : $steps" }
        } else {
            { errorStr, _, _ -> errorStr ?: "null" }
        }

        val printMode =
            if (async) NetworkProgressPrinter.Mode.Async else NetworkProgressPrinter.Mode.Sync

        val printType =
            if (inPlace) NetworkProgressPrinter.Type.InPlace else NetworkProgressPrinter.Type.Sequential

        val config = Config(
            input = input,
            alwaysFresh = !unfreeze,
            freeze = freeze,
            teacherMode = mode,
            printMode = printMode,
            printType = printType,
            printFormatter = printFormatter,
            separator = separator
        )

        // For now just starts exercise 3 process.
        // More details in DefinedLearning and Exercise3 classes
        DefinedLearning.get(definedNetworks, config).run()
    }
}

fun main(args: Array<String>): Unit = Main().main(args)

object Help {

    const val input: String = "Destination of training data set."

    const val teacherMode: String = "Learning mode [online/offline]. Default value = online"

    const val definedNetworks: String = "Set of neural networks to run. Default value = Exercise3"

    const val separator: String = "Sets data separator. Default value = ';'"

    const val freeze: String = "If set neural networks will be freezed after program end."

    const val unfreeze: String = "If set, neural networks will be unfreezed if available instead of learning."

    const val async: String = "Switches between asynchronous and synchronous printing." +
            " The first one may omit some results, but it doesn't block learning." +
            " Synchronous, on the other hand, blocks learning but prints every result."
    const val inPlace: String = "Switches between in place and sequential printing." +
            " The first one prints every result updating a single line." +
            " Sequential printing puts every result in a new line."

    const val formatter: String = "Switches between full progress and only error printing."
}