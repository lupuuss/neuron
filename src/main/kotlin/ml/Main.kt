package ml

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.IncorrectArgumentValueCount
import com.github.ajalt.clikt.core.PrintMessage
import com.github.ajalt.clikt.core.context
import com.github.ajalt.clikt.output.CliktHelpFormatter
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.multiple
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.enum
import com.github.ajalt.clikt.parameters.types.file
import ml.defined.Config
import ml.defined.DefinedLearning
import ml.defined.exceptions.DataNotFound
import ml.input.AutoDataPicker
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.output.ProgressFormatter
import java.io.File
import java.lang.Exception

open class Main : CliktCommand(
    printHelpOnEmptyArgs = true
) {

    init {
        context {
            helpFormatter = CliktHelpFormatter(showDefaultValues = true)
        }
    }

    private val inputs by argument(help = Help.input)
        .file(mustExist = true, mustBeReadable = true)
        .multiple(required = false)

    private val offline by option("--off", "--offline", help = Help.offline)
        .flag("--on", "--online", default = false)


    private val definedNetworks by option("--def", "-d", help = Help.definedNetworks)
        .enum<DefinedLearning.Type>()
        .default(DefinedLearning.Type.Exercise3)

    private val separator: String by option("--separator", "-s", help = Help.separator).default(";")

    private val freeze by option("--freeze", "-f", help = Help.freeze).flag(default = false)

    private val unfreeze by option("--unfreeze", "-u", help = Help.unfreeze).flag(default = false)

    private val async by option("--print-async", help = Help.async).flag("--print-sync", default = false)

    private val inPlace by option("--print-in-place", help = Help.inPlace).flag("--print-seq", default = false)

    private val fullFormatter by option("--full", help = Help.formatter).flag("--error-only", default = false)

    private val stackTrace by option("--stack", help = Help.stack).flag(default = false)

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

        val mode =
            if (offline) NetworkTeacher.Mode.Offline else NetworkTeacher.Mode.Online

        val pickedInput = try {

            val picker = AutoDataPicker(File(".\\data"))

            if (inputs.isEmpty()) {
                picker.pickData(definedNetworks)
            } else {
                inputs
            }
        } catch (e: Exception) {
            throw IncorrectArgumentValueCount(
                this.registeredArguments().find { it.name == "inputs" }!!,
                this.currentContext
            )
        }

        val config = Config(
            inputs = pickedInput,
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
        try {

            DefinedLearning.get(definedNetworks, config).run()

        } catch (e: Exception) {

            if (stackTrace) {
                e.printStackTrace()
            }

            throw PrintMessage("Error occurred: ${e.message}")
        }
    }
}

fun main(args: Array<String>): Unit = Main().main(args)

object Help {

    const val input: String = "Destinations of training data files."

    const val offline: String = "Switches between online and offline learning"

    const val definedNetworks: String = "Selects a set of neural networks."

    const val separator: String = "Sets data separator."

    const val freeze: String = "If set neural networks will be freezed after program end."

    const val unfreeze: String = "If set, neural networks will be unfreezed if available instead of learning."

    const val async: String = "Switches between asynchronous and synchronous printing." +
            " The first one may omit some results, but it doesn't block learning." +
            " Synchronous, on the other hand, blocks learning but prints every result."
    const val inPlace: String = "Switches between in place and sequential printing." +
            " The first one prints every result updating a single line." +
            " Sequential printing puts every result in a new line."

    const val formatter: String = "Switches between full progress and only error printing."

    const val stack: String = "Prints stack trace when error occurs"
}