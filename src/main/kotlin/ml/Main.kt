package ml

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.default
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.choice
import com.github.ajalt.clikt.parameters.types.enum
import com.github.ajalt.clikt.parameters.types.file
import com.github.ajalt.clikt.parameters.types.int
import ml.defined.Config
import ml.defined.DefinedLearning
import ml.defined.Exercise3
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.output.ProgressFormatter
import ml.spine.Activation
import ml.spine.Network
import org.knowm.xchart.style.markers.None
import java.awt.Color
import java.util.*

class Main : CliktCommand() {

    private val inputFile by argument(help = "Destination of training data set.").file(
        mustExist = true,
        mustBeReadable = true
    )

    private val mode by argument(help = "Learning mode [online/offline]. Default value = 'online'")
        .enum<NetworkTeacher.Mode>(ignoreCase = true)
        .default(NetworkTeacher.Mode.Online)

    private val definedSet by argument(help = "Chosen set of neural networks. Default value = Exercise3")
        .enum<DefinedLearning.Type>()
        .default(DefinedLearning.Type.Exercise3)

    private val separator: String by option(
        "--separator",
        "-s",
        help = "Sets data separator. Default value = ';'"
    ).default(";")

    private val freeze by option(help = "If set neural networks will be freezed after program end.")
        .flag("-f", default = false)

    private val alwaysFresh by option(help = "If set, neural networks will be taught even if freezed version is available.")
        .flag("-af", default = true)

    private val printMode by option(help = "Sets print mode. Default value = Sync")
        .enum<NetworkProgressPrinter.Mode>()
        .default(NetworkProgressPrinter.Mode.Sync)

    private val printType by option(help = "Sets print type. Default value = Sequential")
        .enum<NetworkProgressPrinter.Type>()
        .default(NetworkProgressPrinter.Type.Sequential)

    private val formatter by option(help = "Sets print formatter. Default value = ErrorOnly")
        .choice("ErrorOnly", "Full")
        .default("ErrorOnly")

    override fun run() {

        val printFormatter: ProgressFormatter = when (formatter) {
            "ErrorOnly" -> { errorStr, _, _ -> errorStr ?: "null" }
            "Full" -> { errorStr, steps, metric -> "Error: $errorStr | $metric : $steps"}
            else -> throw IllegalAccessException("Formatter not found!")
        }

        val config = Config(
            input = inputFile,
            alwaysFresh = alwaysFresh,
            freeze = freeze,
            teacherMode = mode,
            printMode = printMode,
            printType = printType,
            printFormatter = printFormatter,
            separator = separator
        )

        // For now just starts exercise 3 process.
        // More details in DefinedLearning and Exercise3 classes
        DefinedLearning.get(definedSet, config).run()
    }
}

fun main(args: Array<String>): Unit = Main().main(args)