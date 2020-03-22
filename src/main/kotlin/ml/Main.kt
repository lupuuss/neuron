package ml

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.default
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.flag
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.enum
import com.github.ajalt.clikt.parameters.types.file
import com.github.ajalt.clikt.parameters.types.int
import ml.defined.Config
import ml.defined.Exercise3
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
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

    private val mode: NetworkTeacher.Mode by argument(help = "Learning mode [online/offline]. Default value = 'online'")
        .enum(ignoreCase = true)

    private val separator: String by option(
        "--separator",
        "-s",
        help = "Sets data separator. Default value = ';'"
    ).default(";")

    private val freeze by option(help = "").flag("-f", default = false)

    override fun run() {

        val config = Config(
            input = inputFile,
            alwaysFresh = true,
            freeze = freeze,
            alpha = 0.1,
            beta = 0.0,
            teacherMode = NetworkTeacher.Mode.Online,
            printMode = NetworkProgressPrinter.Mode.Sync,
            printType = NetworkProgressPrinter.Type.Sequential,
            printFormatter = { errorStr, _, _ -> errorStr ?: "null" },
            errorGoal = 0.01,
            separator = separator
        )

        Exercise3(config).run()
    }
}

fun main(args: Array<String>): Unit = Main().main(args)