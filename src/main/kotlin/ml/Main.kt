package ml

import com.github.ajalt.clikt.core.CliktCommand
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
import ml.defined.base.Config
import ml.defined.base.Learning
import ml.input.AutoDataPicker
import ml.learn.NetworkTeacher
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


    private val experimentName by option("--def", "-d", help = Help.experimentName)
        .enum<Learning.Type>()
        .default(Learning.Type.Transformation)

    private val separator: String? by option("--separator", "-s", help = Help.separator)

    private val stackTrace by option("--stack", help = Help.stack).flag(default = false)

    override fun run() {

        val mode =
            if (offline) NetworkTeacher.Mode.Offline else NetworkTeacher.Mode.Online

        try {

            val picker = AutoDataPicker(File(".\\data"))

            val pickedInput = if (inputs.isEmpty()) {
                picker.pickData(experimentName)
            } else {
                inputs
            }

            val config = Config(
                inputs = pickedInput,
                teacherMode = mode,
                separator = separator ?: picker.pickSeparator(experimentName)
            )

            Learning.get(experimentName, config).run()

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

    const val experimentName: String = "Selects an experiment."

    const val separator: String = "Sets data separator."

    const val stack: String = "Prints stack trace when error occurs"
}