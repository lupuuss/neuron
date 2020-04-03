package ml.defined

import ml.defined.base.NetworksLearning
import ml.input.DataParser
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.spine.Activation
import ml.spine.Network
import java.util.*

class Iris(config: Config) : NetworksLearning(config) {
    override val errorGoal: Double = 0.1
    override val stepsLimit: Int = 200_000

    private lateinit var data: List<Pair<List<Double>, List<Double>>>

    private var progressPrinter: NetworkProgressPrinter? = null

    private lateinit var fieldMask: List<Boolean>
    private var hiddenNeurons: Int = 0

    override fun setup() {

        val parser = DataParser(
            ",",
            4,
            3
        )

        parser.lineTransformer = {
            it.replace("Iris-setosa", "1,0,0")
                .replace("Iris-versicolor", "0,1,0")
                .replace("Iris-virginica", "0,0,1")
        }


        println("Fields mask: ")
        val userInput = Scanner(System.`in`)

        while (!userInput.hasNext("[0,1]{4}")) {
            userInput.nextLine()
            println("Invalid mask!")
        }

        fieldMask = userInput.nextLine().map { it == '1' }

        println(fieldMask.count { it })

        println("Neurons: ")

        while (!userInput.hasNextInt()) {
            println("Integer is required!")
        }

        hiddenNeurons = userInput.nextInt()

        data = parser.parse(Scanner(config.inputs[0])).map {
            val result = mutableListOf<Double>()

            for (i in it.first.indices) {
                if (fieldMask[i]) {
                    result.add(it.first[i])
                }
            }

            result to it.second
        }

    }

    override fun buildNetworks(): List<Network> = listOf(
        Network.Builder()
            .name("Iris_${fieldMask.count { it }}_$hiddenNeurons")
            .inputs(fieldMask.count { it })
            .setDefaultActivation(Activation.Sigmoid)
            .hiddenLayer(hiddenNeurons, true)
            .outputLayer(3, true)
    )

    override fun buildTeachers(): List<NetworkTeacher> = listOf(
        NetworkTeacher.get(config.teacherMode, 0.0001, 0.1).apply {
            trainingSet = data.filterIndexed { index, _ -> index % 2 != 0 }
            verificationSet = data.filterIndexed { index, _ -> index % 2 == 0 }
        }
    )

    override fun unfreezing(): List<Network> = listOf(NetworkFreezer.unfreezeFile("Iris")!!)

    override fun beforeLearning(network: Network, teacher: NetworkTeacher) {
        progressPrinter = NetworkProgressPrinter(System.out)
        progressPrinter?.apply {
            type = config.printType
            mode = config.printMode
            formatter = config.printFormatter
            stepMetric = teacher.metric
        }
    }

    override fun eachLearningStep(network: Network, errorVector: List<Double>, steps: Int) {
        progressPrinter?.updateData(errorVector, steps)
    }

    override fun afterLearning(network: Network, errorVector: List<Double>?, steps: Int?) {
        progressPrinter?.close()
    }

    override fun allNetworksReady(restored: Boolean) {

        var bad = 0

        data.forEach { (input, expected) ->
            val answer = networks.first().answer(input)
            val v = answer.map { if (it == answer.max()) 1.0 else 0.0 }

            if (v != expected) {
                println(answer)
                println(v)
                println(expected)
                println()
                bad++
            }
        }

        println("$bad ${data.size}")
    }

}