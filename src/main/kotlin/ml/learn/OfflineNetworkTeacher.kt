package ml.learn

import ml.spine.Network

class OfflineNetworkTeacher(alpha: Double, beta: Double) : NetworkTeacher(alpha, beta) {

    override fun teach(network: Network) {
        TODO("Not yet implemented")
    }

    override fun verify(network: Network): List<Double> {
        TODO("Not yet implemented")
    }

}

