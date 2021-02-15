/**
 * @author Andrey Larionov
 */
#include "Simulation.h"
#include "Marshal.h"
#include <chrono>
#include <random>
#include <iostream>


namespace cqumo {

// Class NodeData
// --------------------------------------------------------------------------

NodeData::NodeData(const NodeJournal &records)
        : systemSize(records.systemSize()->pmf()),
          queueSize(records.queueSize()->pmf()),
          serverSize(records.serverSize()->pmf()),
          delays(*records.delays()),
          departures(*records.departures()),
          waitTime(*records.waitTimes()),
          responseTime(*records.responseTimes()),
          numPacketsGenerated(records.numPacketsGenerated()->value()),
          numPacketsDelivered(records.numPacketsDelivered()->value()),
          numPacketsLost(records.numPacketsLost()->value()),
          numPacketsArrived(records.numPacketsArrived()->value()),
          numPacketsServed(records.numPacketsServed()->value()),
          numPacketsDropped(records.numPacketsDropped()->value()) {
    unsigned numPacketsProcessed = numPacketsDelivered + numPacketsLost;
    lossProb = numPacketsProcessed != 0
            ? static_cast<double>(numPacketsLost) / numPacketsProcessed
            : 0.0;
    dropProb = numPacketsArrived != 0
            ? static_cast<double>(numPacketsDropped) / numPacketsArrived
            : 0.0;
    deliveryProb = 1.0 - lossProb;
}


// Class SimData
// --------------------------------------------------------------------------

SimData::SimData(
        const NetworkJournal &journal,
        double simTime,
        double realTimeMs)
{
    for (auto &addrNodeRec: journal.nodeJournals()) {
        auto address = addrNodeRec.first;
        auto records = addrNodeRec.second;
        nodeData[address] = NodeData(*records);
    }
    numPacketsGenerated = journal.numPacketsGenerated()->value();
    this->simTime = simTime;
    this->realTimeMs = realTimeMs;
}


// Simulation functions
// --------------------------------------------------------------------------

SimData simMM1(
        double arrivalRate,
        double serviceRate,
        int queueCapacity,
        int maxPackets) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto gen = std::default_random_engine(seed);
    struct Context {
        std::default_random_engine *gen = nullptr;
        std::exponential_distribution<double> fn;
    };
    Context arrivalContext = {&gen, std::exponential_distribution<double>(
            arrivalRate)};
    Context serviceContext = {&gen, std::exponential_distribution<double>(
            serviceRate)};
    auto intervalBuilder = [](Context *ctx) {
        return ContextFunctor([](void *ctx) {
            auto ctx_ = static_cast<Context *>(ctx);
            return ctx_->fn(*(ctx_->gen));
        }, ctx);
    };
    auto arrival = intervalBuilder(&arrivalContext);
    auto service = intervalBuilder(&serviceContext);
    return simGG1(arrival, service, queueCapacity, maxPackets);
}


SimData simGG1(
        const DblFn &arrival,
        const DblFn &service,
        int queueCapacity,
        int maxPackets) {
    auto startedAt = std::chrono::system_clock::now();
    auto network = buildOneHopeNetwork(arrival, service, queueCapacity);
    auto journal = new NetworkJournal;
    for (auto &addrNodePair: network->nodes()) {
        journal->addNodeJournal(addrNodePair.second);
    }
    auto system = new System;

    // Execute main loop
    runMainLoop(network, system, journal, maxPackets);
    journal->commit();

    // Build node data
    double realTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - startedAt).count();
    auto simData = SimData(*journal, system->time(), realTimeMs);

    // Clear
    delete network;
    delete journal;
    delete system;
    return simData;
}


// TODO: refactor, remove duplicated code
SimData simTandem(
        DblFn arrival,
        const std::vector<DblFn>& services,
        int queueCapacity,
        int maxPackets) {
    auto startedAt = std::chrono::system_clock::now();
    auto network = buildTandemNetwork(arrival, services, queueCapacity);
    auto journal = new NetworkJournal;
    for (auto &addrNodePair: network->nodes()) {
        journal->addNodeJournal(addrNodePair.second);
    }
    auto system = new System;

    // Execute main loop
    runMainLoop(network, system, journal, maxPackets);
    journal->commit();

    // Build node data
    double realTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - startedAt).count();
    auto simData = SimData(*journal, system->time(), realTimeMs);

    // std::cout << toYaml(simData) << std::endl;

    // Clear
    delete network;
    delete journal;
    delete system;

    return simData;
}

}
