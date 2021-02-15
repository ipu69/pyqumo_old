/**
 * @author Andrey Larionov
 */
#include "Journals.h"
#include "Components.h"
#include <sstream>

namespace cqumo {

// Class NetworkJournal
// --------------------------------------------------------------------------
NetworkJournal::NetworkJournal(
        unsigned int windowSize,
        unsigned int numMoments,
        double time) :
        windowSize_(windowSize),
        numMoments_(numMoments),
        initTime_(time),
        numPacketsGenerated_(new Counter(0))
{}

NetworkJournal::~NetworkJournal() {
    for (auto &kv: nodeRecordsMap_) {
        delete kv.second;
    }
    delete numPacketsGenerated_;
}

void NetworkJournal::addNodeJournal(Node *node) {
    auto address = node->address();
    auto it = nodeRecordsMap_.find(address);
    if (it != nodeRecordsMap_.end()) {
        delete it->second;
    }
    nodeRecordsMap_[address] = new NodeJournal(this, node, initTime_);
}

void NetworkJournal::reset(double time) {
    for (auto &addrRecordsPair: nodeRecordsMap_) {
        addrRecordsPair.second->reset(time);
    }
    delete numPacketsGenerated_;
    numPacketsGenerated_ = new Counter(0);
}

void NetworkJournal::commit() {
    for (auto &addrRecordsPair: nodeRecordsMap_) {
        addrRecordsPair.second->commit();
    }
}

std::string NetworkJournal::toString() const {
    std::stringstream ss;
    ss << "(NetworkJournal: windowSize=" << windowSize_
       << ", numMoments=" << numMoments_
       << ", initTime=" << initTime_
       << ", records={";
    bool first = true;
    for (auto &addrRecordsPair: nodeRecordsMap_) {
        if (!first) ss << ", "; else first = false;
        ss << addrRecordsPair.first << ": "
           << addrRecordsPair.second->toString();
    }
    ss << "}";
    return ss.str();
}


// Class NodeJournal
// --------------------------------------------------------------------------
NodeJournal::NodeJournal(NetworkJournal *journal, Node *node, double time)
: node_(node), networkJournal_(journal)
{
    build(time);
}

NodeJournal::~NodeJournal() {
    clean();
}

void NodeJournal::reset(double time) {
    clean();
    build(time);
}

void NodeJournal::commit() {
    if (delays_) delays_->commit();
    if (departures_) departures_->commit();
    if (waitTimes_) waitTimes_->commit();
    if (responseTimes_) responseTimes_->commit();
}

void NodeJournal::clean() {
    delete systemSize_;
    systemSize_ = nullptr;
    delete queueSize_;
    queueSize_ = nullptr;
    delete serverSize_;
    serverSize_ = nullptr;
    delete delays_;
    delays_ = nullptr;
    delete departures_;
    departures_ = nullptr;
    delete waitTimes_;
    waitTimes_ = nullptr;
    delete responseTimes_;
    responseTimes_ = nullptr;
    delete numPacketsGenerated_;
    numPacketsGenerated_ = nullptr;
    delete numPacketsDelivered_;
    numPacketsDelivered_ = nullptr;
    delete numPacketsLost_;
    numPacketsLost_ = nullptr;
    delete numPacketsArrived_;
    numPacketsArrived_ = nullptr;
    delete numPacketsServed_;
    numPacketsServed_ = nullptr;
    delete numPacketsDropped_;
    numPacketsDropped_ = nullptr;
}

void NodeJournal::build(double time) {
    auto numMoments = networkJournal_->numMoments();
    auto windowSize = networkJournal_->windowSize();

    systemSize_ = new TimeSizeSeries(time, node_->size());
    queueSize_ = new TimeSizeSeries(time, node_->queue()->size());
    serverSize_ = new TimeSizeSeries(time, node_->server()->size());
    delays_ = new Series(numMoments, windowSize);
    departures_ = new Series(numMoments, windowSize);
    waitTimes_ = new Series(numMoments, windowSize);
    responseTimes_ = new Series(numMoments, windowSize);
    numPacketsGenerated_ = new Counter(0);
    numPacketsDelivered_ = new Counter(0);
    numPacketsLost_ = new Counter(0);
    numPacketsArrived_ = new Counter(0);
    numPacketsServed_ = new Counter(0);
    numPacketsDropped_ = new Counter(0);
}

std::string NodeJournal::toString() const {
    std::stringstream ss;
    ss << "(NodeJournal: address=" << node_->address()
       << ", systemSize=" << systemSize_->toString()
       << ", queueSize=" << queueSize_->toString()
       << ", serverSize=" << serverSize_->toString()
       << ", delays=" << delays_->toString()
       << ", departures=" << departures_->toString()
       << ", waitTimes=" << waitTimes_->toString()
       << ", responseTimes=" << responseTimes_->toString()
       << ", numPacketsGenerated=" << numPacketsGenerated_->toString()
       << ", numPacketsDelivered=" << numPacketsDelivered_->toString()
       << ", numPacketsLost=" << numPacketsLost_->toString()
       << ", numPacketsArrived=" << numPacketsArrived_->toString()
       << ", numPacketsServed=" << numPacketsServed_->toString()
       << ", numPacketsDropped=" << numPacketsDropped_->toString()
       << ")";
    return ss.str();
}

}