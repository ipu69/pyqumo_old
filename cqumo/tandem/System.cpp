/**
 * @author Andrey Larionov
 */
#include "System.h"
#include <sstream>

namespace cqumo {

// Event and EventType
// ---------------------------------------------------------------------------

std::string Event::toString() const {
    std::stringstream ss;
    ss << "(Event: time=" << time
       << ", id=" << id
       << ", address=" << address
       << ", type=" << cqumo::toString(type)
       << ")";
    return ss.str();
}


std::string toString(EventType value) {
    switch (value) {
        case STOP:
            return "STOP";
        case SOURCE_TIMEOUT:
            return "SOURCE_TIMEOUT";
        case SERVER_TIMEOUT:
            return "SERVER_TIMEOUT";
        default:
            return "???";
    }
}

// Class System
// ---------------------------------------------------------------------------
System::System() {
    EventCmp cmp = [](Event *left, Event *right) {
        return (left->time > right->time) ||
               (left->time == right->time && left->id > right->id);
    };
    eventsQueue_ = new EventQueue(cmp);
}

System::~System() {
    while (!eventsQueue_->empty()) {
        delete eventsQueue_->top();
        eventsQueue_->pop();
    }
    delete eventsQueue_;
}

void System::schedule(EventType type, double interval, int address) {
    auto event = new Event{
        .id = nextId_++,
        .time = time_ + interval,
        .address = address,
        .type = type
    };
    debug("\t- <SYSTEM> scheduled %s\n", event->toString().c_str());
    eventsQueue_->push(event);
}

Event *System::nextEvent() {
    if (eventsQueue_->empty()) {
        return new Event{
            .id = nextId_++,
            .time = time_,
            .address = -1,
            .type = STOP
        };
    }
    auto event = eventsQueue_->top();
    eventsQueue_->pop();
    debug("\t- <SYSTEM> extracted event %s\n", event->toString().c_str());
    time_ = event->time;
    return event;
}


// Helpers for event handlers
// ---------------------------------------------------------------------------
void startService(
        Server *server,
        Packet *packet,
        System *system,
        NetworkJournal *journal) {
    auto time = system->time();
    auto address = server->owner()->address();

    // Start service and schedule end:
    server->push(packet);
    double interval = server->interval();
    system->schedule(SERVER_TIMEOUT, interval, address);
    debug("\t- scheduled service end at %.3f (interval = %.3f)\n",
          time + interval, interval);

    // Update statistics:
    auto nodeRecords = journal->nodeJournal(address);
    nodeRecords->waitTimes()->record(time - packet->arrivedAt());
    nodeRecords->serverSize()->record(time, server->size());
}


void handleArrival(
        Packet *packet,
        Node *node,
        System *system,
        NetworkJournal *journal) {
    auto server = node->server();
    auto queue = node->queue();
    auto time = system->time();
    auto address = node->address();
    auto records = journal->nodeJournal(address);

    // Update number of arrived packets and arrival time:
    records->numPacketsArrived()->inc();
    packet->setArrivedAt(time);

    if (server->ready()) {
        // Server was empty: start service and record server size statistics:
        debug("\t- server was empty, start service\n");
        startService(server, packet, system, journal);
        records->systemSize()->record(time, node->size());
    } else if (queue->push(packet)) {
        // Packet was pushed: record queue and system size statistics:
        debug("\t- server was busy and queue wasn't full, pushing packet_\n");
        records->queueSize()->record(time, queue->size());
        records->systemSize()->record(time, node->size());
    } else {
        // Packet was dropped: increment number of lost and dropped packets,
        // and delete the packet_ itself:
        debug("\t- server was busy and queue was full, dropping packet_\n");
        records->numPacketsDropped()->inc();
        journal->nodeJournal(packet->source())->numPacketsLost()->inc();
        delete packet;
    }
}


// Event handlers
// ---------------------------------------------------------------------------
void handleSourceTimeout(Node *node, System *system, NetworkJournal *journal) {
    debug("[%.3f] packet_ arrived at %d\n", system->time(),
          node->address());
    auto source = node->source();
    auto address = node->address();
    auto records = journal->nodeJournal(address);

    // Update statistics:
    records->numPacketsGenerated()->inc();
    journal->numPacketsGenerated()->inc();

    // Schedule next event, create the packet_ and start serving it:
    double interval = source->interval();
    system->schedule(SOURCE_TIMEOUT, interval, address);
    auto packet = source->createPacket(system->time());

    debug("\t- scheduled next arrival at %.3f (interval = %.3f)\n",
          system->time() + interval, interval);

    // Call helper to process the packet
    handleArrival(packet, node, system, journal);

    debug("\t- server size: %zd, queue size: %zd\n", node->server()->size(),
          node->queue()->size());
}


void handleServerTimeout(Node *node, System *system, NetworkJournal *journal) {
    auto address = node->address();
    auto server = node->server();
    auto queue = node->queue();
    auto time = system->time();
    auto records = journal->nodeJournal(address);

    auto packet = server->pop();

    debug("[%.3f] server %d finished serving %s\n", time, address,
          packet->toString().c_str());

    // Update number of served packets and response time:
    records->numPacketsServed()->inc();
    records->responseTimes()->record(time - packet->arrivedAt());
    records->departures()->record(time - server->lastDepartureAt());
    server->setLastDepartureAt(time);

    // Decide, what to do with the packet_:
    // - if its target is this node, deliver
    // - otherwise, forward to the next node
    if (packet->target() == node->address()) {
        // Packet was delivered: record statistics and delete packet_
        debug("\t- packet_ was delivered\n");
        auto sourceRecords = journal->nodeJournal(packet->source());
        sourceRecords->numPacketsDelivered()->inc();
        sourceRecords->delays()->record(time - packet->createdAt());
        delete packet;
    } else {
        // Packet should be forwarded to the next hop:
        debug("\t- forwarding packet_ to %d\n",
              node->nextHop()->address());
        handleArrival(packet, node->nextHop(), system, journal);
    }

    // Check whether next packet_ can be served:
    if (!queue->empty()) {
        // Queue has packets: extract one, start serving and record queue size:
        debug("\t- getting next packet_ from the queue\n");
        packet = queue->pop();
        startService(server, packet, system, journal);
        records->queueSize()->record(time, queue->size());
    } else {
        // Queue is empty - record the server became ready:
        debug("\t- server is ready, queue is empty\n");
        records->serverSize()->record(time, 0);
    }

    // Update system size:
    debug("\t- server size: %zd, queue size: %zd\n",
          server->size(), queue->size());
    records->systemSize()->record(time, node->size());
}


// Main simulation (DES) loop
// ---------------------------------------------------------------------------
void runMainLoop(
        Network *network,
        System *system,
        NetworkJournal *journal,
        int maxPackets) {
    // Initialize the model:
    debug("==== INIT ====\nnetwork: %s\n", network->toString().c_str());
    journal->reset(system->time());
    for (auto &addrNodePair: network->nodes()) {
        auto node = addrNodePair.second;
        auto source = node->source();
        if (source) {
            auto address = addrNodePair.first;
            system->schedule(SOURCE_TIMEOUT, source->interval(), address);
        }
    }

    debug("==== RUN ====\n");
    while (!system->stopped()) {
        // Check whether enough packets were generated:
        if (maxPackets >= 0 &&
                journal->numPacketsGenerated()->value() >= maxPackets) {
            system->stop();
            continue;
        }

        // If stop conditions not satisfied, extract the next event.
        auto event = system->nextEvent();
        if (event->type == STOP) {
            system->stop();
        } else if (event->type == SOURCE_TIMEOUT) {
            handleSourceTimeout(network->node(event->address), system,
                                journal);
        } else if (event->type == SERVER_TIMEOUT) {
            handleServerTimeout(network->node(event->address), system,
                                journal);
        }
        delete event;
    }
}

}
