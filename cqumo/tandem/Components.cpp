/**
 * @author Andrey Larionov
 */
#include "Components.h"

#include <sstream>
#include <utility>
#include <iostream>

namespace cqumo {

// Class Packet
// --------------------------------------------------------------------------
Packet::Packet(int source, int target, double createdAt)
: source_(source), target_(target), createdAt_(createdAt) {
}

std::string Packet::toString() const {
    std::stringstream ss;
    ss << "(Packet: source=" << source_
        << ", target=" << target_
        << ", createdAt=" << createdAt_
        << ")";
    return ss.str();
}


// Class NodeComponent
// --------------------------------------------------------------------------
int NodeComponent::address() const {
    return owner_->address();
}


// Class Queue
// --------------------------------------------------------------------------
Queue::Queue(int capacity) : capacity_(capacity) {}

Queue::~Queue() {
    while (!packets_.empty()) {
        delete packets_.front();
        packets_.pop();
    }
}

int Queue::push(Packet *packet) {
    if (full())
        return 0;
    packets_.push(packet);
    return 1;
}

Packet *Queue::pop() {
    if (packets_.empty())
        return nullptr;
    auto value = packets_.front();
    packets_.pop();
    return value;
}

std::string Queue::toString() const {
    std::stringstream ss;
    ss << "(Queue: size=" << packets_.size() << ", capacity_="
       << (capacity_ < 0 ? "inf" : std::to_string(capacity_))
       << ")";
    return ss.str();
}


// Class Server
// --------------------------------------------------------------------------
Server::Server(const DblFn& intervals) : intervals_(intervals)
{} // NOLINT(modernize-pass-by-value)

Server::~Server() {
    delete packet_;
}

int Server::push(Packet *packet) {
    if (busy())
        return 0;
    this->packet_ = packet;
    return 1;
}

Packet *Server::pop() {
    if (ready())
        return nullptr;
    auto value = this->packet_;
    this->packet_ = nullptr;
    return value;
}

std::string Server::toString() const {
    std::stringstream ss;
    ss << "(Server: "
        << "packet_=" << (packet_ ? packet_->toString() : "NULL")
        << ")";
    return ss.str();
}


// Class Source
// --------------------------------------------------------------------------
Source::Source(const DblFn& intervals, int target)
: intervals_(intervals), target_(target) {} // NOLINT(modernize-pass-by-value)

Packet *Source::createPacket(double time) const {
    return new Packet(owner()->address(), target_, time);
}

std::string Source::toString() const {
    std::stringstream ss;
    ss << "(Source: target=" << target_ << ")";
    return ss.str();
}


// Class Node
// --------------------------------------------------------------------------
Node::Node(int address, Queue *queue, Server *server, Source *source)
: address_(address),
  queue_(queue),
  server_(server),
  source_(source),
  nextHop_(nullptr)
{
    queue->setOwner(this);
    server->setOwner(this);
    if (source) {
        source->setOwner(this);
    }
}

Node::~Node() {
    delete queue_;
    delete server_;
    if (source_) {
        delete source_;
    }
}

std::string Node::toString() const {
    std::stringstream ss;
    ss << "(Node: address=" << address_
       << ", server=" << server_->toString()
       << ", queue=" << queue_->toString()
       << ", nextNodeAddr="
       << (nextHop_ ? std::to_string(nextHop_->address()) : "NULL")
       << ")";
    return ss.str();
}


// Class Network
// --------------------------------------------------------------------------
Network::~Network() {
    for (auto &kv: nodes_) {
        delete kv.second;
    }
}

void Network::addNode(Node *node) {
    if (node == nullptr) {
        throw std::runtime_error("node = nullptr in Network::addNodeJournal()");
    }
    auto address = node->address();
    if (nodes_.count(address)) {
        throw std::runtime_error(
                std::string("node with address ") +
                std::to_string(address) +
                std::string(" already exists"));
    }
    nodes_[address] = node;
}

std::string Network::toString() const {
    std::stringstream ss;
    ss << "(Network: nodes=[";
    for (auto &kv: nodes_) {
        ss << "\n\t" << kv.first << ": " << kv.second->toString();
    }
    ss << "])";
    return ss.str();
}


// Helpers
// --------------------------------------------------------------------------
Network *buildOneHopeNetwork(
        const DblFn &arrival,
        const DblFn &service,
        int queueCapacity) {
    auto queue = new Queue(queueCapacity);
    auto server = new Server(service);
    auto source = new Source(arrival, 0);
    auto node = new Node(0, queue, server, source);
    auto network = new Network;
    network->addNode(node);
    return network;
}

Network *buildTandemNetwork(
        const DblFn& arrival,
        const std::vector<DblFn>& services,
        int queueCapacity) {
    auto network = new Network;
    int numNodes = static_cast<int>(services.size());
    for (int i = 0; i < static_cast<int>(services.size()); ++i){
        auto queue = new Queue(queueCapacity);
        auto server = new Server(services[i]);
        auto source = i == 0 ? new Source(arrival, numNodes - 1) : nullptr;
        auto node = new Node(i, queue, server, source);
        network->addNode(node);
    }
    for (unsigned i = 1; i < services.size(); ++i) {
        network->node(i - 1)->setNextHop(network->node(i));
    }
    // std::cout << network->toString() << std::endl;
    return network;
}

}