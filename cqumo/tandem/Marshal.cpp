/**
 * @author Andrey Larionov
 */
#include "Marshal.h"

namespace cqumo {

std::string toYaml(
        const SimData& value,
        const std::string& indent,
        bool first) {
    std::stringstream ss;
    if (first)
        ss << "---" << std::endl;
    ss << indent << "numPacketsGenerated: "
        << toYaml(value.numPacketsGenerated) << std::endl
        << indent << "simTime: " << toYaml(value.simTime) << std::endl
        << indent << "realTimeMs: " << toYaml(value.realTimeMs) << std::endl
        << indent << "nodes:" << std::endl;
    for (auto &kv: value.nodeData) {
        ss << indent << "  " << "- address: " << toYaml(kv.first) << std::endl;
        ss << toYaml(kv.second, indent + "    ");
    }
    return ss.str();
}

std::string toYaml(
        const NodeData& value,
        const std::string& indent,
        bool first) {
    std::stringstream ss;
    if (first)
        ss << "---" << std::endl;
    auto nextIndent = indent + "  ";
    ss << indent << "systemSize: " << std::endl
            << toYaml(value.systemSize, nextIndent)
        << indent << "queueSize: " << std::endl
            << toYaml(value.queueSize, nextIndent)
        << indent << "serverSize: " << std::endl
            << toYaml(value.serverSize, nextIndent)
        << indent << "delays: " << std::endl
            << toYaml(value.delays, nextIndent)
        << indent << "departures: " << std::endl
            << toYaml(value.departures, nextIndent)
        << indent << "waitTime: " << std::endl
            << toYaml(value.waitTime, nextIndent)
        << indent << "responseTime: " << std::endl
            << toYaml(value.responseTime, nextIndent)
        << indent << "numPacketsGenerated: "
            << toYaml(value.numPacketsGenerated) << std::endl
        << indent << "numPacketsDelivered: "
            << toYaml(value.numPacketsDelivered) << std::endl
        << indent << "numPacketsLost: "
            << toYaml(value.numPacketsLost) << std::endl
        << indent << "numPacketsArrived: "
            << toYaml(value.numPacketsArrived) << std::endl
        << indent << "numPacketsServed: "
            << toYaml(value.numPacketsServed) << std::endl
        << indent << "numPacketsDropped: "
            << toYaml(value.numPacketsDropped) << std::endl
        << indent << "lossProb: " << toYaml(value.lossProb) << std::endl
        << indent << "dropProb: " << toYaml(value.dropProb) << std::endl
        << indent << "deliveryProb: " << toYaml(value.deliveryProb)
            << std::endl;
    return ss.str();
}

std::string toYaml(
        const VarData& value,
        const std::string& indent,
        bool first) {
    std::stringstream ss;
    if (first)
        ss << "---" << std::endl;
    ss << indent << "mean: " << value.mean << std::endl
        << indent << "std: " << value.std << std::endl
        << indent << "var: " << value.var << std::endl
        << indent << "count: " << value.count << std::endl
        << indent << "moments: " << toYaml(value.moments) << std::endl;
    return ss.str();
}

std::string toYaml(
        const SizeDist& value,
        const std::string& indent,
        bool first) {
    std::stringstream ss;
    if (first)
        ss << "---" << std::endl;
    ss << indent << "mean: " << value.mean() << std::endl
        << indent << "std: " << value.std() << std::endl
        << indent << "var: " << value.var() << std::endl
        << indent << "pmf: " << toYaml(value.pmf()) << std::endl;
    return ss.str();
}

std::string toYaml(const std::vector<double>& value) {
    return cqumo::toString(value);
}

}