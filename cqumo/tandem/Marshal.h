/**
 * Module with routines for marshalling and unmarshalling of model data
 * and results.
 *
 * Formats for marshalling results:
 *
 * - YAML
 *
 * Unmarshalling is not implemented yet.
 *
 * @author Andrey Larionov
 */

#ifndef CQUMO_TANDEM_MARSHAL_H
#define CQUMO_TANDEM_MARSHAL_H

#include "Simulation.h"

namespace cqumo {

/** Get YAML representation of SimData object. */
std::string toYaml(
        const SimData& value,
        const std::string& indent = "",
        bool first = true);

/** Get YAML representation of NodeData object. */
std::string toYaml(
        const NodeData& value,
        const std::string& indent = "",
        bool first = false);

/** Get YAML representation of VarData object. */
std::string toYaml(
        const VarData& value,
        const std::string& indent = "",
        bool first = false);

/** Get YAML representation of SizeDist object. */
std::string toYaml(
        const SizeDist& value,
        const std::string& indent = "",
        bool first = false);

/** Get YAML representation of scalar values of simple standard types. */
template<typename T>
std::string toYaml(T value) {
    return std::to_string(value);
}

/** Get YAML representation of std::vector<double> value. */
std::string toYaml(const std::vector<double>& value);

}

#endif //CQUMO_TANDEM_MARSHAL_H
