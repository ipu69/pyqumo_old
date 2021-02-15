/**
 * Module with base definitions used in all other modules.
 * It defines:
 *
 * - Object: base class for all model objects
 * - toString() methods
 * - debug(): macros for debug output in printf-like manner
 *
 * If DEBUG is defined, debug() macro will write messages to
 * standard output. Otherwise, these macros will be empty,
 * so no performance penalty except arguments preparation
 * will take place.
 *
 * @author Andrey Larionov
 */
#ifndef CQUMO_BASE_H
#define CQUMO_BASE_H

#include <string>
#include <sstream>
#include <vector>

// #define DEBUG 1
#ifdef DEBUG
#define debug(...) printf(__VA_ARGS__)
#else
#define debug(...) /* nop */
#endif

namespace cqumo {

/**
 * Base class for most of the objects in the model.
 * Defines only one virtual method - toString().
 */
class Object {
  public:
    Object() = default;
    virtual ~Object() = default;

    /** Get one-line string representation for internal usage. */
    virtual std::string toString() const;
};


/**
 * Convert an array into a string with a separator between elements.
 * @tparam T elements type
 * @param array a vector
 * @param delim delimiter string (default: ", ")
 * @return string
 */
template<typename T>
std::string toString(
        const std::vector<T> &array,
        const std::string &delim = ", ") {
    std::stringstream ss;
    ss << "[";
    if (array.size() > 0) {
        ss << array[0];
        for (unsigned i = 1; i < array.size(); i++) {
            ss << delim << array[i];
        }
    }
    ss << "]";
    return ss.str();
}

}

#endif //CQUMO_TANDEM_BASE_H
