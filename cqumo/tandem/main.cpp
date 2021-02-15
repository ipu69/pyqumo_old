#include <iostream>
#include <string>
#include <chrono>

#include "Simulation.h"
#include "Marshal.h"


const char *FORMAT =
        "Format: gg1.x <ARRIVAL_RATE> <SERVICE_RATE> "
        "<QUEUE_CAPACITY> [NUM_PACKETS]";


int main(int argc, char **argv) {
    // Parse mandatory parameters
    if (argc < 4 || argc > 5) {
        std::cout << FORMAT << std::endl;
        return 1;
    }
    double arrivalRate = std::stod(argv[1]);
    double serviceRate = std::stod(argv[2]);
    int queueCapacityInt = std::stoi(argv[3]);

    if (queueCapacityInt < 0) {
        std::cout << "ERROR: queue capacity_ must be non-negative\n";
        return 1;
    }
    if (arrivalRate <= 0 || serviceRate <= 0) {
        std::cout << "ERROR: arrival and service rates must be positive\n";
        return 1;
    }
    ssize_t queueCapacity = static_cast<size_t>(queueCapacityInt);

    // Check whether number of packets were provided:
    size_t maxPackets = 10000;
    if (argc == 5) {
        int maxPackets_ = std::stoi(argv[4]);
        if (maxPackets_ <= 0) {
            std::cerr << "ERROR: number of packets must be positive\n";
            return 1;
        }
        maxPackets = static_cast<size_t>(maxPackets_);
    }

    auto ret = cqumo::simMM1(
            arrivalRate,
            serviceRate,
            queueCapacity,
            maxPackets);

    // Print results to stdout:
    std::cout << toYaml(ret) << std::endl;
    return 0;
}
