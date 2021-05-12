#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

#include "Simulation.h"
#include "Marshal.h"
#include "../Randoms.h"

using namespace std;
using namespace cqumo;

const char *FORMAT =
        "Format: cqumo_tandem <NUM_STATIONS> <ARRIVAL_RATE> <SERVICE_RATE> "
        "<QUEUE_CAPACITY> <IS_FIXED_SERVICE> [NUM_PACKETS]";


void printResults(const SimData& data);

int main(int argc, char **argv) {
    // Parse mandatory parameters
    if (argc < 6 || argc > 7) {
        cout << FORMAT << endl;
        return 1;
    }
    int numStations = stoi(argv[1]);
    double arrivalRate = stod(argv[2]);
    double serviceRate = stod(argv[3]);
    int queueCapacityInt = stoi(argv[4]);
    bool isFixedService = bool(stoi(argv[5]));

    if (queueCapacityInt < 0) {
        cout << "ERROR: queue capacity_ must be non-negative\n";
        return 1;
    }
    if (arrivalRate <= 0 || serviceRate <= 0) {
        cout << "ERROR: arrival and service rates must be positive\n";
        return 1;
    }
    ssize_t queueCapacity = static_cast<size_t>(queueCapacityInt);

    // Check whether number of packets were provided:
    size_t maxPackets = 10000;
    if (argc == 5) {
        int maxPackets_ = stoi(argv[4]);
        if (maxPackets_ <= 0) {
            cerr << "ERROR: number of packets must be positive\n";
            return 1;
        }
        maxPackets = static_cast<size_t>(maxPackets_);
    }

    auto factory = Randoms();
    auto arrivalVar = factory.createExponential(arrivalRate);
    auto serviceVar = factory.createExponential(serviceRate);

    DblFn arrivalFn = [&arrivalVar](){ return arrivalVar->eval(); };
    DblFn serviceFn = [&serviceVar](){ return serviceVar->eval(); };
    vector<DblFn> services(numStations, serviceFn);

    auto ret = simTandem(
            arrivalFn,
            services,
            queueCapacity,
            isFixedService,
            maxPackets);

    delete arrivalVar;
    delete serviceVar;

    // Print results to stdout:
    printResults(ret);
    return 0;
}

void printResults(const SimData& data) {
    cout << left << setw(20) << "Packets generated:" << data.numPacketsGenerated << endl;
    cout << left << setw(20) << "Time elasped:" << data.realTimeMs << endl;
    cout << left << setw(20) << "Simulation time:" << data.simTime << endl;
    cout << endl;
    auto headers = std::vector<string>{
        "Node", "Sys. size", "Queue size", "Busy", "Delay", "Depart.",
        "Wait t.", "Resp. t.", "Loss prob."
        };
    auto widths = std::vector<int>(headers.size(), 0);
    cout << "| ";
    for (int i = 0; i < (int)headers.size(); i++) {
        auto& header = headers[i];
        int width = max((int)header.size(), 5);
        widths[i] = width;
        cout << left << setw(width) << header << " | ";
    }
    cout << endl;

    auto writeLine = [&widths]() {
        cout << "+";
        for (int i = 0; i < (int)widths.size(); i++) {
            cout << string(widths[i]+2, '-') << "+";
        }
        cout << endl;
    };

    writeLine();

    int numNodes = data.nodeData.size();
    for (int i = 0; i < numNodes; i++) {
        auto& nd = data.nodeData.at(i);
        cout << "| " << left << setw(widths[0]) << (i+1) << " | "
            << setw(widths[1]) << setprecision(3) << nd.systemSize.mean() << " | "
            << setw(widths[2]) << setprecision(3) << nd.queueSize.mean() << " | "
            << setw(widths[3]) << setprecision(3) << nd.serverSize.mean() << " | "
            << setw(widths[4]) << setprecision(3) << nd.delays.mean << " | "
            << setw(widths[5]) << setprecision(3) << nd.departures.mean << " | "
            << setw(widths[6]) << setprecision(3) << nd.waitTime.mean << " | "
            << setw(widths[7]) << setprecision(3) << nd.responseTime.mean << " | "
            << setw(widths[8]) << setprecision(3) << nd.lossProb << " |" << endl;
    }

    writeLine();
}
