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
        "<QUEUE_CAPACITY> <IS_FIXED_SERVICE> <HAS_CT> [NUM_PACKETS]";


void printResults(const SimData& data);

int main(int argc, char **argv) {
    // Parse mandatory parameters
    if (argc < 7 || argc > 8) {
        cout << FORMAT << endl;
        return 1;
    }
    int numStations = stoi(argv[1]);
    double arrivalRate = stod(argv[2]);
    double serviceRate = stod(argv[3]);
    int queueCapacityInt = stoi(argv[4]);
    bool isFixedService = bool(stoi(argv[5]));
    bool hasCrossTraffic = bool(stoi(argv[6]));

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
    cout << "argc = " << argc << endl;
    if (argc == 8) {
        int maxPackets_ = stoi(argv[7]);
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
    map<int,DblFn> arrivals;
    arrivals[0] = arrivalFn;
    if (hasCrossTraffic) {
        for (int i = 1; i < numStations; i++) {
            arrivals[i] = arrivalFn;
        }
    }

    auto ret = simTandem(
            arrivals,
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
    for (int i = 0; i < (int)headers.size(); i++) {
        widths[i] = max((int)headers[i].size(), 5);
    }

    auto writeLine = [&widths]() {
        cout << "+";
        for (int i = 0; i < (int)widths.size(); i++) {
            cout << string(widths[i]+2, '-') << "+";
        }
        cout << endl;
    };

    writeLine();

    cout << "| ";
    for (int i = 0; i < (int)headers.size(); i++) {
        cout << left << setw(widths[i]) << headers[i] << " | ";
    }
    cout << endl;

    writeLine();

    int numNodes = data.nodeData.size();
    for (int i = 0; i < numNodes; i++) {
        auto& nd = data.nodeData.at(i);
        cout << "| " << left << setw(widths[0]) << (i+1) << " | "
            << setw(widths[1]) << setprecision(2) << nd.systemSize.mean() << " | "
            << setw(widths[2]) << setprecision(2) << nd.queueSize.mean() << " | "
            << setw(widths[3]) << setprecision(2) << nd.serverSize.mean() << " | "
            << setw(widths[4]) << setprecision(2) << nd.delays.mean << " | "
            << setw(widths[5]) << setprecision(2) << nd.departures.mean << " | "
            << setw(widths[6]) << setprecision(2) << nd.waitTime.mean << " | "
            << setw(widths[7]) << setprecision(2) << nd.responseTime.mean << " | "
            << setw(widths[8]) << setprecision(2) << nd.lossProb << " |" << endl;
    }

    writeLine();
}
