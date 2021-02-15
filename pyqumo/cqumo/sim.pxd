from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.functional cimport function


cdef extern from "Statistics.h" namespace "cqumo":
    cdef cppclass SizeDist:
        double mean()
        double var()
        double std()
        double moment(int order)
        vector[double] pmf()

    cdef cppclass VarData:
        double mean
        double std
        double var
        unsigned count
        vector[double] moments


cdef extern from "Functions.h" namespace "cqumo":
    ctypedef function[double()] DblFn
    cdef DblFn makeDblFn(double (*ctxFn)(void*), void* context)


cdef extern from "Simulation.h" namespace "cqumo":
    cdef cppclass NodeData:
        SizeDist systemSize
        SizeDist queueSize
        SizeDist serverSize
        VarData delays
        VarData departures
        VarData waitTime
        VarData responseTime
        unsigned numPacketsGenerated
        unsigned numPacketsDelivered
        unsigned numPacketsLost
        unsigned numPacketsArrived
        unsigned numPacketsServed
        unsigned numPacketsDropped
        double lossProb
        double dropProb
        double deliveryProb

    cdef cppclass SimData:
        map[int, NodeData] nodeData
        unsigned numPacketsGenerated
        double simTime
        double realTimeMs

    # noinspection PyPep8Naming
    SimData simMM1(
            double arrivalRate,
            double serviceRate,
            int queueCapacity,
            int maxPackets)

    SimData simGG1(
            DblFn arrival,
            DblFn service,
            int queueCapacity,
            int maxPackets)
    
    SimData simTandem(
            DblFn arrival,
            vector[DblFn]& services,
            int queueCapacity,
            int maxPackets)
