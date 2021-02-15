cmake_minimum_required(VERSION 3.16)
project(cqumo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # not necessary, but encouraged

add_executable(cqumo_tandem tandem/main.cpp Base.h Base.cpp tandem/Components.h tandem/Components.cpp tandem/Statistics.h tandem/Statistics.cpp tandem/System.h tandem/System.cpp tandem/Journals.h tandem/Journals.cpp tandem/Simulation.h tandem/Simulation.cpp Functions.h Functions.cpp tandem/Marshal.h tandem/Marshal.cpp Randoms.h Randoms.cpp)
