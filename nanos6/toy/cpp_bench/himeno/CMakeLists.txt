cmake_minimum_required(VERSION 2.8)

project(ompss_cluster_cpp_benchmarks)

OPTION(ENABLE_OUTPUT "ENABLE OUTPUT TO FILE FOR VER." OFF)

if (ENABLE_OUTPUT)
	add_definitions(-DENABLE_OUTPUT)
endif()

# PROBLEM SIZES
#	XS = 0
#	S  = 1
#	M  = 2
#	L  = 3
#	XL = 4
add_definitions(-DPROBLEM_SIZE=1)

add_executable(himeno-ompss-2 himeno-ompss-2.cpp)
target_link_libraries(himeno-ompss-2 numa)
target_link_libraries(himeno-ompss-2 rt)

add_executable(himeno-ompss-2-cluster-strong himeno-ompss-2-cluster-strong.cpp)
target_link_libraries(himeno-ompss-2-cluster-strong numa)
target_link_libraries(himeno-ompss-2-cluster-strong rt)

add_executable(himeno-ompss-2-cluster-weak himeno-ompss-2-cluster-weak.cpp)
target_link_libraries(himeno-ompss-2-cluster-weak numa)
target_link_libraries(himeno-ompss-2-cluster-weak rt)

install(TARGETS himeno-ompss-2 himeno-ompss-2-cluster-strong himeno-ompss-2-cluster-weak DESTINATION bin)
