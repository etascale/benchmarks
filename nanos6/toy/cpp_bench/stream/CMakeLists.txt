cmake_minimum_required(VERSION 2.8)

project(ompss_cluster_cpp_benchmarks)

# STREAM ARRAY SIZES
add_definitions(-DSTREAM_ARRAY_SIZE=1024000)
# ENABLE TEAM INITIALIZATION
add_definitions(-DTEAMINIT=1)

add_executable(stream-ompss-2 stream-ompss-2.cpp)
target_link_libraries(stream-ompss-2 numa)
target_link_libraries(stream-ompss-2 rt)

add_executable(stream-ompss-2-cluster-strong stream-ompss-2-cluster-strong.cpp)
target_link_libraries(stream-ompss-2-cluster-strong numa)
target_link_libraries(stream-ompss-2-cluster-strong rt)

add_executable(stream-ompss-2-cluster-weak stream-ompss-2-cluster-weak.cpp)
target_link_libraries(stream-ompss-2-cluster-weak numa)
target_link_libraries(stream-ompss-2-cluster-weak rt)

install(TARGETS stream-ompss-2 stream-ompss-2-cluster-strong stream-ompss-2-cluster-weak DESTINATION bin)
