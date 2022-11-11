cmake_minimum_required(VERSION 2.8)

project(ompss_cluster_c_benchmarks)

add_executable(matvec-strong matvec-strong.c)
target_link_libraries(matvec-strong common)
target_link_libraries(matvec-strong numa)
target_link_libraries(matvec-strong rt)

add_executable(matvec-weak matvec-weak.c)
target_link_libraries(matvec-weak common)
target_link_libraries(matvec-weak numa)
target_link_libraries(matvec-weak rt)

configure_file(json/matvec_weak_strong_scalability.json.in
	json/matvec_weak_strong_scalability.json)
configure_file(json/matvec_weak_single_node.json.in
	json/matvec_weak_single_node.json)

install(TARGETS matvec-strong matvec-weak DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/json/
	DESTINATION share/json)
