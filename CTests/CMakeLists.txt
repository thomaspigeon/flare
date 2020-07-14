add_executable(test
	Structure/test_structure.cpp)

include_directories(../Sources/)
target_link_libraries(test PUBLIC gtest gtest_main flare)
