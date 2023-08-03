add_library(device_mem_utils STATIC device_mem_utils.cc)

add_library(data_generator STATIC data_generator.cc)
if(USE_HIPRAND)
target_link_libraries(data_generator hiprand)
endif()

