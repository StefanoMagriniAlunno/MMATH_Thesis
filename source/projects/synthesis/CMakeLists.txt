cmake_minimum_required(VERSION 3.22.1)
project(synthesis)

find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(inc)

# elenco dei file sorgente in src
set(SOURCE_FILES
  src/csynthesis.c
  main.c
)

add_library(synthesis MODULE ${SOURCE_FILES})

target_link_libraries(synthesis PRIVATE Python3::Python)

set_target_properties(synthesis PROPERTIES OUTPUT_NAME "synthesis")
set_target_properties(synthesis PROPERTIES PREFIX "")

install(TARGETS synthesis DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
