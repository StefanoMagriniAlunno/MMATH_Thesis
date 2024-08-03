cmake_minimum_required(VERSION 3.22.1)
project(synthesis LANGUAGES C)

set(CMAKE_C_COMPILER ${CC})
find_package(OpenMP REQUIRED)
set(Python3_EXECUTABLE ${PY})
find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(inc)

# elenco dei file sorgente in src
set(SOURCE_FILES
  src/csynthesis.c
  main.c
)

add_library(synthesis MODULE ${SOURCE_FILES})

target_link_libraries(synthesis PRIVATE Python3::Python OpenMP::OpenMP_C)

set_target_properties(synthesis PROPERTIES OUTPUT_NAME "synthesis")
set_target_properties(synthesis PROPERTIES PREFIX "lib")

set_source_files_properties(src/synthesis.c PROPERTIES COMPILE_OPTIONS "-Werror;-Wall;-Wextra;-pedantic;-O3")
set_source_files_properties(main.c PROPERTIES COMPILE_OPTIONS "-Werror;-Wall;-Wextra;-pedantic;-O3")

install(TARGETS synthesis DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
