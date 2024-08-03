cmake_minimum_required(VERSION 3.22.1)
project(clustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Python3_EXECUTABLE ${PY})
find_package(Python3 REQUIRED COMPONENTS Development)
set(CMAKE_CUDA_COMPILER ${CU})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

# Includi le directory
include_directories(inc)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# File sorgente
set(SOURCE_FILES
  src/cudafcm.cu
  src/cxxfcm.cpp
  main.cpp
)

# Crea la libreria
add_library(clustering MODULE ${SOURCE_FILES})

# Aggiungi opzioni e link
target_link_libraries(clustering PRIVATE Python3::Python ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Proprietà del target
set_target_properties(clustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "clustering"
    PREFIX "lib"
    CUDA_ARCHITECTURES "75"
)

# Opzioni di compilazione
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-Werror;-Wall;-Wextra;-pedantic;-g")
set_source_files_properties(src/cxxfcm.cpp PROPERTIES COMPILE_OPTIONS "-Werror;-Wall;-Wextra;-pedantic;-g")
set_source_files_properties(src/cudafcm.cu PROPERTIES COMPILE_OPTIONS "-Xcompiler=-Werror,-Wall,-Wextra,-g")
target_compile_options(clustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>)

# Installazione
install(TARGETS clustering DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
