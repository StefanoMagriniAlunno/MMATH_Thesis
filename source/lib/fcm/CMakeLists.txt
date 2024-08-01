cmake_minimum_required(VERSION 3.22.1)
project(fcm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trova e configura Python
find_package(Python3 REQUIRED COMPONENTS Development)

# Trova e configura CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
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
add_library(fcm MODULE ${SOURCE_FILES})

# Aggiungi opzioni e link
target_link_libraries(fcm PRIVATE Python3::Python ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Proprietà del target
set_target_properties(fcm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "fcm"
    PREFIX "lib"
    CUDA_ARCHITECTURES "75"
)

# Opzioni di compilazione
target_compile_options(fcm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>
    -Wall
    -Wextra
    -pedantic
    -g
)

# Installazione
install(TARGETS fcm DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
