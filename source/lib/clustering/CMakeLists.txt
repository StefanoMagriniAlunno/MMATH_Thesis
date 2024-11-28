cmake_minimum_required(VERSION 3.22.1)
project(clustering LANGUAGES CXX CUDA)

set(MODE_FLAG -O3)

set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
set(Python3_EXECUTABLE ${PY})
find_package(Python3 REQUIRED COMPONENTS Development)
set(CMAKE_CUDA_COMPILER ${CU})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MODE_FLAG} -arch=sm_75")
set(CUDA_HOST_COMPILER ${CXX})
find_package(CUDAToolkit REQUIRED)

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
target_link_libraries(clustering PRIVATE Python3::Python CUDA::cublas CUDA::cudart OpenMP::OpenMP_CXX)

# Proprietà del target
set_target_properties(clustering
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "clustering"
    PREFIX "lib"
    CUDA_ARCHITECTURES "75"
    CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER}
)

# Opzioni di compilazione per il compilatore CUDA:
# -g e -arch=sm_75 in modo da poter debuggare il codice CUDA
# -O3 per ottimizzare il codice CUDA
target_compile_options(clustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${MODE_FLAG} -arch=sm_75>)
# Opzioni di compilazione per il compilatore C++
# -g in modo da poter debuggare il codice CUDA
# -O3 per ottimizzare il codice CUDA
target_compile_options(clustering PRIVATE $<$<COMPILE_LANGUAGE:CXX>: ${MODE_FLAG}>)


# Installazione
install(TARGETS clustering DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
