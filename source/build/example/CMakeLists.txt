cmake_minimum_required(VERSION 3.22.1)
project(my_project)

# Specifica la versione di Python desiderata
find_package(Python3 REQUIRED COMPONENTS Development)

# Aggiungi la sottocartella src per le librerie C
add_subdirectory(src)

# Installa la libreria condivisa nella stessa cartella di Python
install(TARGETS example LIBRARY DESTINATION ${Python3_SITEARCH})

# Opzionale: aggiungi un eseguibile per i test in C
if(BUILD_TESTING)
    add_executable(test_main main.c)
    target_link_libraries(test_main PRIVATE example)
endif()
