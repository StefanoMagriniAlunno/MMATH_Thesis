cmake_minimum_required(VERSION 3.22.1)
project(example)

# Trova il pacchetto Python3
find_package(Python3 REQUIRED COMPONENTS Development)

# Aggiungi la cartella di inclusione per gli header personalizzati
include_directories(inc)

# Crea la libreria condivisa (modulo Python) example.so
add_library(example MODULE main.c src/sum.c)

# Linka la libreria Python3::Python alla libreria example.so
target_link_libraries(example PRIVATE Python3::Python)

# Imposta il nome del file di output del modulo
set_target_properties(example PROPERTIES OUTPUT_NAME "main")

# Rimuovi il prefisso "lib" dal nome del modulo (solo necessario su alcune piattaforme)
set_target_properties(example PROPERTIES PREFIX "")

# Installa il modulo Python nella cartella corrente
install(TARGETS example DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
