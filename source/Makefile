# Questo makefile ha il compito di:
# 1. costruire i progetti cmake: crea la cartella build, ci entra e lancia cmake
# 2. compilare i progetti cmake: esegue make su ogni build
# 3. sposta la libreria ottenuta in lib

# Variabili: elenco dei progetti in projects
PROJECTS = $(shell ls -d projects/* | cut -d "/" -f 2)
PYTHON_VENV = ".venv"

# Target: all
# Costruisce, compila tutti i progetti e infine sposta le librerie in lib
all: compile
	@echo "Moving libraries..."
	@for project in $(PROJECTS); do \
		echo "move projects/$$project/build/$$project.so to ../$(PYTHON_VENV)/lib/python3.10/site-packages/"; \
		mv projects/$$project/build/$$project.so lib; \
	done

# Target: compile
# Compila i progetti
compile: build
	@echo "Compiling projects..."
	@for project in $(PROJECTS); do \
		cd projects/$$project/build; \
		make; \
		cd ../..; \
	done

# Target: build
# Costruisce i progetti (rimuove build se presente)
build: clean
	@echo "Building projects: $(PROJECTS)"
	@for project in $(PROJECTS); do \
		rm -rf projects/$$project/build; \
		mkdir -p projects/$$project/build; \
		cd projects/$$project/build; \
		cmake ..; \
		cd ../..; \
	done

# Target: clean
# Pulisce i progetti (elimina le cartelle di build e il contenuto di lib)
clean:
	@echo "Cleaning projects..."
	@rm -rf lib/*
