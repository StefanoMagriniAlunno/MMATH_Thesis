# Questo makefile ha il compito di:
# 1. costruire i progetti cmake: crea la cartella build, ci entra e lancia cmake
# 2. compilare i progetti cmake: esegue make su ogni build
# 3. sposta la libreria ottenuta in lib

# Variabili: elenco dei progetti in projects
PROJECTS = $(shell ls -d lib/* | cut -d "/" -f 2)
PYTHON_VENV = ".venv"

all:
	@echo "Building projects: $(PROJECTS)"
	@for project in $(PROJECTS); do \
		rm -rf lib/$$project/build; \
		mkdir -p lib/$$project/build; \
		cd lib/$$project/build; \
		cmake ..; \
		cd ../..; \
	done
	@for project in $(PROJECTS); do \
		cd lib/$$project/build; \
		make; \
		cd ../..; \
	done
	@for project in $(PROJECTS); do \
		mv lib/$$project/build/lib$$project.so .cache/lib$$project.so; \
	done
