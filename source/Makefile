# Questo makefile ha il compito di:
# 1. costruire i progetti cmake: crea la cartella build, ci entra e lancia cmake
# 2. compilare i progetti cmake: esegue make su ogni build
# 3. sposta la libreria ottenuta in lib

# Variabili: elenco dei progetti in projects
PROJECTS = $(shell ls -d projects/* | cut -d "/" -f 2)
PYTHON_VENV = ".venv"

all:
	@mkdir -p projects
	@echo "Building projects: $(PROJECTS)"
	@rm -rf lib/*
	@mkdir -p lib
	@for project in $(PROJECTS); do \
		rm -rf projects/$$project/build; \
		mkdir -p projects/$$project/build; \
		cd projects/$$project/build; \
		cmake ..; \
		cd ../..; \
	done
	@for project in $(PROJECTS); do \
		cd projects/$$project/build; \
		make; \
		cd ../..; \
	done
	@for project in $(PROJECTS); do \
		mv projects/$$project/build/$$project.so lib/$$project.so; \
	done
