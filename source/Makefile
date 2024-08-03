# Questo makefile ha il compito di:
# 1. costruire i progetti cmake: crea la cartella build, ci entra e lancia cmake
# 2. compilare i progetti cmake: esegue make su ogni build
# 3. sposta la libreria ottenuta in lib

# Variabili: elenco dei progetti in projects
PROJECTS = $(shell ls -d lib/* | cut -d "/" -f 2)
PYTHON_VENV = ".venv"

all:
	@echo "Building projects: $(PROJECTS)"
	@echo "using Python: $(PY)"
	@echo "using GCC: $(CC)"
	@echo "using GXX: $(CXX)"
	@echo "using NVCC: $(CU)"
	@rm -rf .cache
	@mkdir -p .cache
	@for project in $(PROJECTS); do \
		rm -rf lib/$$project/build; \
		mkdir -p lib/$$project/build; \
		cd lib/$$project/build; \
		cmake .. --trace -DPY=$(PY) -DCC=$(CC) -DCXX=$(CXX) -DCU=$(CU); \
		if [ $$? -ne 0 ]; then \
			echo "CMake failed for project $$project"; \
			exit 1; \
		fi; \
		cd ../../..; \
		make -C lib/$$project/build --trace; \
		if [ $$? -ne 0 ]; then \
			echo "Make failed for project $$project"; \
			exit 1; \
		fi; \
		mv lib/$$project/build/lib$$project.so .cache/lib$$project.so; \
	done
