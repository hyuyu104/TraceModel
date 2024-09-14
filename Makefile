compileCpp:
	cd traceHMM/cpp; cmake .; make 

clearCpp:
	rm -r traceHMM/cpp/CMakeFiles
	rm traceHMM/cpp/cmake_install.cmake
	rm traceHMM/cpp/CMakeCache.txt
	rm traceHMM/cpp/Makefile
	rm traceHMM/cpp/update.cpython-312-darwin.so
	rm -r traceHMM/__pycache__

# keepCppCompiledOnly:
# 	mv traceHMM/cpp/update.cpython-312-darwin.so traceHMM/
# 	rm -r traceHMM/cpp
# 	mkdir traceHMM/cpp
# 	mv traceHMM/update.cpython-312-darwin.so traceHMM/cpp/

gitAdd:
	git add -A traceHMM
	git add -A docs
	git add Makefile
	git add README.md
	git add pyproject.toml
	git add .readthedocs.yaml
	git add LICENSE
	git add notebooks
	git add scripts
	git add .gitattributes

devel:
	python -m pip install -e .

initSphinx:
	sphinx-quickstart docs
	sphinx-build -M html docs/source/ docs/build/

	# new theme
	python -m pip install furo
	# parsing numpy docstrings
	python -m pip install sphinxcontrib-napoleon
	# install autoapi
	python -m pip install sphinx-autoapi

html: #devel
	# sphinx-build -M html docs/source/ docs/build/
	sh scripts/jupyterrst.sh # convert jupyter files
	cd docs; sphinx-apidoc -o source ../traceHMM; \
		make clean; make html