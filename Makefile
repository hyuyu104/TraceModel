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
	git add Makefile
	git add README.md
	git add -A docs
	git add pyproject.toml
	git add .readthedocs.yaml
	git add LICENSE

devel:
	python -m pip install -e .

html: devel
	cd docs; sphinx-apidoc -o . ../; \
		make clean; make html