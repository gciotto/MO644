APP_NAME=code-t12-cloud
CLANG_FLAGS=-fopenmp -omptargets=x86_64-unknown-linux-spark
#CLANG_FLAGS=-fopenmp -omp-target=linux-unknown-spark

all:
	/opt/ompcloud/bin/clang $(CLANG_FLAGS) $(APP_NAME).c -o $(APP_NAME) -DN=$(N)

clean:
	rm -f $(APP_NAME) *.scala *.sbt
	rm -f -r project  spark-warehouse  target
