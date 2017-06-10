APP_NAME=code-t12-cloud
CLANG_FLAGS=-fopenmp -omptargets=x86_64-unknown-linux-spark

all: clean
	clang $(CLANG_FLAGS) $(APP_NAME).c -o $(APP_NAME)

clean:
	rm -f $(APP_NAME) *.scala *.sbt
	rm -f -r project  spark-warehouse  target
