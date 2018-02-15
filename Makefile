GCC=g++
LIBS=-lboost_filesystem -lboost_system
ARGS=--std=c++11
filesystem.o: filesystem.cpp filesystem.h
	${GCC} -c filesystem.cpp ${LIBS} ${ARGS}

test: test.cpp filesystem.o
	${GCC} -o test test.cpp filesystem.o ${LIBS} ${ARGS}

all: test
