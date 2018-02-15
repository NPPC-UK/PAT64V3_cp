GCC=g++
LIBS=-lboost_filesystem -lboost_system -lopencv_core `pkg-config --libs opencv`
ARGS=--std=c++11
filesystem.o: filesystem.cpp filesystem.h
	${GCC} -c filesystem.cpp ${LIBS} ${ARGS}

test: test.cpp filesystem.o
	${GCC} -o test test.cpp filesystem.o ${LIBS} ${ARGS}

PAT32: PAT32.cpp filesystem.o
	${GCC} -o PAT32 PAT32.cpp filesystem.o ${LIBS} ${ARGS}
