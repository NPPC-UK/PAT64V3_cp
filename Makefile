GCC=g++
LIBS=-lboost_filesystem -lboost_program_options -lboost_system -lopencv_core `pkg-config --libs opencv`
ARGS=--std=c++11
filesystem.o: filesystem.cpp filesystem.h
	${GCC} -c filesystem.cpp ${LIBS} ${ARGS}

analysis.o: analysis.cpp analysis.h
	${GCC} -c analysis.cpp ${LIBS} ${ARGS}

test: test.cpp filesystem.o
	${GCC} -o test test.cpp filesystem.o ${LIBS} ${ARGS}

PAT32: PAT32.cpp filesystem.o analysis.o
	${GCC} -o PAT32 PAT32.cpp filesystem.o analysis.o ${LIBS} ${ARGS}
