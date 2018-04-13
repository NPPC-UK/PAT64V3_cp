GCC=g++
LIBS=-lboost_filesystem -lboost_program_options -lboost_system -lopencv_core `pkg-config --libs opencv`
ARGS=--std=c++11
filesystem.o: filesystem.cpp filesystem.h
	${GCC} -c filesystem.cpp ${LIBS} ${ARGS} -ggdb

format_string.o: format_string.cpp format_string.h
	${GCC} -c format_string.cpp ${LIBS} ${ARGS} -ggdb

analysis_common.o: analysis_common.cpp analysis.h

analysis_bb.o: analysis_bb.cpp analysis_common.o analysis.h
	${GCC} -c analysis_bb.cpp analysis_common.o ${LIBS} ${ARGS} -ggdb

analysis_wb.o: analysis_wb.cpp analysis_common.o analysis.h 
	${GCC} -c analysis_wb.cpp analysis_common.o ${LIBS} ${ARGS} -ggdb

test: test.cpp filesystem.o
	${GCC} -o test test.cpp filesystem.o ${LIBS} ${ARGS} -ggdb

PAT32_bb: PAT32.cpp filesystem.o analysis_bb.o format_string.o analysis_common.o
	${GCC} -o PAT32_bb PAT32.cpp filesystem.o analysis_bb.o format_string.o analysis_common.o ${LIBS} ${ARGS} -ggdb

PAT32_wb: PAT32.cpp filesystem.o analysis_wb.o format_string.o analysis_common.o
	${GCC} -o PAT32_wb PAT32.cpp filesystem.o analysis_wb.o format_string.o analysis_common.o ${LIBS} ${ARGS} -ggdb

clean:
	rm *.o PAT32_wb PAT32_bb
