GCC=g++
LIBS=-lboost_filesystem -lboost_program_options -lboost_system -lopencv_core `pkg-config --libs opencv`
ARGS=--std=c++11
filesystem.o: filesystem.cpp filesystem.h
	${GCC} -c filesystem.cpp ${LIBS} ${ARGS} -ggdb

format_string.o: format_string.cpp format_string.h
	${GCC} -c format_string.cpp ${LIBS} ${ARGS} -ggdb

plant_data.o: plant_data.cpp plant_data.h

plant_data_wheat.o: plant_data_wheat.cpp plant_data_wheat.h plant_data.o

analysis_bb.o: analysis_bb.cpp analysis.h plant_data_wheat.h
	${GCC} -c analysis_bb.cpp ${LIBS} ${ARGS} -ggdb

analysis_wb.o: analysis_wb.cpp analysis.h plant_data_wheat.h 
	${GCC} -c analysis_wb.cpp ${LIBS} ${ARGS} -ggdb

test: test.cpp filesystem.o
	${GCC} -o test test.cpp filesystem.o ${LIBS} ${ARGS} -ggdb

PAT32_bb: PAT32.cpp filesystem.o analysis_bb.o format_string.o plant_data.o plant_data_wheat.o
	${GCC} -o PAT32_bb PAT32.cpp filesystem.o analysis_bb.o format_string.o plant_data.o plant_data_wheat.o ${LIBS} ${ARGS} -ggdb

PAT32_wb: PAT32.cpp filesystem.o analysis_wb.o format_string.o plant_data.o plant_data_wheat.o
	${GCC} -o PAT32_wb PAT32.cpp filesystem.o analysis_wb.o format_string.o plant_data.o plant_data_wheat.o ${LIBS} ${ARGS} -ggdb

clean:
	rm *.o PAT32_wb PAT32_bb
