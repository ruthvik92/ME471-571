CC = mpicc

SRC = ../../src
INCLUDE_DIR = ../../src

objs = ${SRC}/demo_util.c

.PHONY: demo5 demo6
 
demo5: demo5.c ${objs}
	${CC} -o mpi_demo5  -I${INCLUDE_DIR} demo5.c ${objs}

demo6: demo6.c ${objs}
	${CC} -o mpi_demo6  -I$(INCLUDE_DIR) demo6.c ${objs}

demo6_bug: demo6.c ${objs}
	${CC} -o mpi_demo6_bug  -I$(INCLUDE_DIR) demo6_bug.c ${objs}

