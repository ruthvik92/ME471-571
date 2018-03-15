CC = mpicc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: datatype_demo io_demo
 
datatype_demo: datatype_demo.c ${objs}
	${CC} -o datatype_demo  ${INCLUDE_DIR} datatype_demo.c ${objs} -lm
	
io_demo: io_demo.c ${objs}
	${CC} -o io_demo  ${INCLUDE_DIR} io_demo.c ${objs} -lm
