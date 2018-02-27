CC = mpicc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: hmwk3_demo
 
hmwk3_demo : hmwk3_demo.c ${objs}
	${CC} -o hmwk3_demo  ${INCLUDE_DIR} hmwk3_demo.c ${objs} -lm

