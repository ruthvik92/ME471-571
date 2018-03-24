CC = mpicc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: casino_sample
 
casino_sample: casino_sample.c ${objs}
	${CC} -o casino_sample  ${INCLUDE_DIR} casino_sample.c ${objs} -lm
