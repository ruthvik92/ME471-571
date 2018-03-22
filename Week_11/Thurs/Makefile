CC = mpicc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: io_binary_2d_demo

io_binary_2d_demo: io_binary_2d_demo.c ${objs}
	${CC} -o io_binary_2d_demo  ${INCLUDE_DIR} io_binary_2d_demo.c ${objs} -lm
