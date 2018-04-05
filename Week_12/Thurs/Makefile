CC = gcc
NVC = nvcc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

.PHONY: hello_demo

hello_demo: hello_demo.c ${objs}
	${NVC} -o hello_demo  ${INCLUDE_DIR} hello_demo.cu ${objs} -lm