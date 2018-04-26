CC = gcc
NVC = nvcc

me471 = ${HOME}/ME471-571

SRC = ${me471}/src
INCLUDE_DIR = -I${me471}/src -I/usr/include

objs = ${SRC}/demo_util.c

CFLAGS=-O2

.PHONY: device_info sm host blockdim warpdiv

device_info: device_info.cu ${objs}
	${NVC} -o device_info  ${INCLUDE_DIR} device_info.cu ${objs} -lm

sm: sm.cu ${objs}
	${NVC} -o sm  ${INCLUDE_DIR} sm.cu ${objs} -lm

blockdim: blockdim.cu ${objs}
	${NVC} -o blockdim  ${CFLAGS} ${INCLUDE_DIR} blockdim.cu ${objs} -lm

host: host.cu ${objs}
	${NVC} -o host  ${CFLAGS} ${INCLUDE_DIR} host.cu ${objs} -lm

warpdiv: warpdiv.cu ${objs}
	${NVC} -o warpdiv  ${INCLUDE_DIR} warpdiv.cu ${objs} -lm
