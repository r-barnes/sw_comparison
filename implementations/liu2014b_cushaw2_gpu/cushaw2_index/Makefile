c_sources = bntseq.c bwt.c bwt_gen.c bwtindex.c QSufSort.c utils.c bwtmisc.c bwtio.c is.c main.c
objs_dir = objs
src_dir = src
objs += $(patsubst %.c, $(objs_dir)/%.c.o, $(c_sources))

CC = gcc
CCFLAGS = -O3 -funroll-loops -I $(src_dir) 
libs= -lz -lm 

exec=cushaw2_index
all: dir $(objs)
	$(CC) $(CCFLAGS) -o $(exec) $(objs) $(libs) $(CCLIBS)
	strip $(exec)

dir:
	mkdir -p $(objs_dir)

clean:
	rm -rf $(objs_dir)
	rm -f $(exec)

$(objs_dir)/%.c.o: $(src_dir)/%.c
	$(CC) $(CCFLAGS) -o $@ -c $<


