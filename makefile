CC=mpicc
halo: src.o helper.o
	$(CC) -o halo src.o helper.o -lm

.PHONY: clean
clean:
	rm -f *.o *.x *.out halo
