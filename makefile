CC=mpicc
halo: src.o
	$(CC) -o halo src.o -lm

.PHONY: clean
clean:
	rm -f *.o *.x *.out halo
