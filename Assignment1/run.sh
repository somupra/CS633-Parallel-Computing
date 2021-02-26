#!/usr/bin/env bash
make clean

make

if [ "$#" -ne 1 ]; then
    printf "USAGE: [NUM TIME STEPS]\n"
    exit 1
fi

/users/btech/somupra/UGP/allocator/src/allocator.out 64 8 > dump.txt

for execution in {1..5}
do  
    i=1
	for P in 16 36 49 64
    do
	    for N in $((16*16)) $((32*32)) $((64*64)) $((128*128)) $((256*256)) $((512*512)) $((1024*1024))
        do  
            mpirun -np $P -f hostsimproved ./halo $N $1 >> data$i.txt
        done
        ((i++))
    done
done

printf "data generated, now plotting..."
python plot.py data1.txt data2.txt data3.txt data4.txt