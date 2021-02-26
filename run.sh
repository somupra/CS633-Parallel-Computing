#!/usr/bin/env bash
if [ ! -f halo ]; then
    printf "halo does not exist, execute make first\n"
    exit 1
fi

if [ "$#" -ne 1 ]; then
    printf "USAGE: [NUM TIME STEPS]\n"
    exit 1
fi

/users/btech/somupra/UGP/allocator/src/allocator.out 64 8 > dump.txt

for execution in {1..5}
do
	for P in 16 36 49 64
    do
	    for N in $((16*16)) $((32*32)) $((64*64)) $((128*128)) $((256*256)) $((512*512)) $((1024*1024))
        do  
            # printf "$P $N $1\n"

            mpirun -np $P -f hostsimproved ./halo $N $1 >> data.txt
        done
    done
done
