#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "mpi.h"

double stime, ftime, time;

int main(int argc, int *argv[]){
    MPI_Init(NULL,NULL);
    int my_rank, my_size;
    MPI_Comm_size(MPI_COMM_WORLD,&my_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);    
    
    int N = atoi(argv[1]);
    int data_points = atoi(argv[2]);

    int side_len = sqrt(data_points), root = sqrt(N);

    MPI_Status status[8];
    
    //initialize data
    int data[side_len][side_len];    
    for(int i=0;i<side_len;i++){
        for(int j=0;j<side_len;j++) data[i][j] = rand();
    }
    
    int recv_data[side_len+1][side_len+1];
    for(int i=0;i<side_len+1;i++){
        for(int j=0;j<side_len+1;j++){
                recv_data[i][j] = 0;
                if((i!=0)&&(i!=side_len)&&(j!=0)&&(j!=side_len)){
                    recv_data[i][j] = data[i][j];
                }
        }
    }

    int count=0;
    MPI_Request request[8];

    int buffer[side_len][8],pos=0;
    stime = MPI_Wtime();

    //Sending data
    if(my_rank/root != 0){ //Sending Upwards
        for(int j=0;j<side_len;j++){
            MPI_Pack (&data[0][j], 1, MPI_INT, buffer, side_len*32, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer, side_len*4,MPI_PACKED, my_rank-root,1,MPI_COMM_WORLD,&request[count++]);
    }
    
    if(my_rank/(N - root) ==0){ //Sending Downwards
        for(int j=0;j<side_len;j++){
            MPI_Pack (&data[side_len-1][j], 1, MPI_INT, buffer, side_len*32, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer+side_len, side_len*4,MPI_PACKED, my_rank+root,2,MPI_COMM_WORLD,&request[count++]);
    }
    
    if(my_rank%(root) != 0){ //Sending Leftwards
        for(int i=0;i<side_len;i++){
            MPI_Pack (&data[i][0], 1, MPI_INT, buffer, side_len*32, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer+side_len*2, side_len*4,MPI_PACKED, my_rank-1,3,MPI_COMM_WORLD,&request[count++]);
    }
    
    if(my_rank%root != (root-1)){ //Sending Rightwards
        for(int i=0;i<side_len;i++){
            MPI_Pack (&data[i][side_len-1], 1, MPI_INT, buffer, side_len*32, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer+side_len*3, side_len*4,MPI_PACKED, my_rank+1,4,MPI_COMM_WORLD,&request[count++]);
    }

    
    int buf1[side_len], buf2[side_len], buf3[side_len], buf4[side_len]; 
    //Recieving Data
    if(my_rank/root != 0){ //Recieving from top
        MPI_Irecv(buf1, side_len*4, MPI_PACKED, my_rank-root,2,MPI_COMM_WORLD,&request[count++]);
        pos = 0;
        MPI_Unpack(buf1, side_len*4, &pos, &recv_data[0][1], side_len, MPI_INT, MPI_COMM_WORLD);
    }
    
    if(my_rank/(N - root) ==0){ //Recieving from bottom
        MPI_Irecv(buf2, side_len*4, MPI_PACKED, my_rank+root,1,MPI_COMM_WORLD,&request[count++]);
        pos = 0;
        MPI_Unpack(buf2, side_len*4, &pos, &recv_data[1][0], side_len, MPI_INT, MPI_COMM_WORLD);
    }
    
    if(my_rank%(root) != 0){ //Recieving from left
        MPI_Irecv(buf3, side_len*4, MPI_PACKED, my_rank-1,4,MPI_COMM_WORLD,&request[count++]);
        pos = 0;
        MPI_Unpack(buf3, side_len*4, &pos, &recv_data[2][1], side_len, MPI_INT, MPI_COMM_WORLD);
    }
    
    if(my_rank%root != root-1){ //Recieving from right
        MPI_Irecv(buf4, side_len*4,MPI_PACKED, my_rank+1,3,MPI_COMM_WORLD,&request[count++]);
        pos = 0;
        MPI_Unpack(buf4, side_len*4, &pos, &recv_data[3][1], side_len, MPI_INT, MPI_COMM_WORLD);
    }

    MPI_Waitall(count, request, status);

    for(int i=0;i<side_len;i++){
        for(int j=0;j<side_len;j++){
            data[i][j] = (recv_data[i+1][j] + recv_data[i][j+1] + recv_data[i+2][j+1] + recv_data[i+1][j+2])/4; 
        }
    }

    ftime = MPI_Wtime();
    time = ftime - stime;

    printf("%lf\n",time);


    MPI_Finalize();
    return 0;
}