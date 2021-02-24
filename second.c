#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

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

    int buffer[side_len],pos=0;
    stime = MPI_Wtime();

    //Sending data
    if(my_rank/root != 0){ //Sending Upwards
        for(int j=0;j<side_len;j++){
            MPI_Pack (&data[0][j], 1, MPI_INT, buffer, side_len*4, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer, pos,MPI_PACKED, my_rank-root,1,MPI_COMM_WORLD,&request[count++]);
    }pos=0;
    
    if(my_rank/(N - root) ==0){ //Sending Downwards
        for(int j=0;j<side_len;j++){
            MPI_Pack (&data[side_len-1][j], 1, MPI_INT, buffer, side_len*4, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer, pos,MPI_PACKED, my_rank+root,2,MPI_COMM_WORLD,&request[count++]);
    }pos=0;
    
    if(my_rank%(root) != 0){ //Sending Leftwards
        for(int i=0;i<side_len;i++){
            MPI_Pack (&data[i][0], 1, MPI_INT, buffer, side_len*4, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer, pos,MPI_PACKED, my_rank-1,3,MPI_COMM_WORLD,&request[count++]);
    }pos=0;
    
    if(my_rank%root != (root-1)){ //Sending Rightwards
        for(int i=0;i<side_len;i++){
            MPI_Pack (&data[i][side_len-1], 1, MPI_INT, buffer, side_len*4, &pos, MPI_COMM_WORLD);
        }
        MPI_Isend(buffer, pos,MPI_PACKED, my_rank+1,4,MPI_COMM_WORLD,&request[count++]);
    }pos=0;

    

    //Recieving Data
    if(my_rank/root != 0){ //Recieving from top
        MPI_Irecv(buffer, side_len, MPI_PACKED, my_rank-root,2,MPI_COMM_WORLD,&request[count++]);
        for(int j=1;j<side_len+1;j++) recv_data[0][j] = buffer[j-1];
    }
    
    if(my_rank/(N - root) ==0){ //Recieving from bottom
        MPI_Irecv(buffer, side_len, MPI_PACKED, my_rank+root,1,MPI_COMM_WORLD,&request[count++]);
        for(int j=1;j<side_len+1;j++) recv_data[side_len][j] = buffer[j-1];
    }
    
    if(my_rank%(root) != 0){ //Recieving from left
        MPI_Irecv(buffer, side_len, MPI_PACKED, my_rank-1,4,MPI_COMM_WORLD,&request[count++]);
        for(int i=1;i<side_len+1;i++) recv_data[i][0] = buffer[i-1];
    }
    
    if(my_rank%root != root-1){ //Recieving from right
        MPI_Irecv(buffer, side_len,MPI_PACKED, my_rank+1,3,MPI_COMM_WORLD,&request[count++]);
        for(int i=1;i<side_len+1;i++) recv_data[i][side_len] = buffer[i-1];
    }


    ftime = MPI_Wtime();
    time = ftime - stime;

    MPI_Waitall(count, request, status);
    for(int i=0;i<side_len;i++){
        for(int j=0;j<side_len;j++){
            data[i][j] = (recv_data[i+1][j] + recv_data[i][j+1] + recv_data[i+2][j+1] + recv_data[i+1][j+2])/4; 
        }
    }


    MPI_Finalize();
    return 0;
}
