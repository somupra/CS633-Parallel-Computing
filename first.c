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


    int side_len = sqrt(data_points);
    int root = sqrt(N);

    MPI_Status status[8*side_len];
    
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

    stime = MPI_Wtime();


    int count=0;
    MPI_Request request[8*side_len];
    //Sending data
    for(int i=0;i<side_len;i++){
        for(int j=0;j<side_len;j++){
            if(i==0 && my_rank/(root) != 0){ //Sending Upwards
                MPI_Isend(&data[i][j], 1, MPI_INT, my_rank-root, 1, MPI_COMM_WORLD, &request[count++]);
            }
            if(i==side_len-1 && my_rank/(N - root) ==0){ //Sending Downwards
                MPI_Isend(&data[i][j], 1, MPI_INT, my_rank+root, 2, MPI_COMM_WORLD, &request[count++]);
            }
            if(j==0 && my_rank%(root) != 0){ //Sending Leftwards
                MPI_Isend(&data[i][j], 1, MPI_INT, my_rank-1, 3, MPI_COMM_WORLD, &request[count++]);
            }            
            if(j==side_len-1 && (my_rank%(root) != (root - 1))){ //Sending Rightwards
                MPI_Isend(&data[i][j], 1, MPI_INT, my_rank+1, 4, MPI_COMM_WORLD, &request[count++]);
            }
        }
    }

    //Recieving Data
    for(int i=0;i<side_len;i++){
        for(int j=0;j<side_len;j++){
            if(i==0 && my_rank/(root) != 0){   //Recieving from top
                MPI_Irecv(&recv_data[0][j+1], 1, MPI_INT, my_rank-root, 2, MPI_COMM_WORLD, &request[count++]);
            }
            if(i==side_len-1 && my_rank/(N - root) ==0){ //Recieving from down
                MPI_Irecv(&recv_data[side_len][j+1], 1, MPI_INT, my_rank+root, 1, MPI_COMM_WORLD, &request[count++]);
            }
            if(j==0 && my_rank%(root) !=0){ //Recieving from left 
                MPI_Irecv(&recv_data[i+1][0], 1, MPI_INT, my_rank-1, 4, MPI_COMM_WORLD, &request[count++]);
            }
            if(j==side_len-1 && (my_rank%(root) != (root-1))){ //Recieving from right
                MPI_Irecv(&recv_data[i+1][side_len], 1, MPI_INT, my_rank+1, 3, MPI_COMM_WORLD, &request[count++]);
            }
        }
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
