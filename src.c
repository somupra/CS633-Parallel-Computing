#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include"mpi.h"

int request_count=0;
int can_transfer(char c, int rank, int cluster_len, int np);
double** initialize_data(int n);
void compute_stencil(double** data, int side_len);
void compute_halo(double** data, double** recv, int side_len, int rank, int cluster_len, int p);

int main(int argc, char *argv[]){
    srand(time(NULL));

    if(argc != 2){
        printf("USAGE: halo [DATA POINTS PER PROCESS]\n");
        return -1;
    }

    MPI_Init(NULL,NULL);
    int my_rank, p;
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    
    int data_points = atoi(argv[1]);
    int side_len = sqrt(data_points);
    int cluster_len = sqrt(p);
    int num_time_steps = 50;

    
    double** data = initialize_data(side_len);

    MPI_Status status[8*side_len];
    MPI_Request request[8*side_len];

    double** recv_data = malloc(4*sizeof(double*));
    for(int i=0; i<4; i++){
        recv_data[i] = malloc(side_len*sizeof(double));
    }

    double stime = MPI_Wtime();
    for(int t=0; t<50; t++){
        // set request_count to zero, reuse the same request objects
        request_count = 0;

        // Perform stencil computation
        compute_stencil(data, side_len);

        // If possible, then send the data (single double at a time)
        if(can_transfer('u', my_rank, cluster_len, p)){
            int target = my_rank - cluster_len;
            for(int i=0; i<side_len; i++){
                MPI_Isend(&data[0][i], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int target = my_rank + cluster_len;
            for(int i=0; i<side_len; i++){
                MPI_Isend(&data[side_len-1][i], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int target = my_rank - 1;
            for(int i=0; i<side_len; i++){
                MPI_Isend(&data[i][0], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int target = my_rank + 1;
            for(int i=0; i<side_len; i++){
                MPI_Isend(&data[i][side_len-1], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }

        // If possible, then recieve the data in recv buffer (single double at a time)
        if(can_transfer('u', my_rank, cluster_len, p)){      
            int source = my_rank - cluster_len;
            for(int i=0; i<side_len; i++){
                MPI_Irecv(&recv_data[0][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int source = my_rank + cluster_len;
            for(int i=0; i<side_len; i++){
                MPI_Irecv(&recv_data[1][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int source = my_rank - 1;
            for(int i=0; i<side_len; i++){
                MPI_Irecv(&recv_data[2][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int source = my_rank + 1;
            for(int i=0; i<side_len; i++){
                MPI_Irecv(&recv_data[3][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &request[request_count++]);
            }
        }

        // wait for all send to complete
        MPI_Waitall(request_count, request, status);

        // Get the final averages for time t
        compute_halo(data, recv_data, side_len, my_rank, cluster_len, p);
    }
    
    double ftime = MPI_Wtime();
    double time = ftime - stime;
    printf("%lf\n",time);

    MPI_Finalize();
    return 0;
}
