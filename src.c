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

    double maxTime;
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!my_rank) printf ("%lf\n", maxTime);

    /*****************************************************************************************************************/
    /* Using MPI_Pack */
    int n = side_len;
    while(n--) free(data[n]);
    free(data);

    data = initialize_data(side_len);
    
    double buffer[8][side_len];

    stime = MPI_Wtime();
    for(int t=0; t<1; t++){
        // set request_count to zero, reuse the same request objects
        request_count = 0;

        // Perform stencil computation
        compute_stencil(data, side_len);

        // If possible, then send the data in pack, init pos here
        int pos = 0;

        if(can_transfer('u', my_rank, cluster_len, p)){
            int target = my_rank - cluster_len;
            // if(my_rank == 2){
            //     printf("sending to %d\n", target);
            //     for(int i=0; i<side_len; i++) printf("%lf ", data[0][i]);
            //     printf("\n");
            // }
            for(int j=0; j<side_len; j++){
                MPI_Pack (&data[0][j], 1, MPI_DOUBLE, buffer, side_len*8*8, &pos, MPI_COMM_WORLD);
            }
            MPI_Isend(buffer[0], side_len*8, MPI_PACKED, target, 1, MPI_COMM_WORLD, &request[request_count++]);
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int target = my_rank + cluster_len;
            for(int j=0; j<side_len; j++){
                MPI_Pack (&data[side_len-1][j], 1, MPI_DOUBLE, buffer, side_len*8*8, &pos, MPI_COMM_WORLD);
            }
            MPI_Isend(buffer[1], side_len*8, MPI_PACKED, target, 2, MPI_COMM_WORLD, &request[request_count++]);
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int target = my_rank - 1;
            for(int i=0; i<side_len; i++){
                MPI_Pack (&data[i][0], 1, MPI_DOUBLE, buffer, side_len*8*8, &pos, MPI_COMM_WORLD);
            }
            MPI_Isend(buffer[2], side_len*8, MPI_PACKED, target, 3, MPI_COMM_WORLD, &request[request_count++]);
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int target = my_rank + 1;
            for(int i=0;i<side_len;i++){
                MPI_Pack (&data[i][side_len-1], 1, MPI_DOUBLE, buffer, side_len*8*8, &pos, MPI_COMM_WORLD);
            }
            MPI_Isend(buffer[3], side_len*8, MPI_PACKED, target, 4, MPI_COMM_WORLD, &request[request_count++]);
        }

        // If possible, then recieve the data in recv buffer unpacked
        if(can_transfer('u', my_rank, cluster_len, p)){      
            int source = my_rank - cluster_len;
            MPI_Irecv(buffer[4], side_len*8, MPI_PACKED, source, 2, MPI_COMM_WORLD, &request[request_count++]);
            
            pos = 4*side_len;
            MPI_Unpack(buffer, side_len*8*8, &pos, recv_data[0], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int source = my_rank + cluster_len;
            MPI_Irecv(buffer[5], side_len*8, MPI_PACKED, source, 1, MPI_COMM_WORLD, &request[request_count++]);
            
            pos = 5*side_len;
            MPI_Unpack(buffer, side_len*8*8, &pos, recv_data[1], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int source = my_rank - 1;
            MPI_Irecv(buffer[6], side_len*8, MPI_PACKED, source, 4, MPI_COMM_WORLD, &request[request_count++]);
            
            pos = 6*side_len;
            MPI_Unpack(buffer, side_len*8*8, &pos, recv_data[2], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int source = my_rank + 1;
            MPI_Irecv(buffer[7], side_len*8, MPI_PACKED, source, 3, MPI_COMM_WORLD, &request[request_count++]);
            
            pos = 7*side_len;
            MPI_Unpack(buffer, side_len*8*8, &pos, recv_data[3], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        // wait for all send to complete
        MPI_Waitall(request_count, request, status);

        // if(can_transfer('d', my_rank, cluster_len, p) && my_rank == 0){
        //     printf("recv on rank %d\n", my_rank);
        //     for(int i=0; i<side_len; i++) printf("%lf ", recv_data[1][i]);
        //     printf("\n");
        // }
        // Get the final averages for time t
        compute_halo(data, recv_data, side_len, my_rank, cluster_len, p);
    }
    ftime = MPI_Wtime();
    
    time = ftime - stime;
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!my_rank) printf ("%lf\n", maxTime);

    MPI_Finalize();
    return 0;
}
