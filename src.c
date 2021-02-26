#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include"mpi.h"

int request_count=0;
int can_transfer(char c, int rank, int cluster_len, int np);
double* initialize_data(int n);
void compute_stencil(double* data, int side_len);
void compute_halo(double* data, double** recv, int side_len, int rank, int cluster_len, int p);

int main(int argc, char *argv[]){
    srand(time(NULL));

    if(argc != 3){
        printf("USAGE: halo [DATA POINTS PER PROCESS][NUM TIME STEPS]\n");
        return -1;
    }

    MPI_Init(NULL,NULL);
    int my_rank, p;
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    
    int data_points = atoi(argv[1]);
    int side_len = sqrt(data_points);
    int cluster_len = sqrt(p);
    int num_time_steps = atoi(argv[2]);

    /***************************************************************************************************************************/
    /*******************************************Transferring single double at a time********************************************/
    /***************************************************************************************************************************/
    double* data = initialize_data(side_len);

    double** recv_data = malloc(4*sizeof(double*));
    for(int i=0; i<4; i++){
        recv_data[i] = malloc(side_len*sizeof(double));
    }

    double stime = MPI_Wtime();
    for(int t=0; t<num_time_steps; t++){
        // Perform stencil computation
        compute_stencil(data, side_len);

        // If possible, then send the data (single double at a time)
        if(can_transfer('u', my_rank, cluster_len, p)){
            int target = my_rank - cluster_len;
            for(int i=0; i<side_len; i++){
                MPI_Send(&data[0*side_len + i], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD);
            }
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int target = my_rank + cluster_len;
            for(int i=0; i<side_len; i++){
                MPI_Send(&data[(side_len-1)*side_len + i], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD);
            }
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int target = my_rank - 1;
            for(int i=0; i<side_len; i++){
                MPI_Send(&data[i*side_len + 0], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD);
            }
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int target = my_rank + 1;
            for(int i=0; i<side_len; i++){
                MPI_Send(&data[i*side_len + side_len-1], 1, MPI_DOUBLE, target, i, MPI_COMM_WORLD);
            }
        }

        // If possible, then recieve the data in recv buffer (single double at a time)
        if(can_transfer('u', my_rank, cluster_len, p)){      
            int source = my_rank - cluster_len;
            MPI_Status st;
            for(int i=0; i<side_len; i++){
                MPI_Recv(&recv_data[0][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &st);
            }
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int source = my_rank + cluster_len;
            MPI_Status st;
            for(int i=0; i<side_len; i++){
                MPI_Recv(&recv_data[1][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &st);
            }
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int source = my_rank - 1;
            MPI_Status st;
            for(int i=0; i<side_len; i++){
                MPI_Recv(&recv_data[2][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &st);
            }
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int source = my_rank + 1;
            MPI_Status st;
            for(int i=0; i<side_len; i++){
                MPI_Recv(&recv_data[3][i], 1, MPI_DOUBLE, source, i, MPI_COMM_WORLD, &st);
            }
        }

        // Get the final averages for time t
        compute_halo(data, recv_data, side_len, my_rank, cluster_len, p);
    }
    
    double ftime = MPI_Wtime();
    double time = ftime - stime;

    double maxTime;
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!my_rank) printf ("%lf\n", maxTime);

    /***************************************************************************************************************************/
    /*****************************************USING MPI_Pack and MPI_Unpack for transfer****************************************/
    /***************************************************************************************************************************/
    free(data);

    // re-init data
    data = initialize_data(side_len);

    // buffers to unpack from 4 directions
    double r_ubuf[side_len];
    double r_dbuf[side_len];
    double r_lbuf[side_len];
    double r_rbuf[side_len];

    // buffer to store the pack to send
    double s_ubuf[side_len];
    double s_dbuf[side_len];
    double s_lbuf[side_len];
    double s_rbuf[side_len];

    stime = MPI_Wtime();
    for(int t=0; t<num_time_steps; t++){

        // Perform stencil computation
        compute_stencil(data, side_len);

        if(can_transfer('u', my_rank, cluster_len, p)){
            int target = my_rank - cluster_len, upos = 0;
            for(int j=0; j<side_len; j++){
                MPI_Pack (&data[0*side_len + j], 1, MPI_DOUBLE, s_ubuf, side_len*8, &upos, MPI_COMM_WORLD);
            }
            MPI_Send(s_ubuf, side_len*8, MPI_PACKED, target, 1, MPI_COMM_WORLD);
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int target = my_rank + cluster_len, dpos = 0;
            for(int j=0; j<side_len; j++){
                MPI_Pack (&data[(side_len-1)*side_len + j], 1, MPI_DOUBLE, s_dbuf, side_len*8, &dpos, MPI_COMM_WORLD);
            }
            MPI_Send(s_dbuf, side_len*8, MPI_PACKED, target, 2, MPI_COMM_WORLD);
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int target = my_rank - 1, lpos = 0;
            for(int i=0; i<side_len; i++){
                MPI_Pack (&data[i*side_len + 0], 1, MPI_DOUBLE, s_lbuf, side_len*8, &lpos, MPI_COMM_WORLD);
            }
            MPI_Send(s_lbuf, side_len*8, MPI_PACKED, target, 3, MPI_COMM_WORLD);
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int target = my_rank + 1, rpos = 0;
            for(int i=0;i<side_len;i++){
                MPI_Pack (&data[i*side_len + side_len-1], 1, MPI_DOUBLE, s_rbuf, side_len*8, &rpos, MPI_COMM_WORLD);
            }
            MPI_Send(s_rbuf, side_len*8, MPI_PACKED, target, 4, MPI_COMM_WORLD);
        }

        // Recieve and unpack from possible directions
        if(can_transfer('u', my_rank, cluster_len, p)){      
            int source = my_rank - cluster_len, curr_pos = 0;
            MPI_Status st;
            MPI_Recv(r_ubuf, side_len*8, MPI_PACKED, source, 2, MPI_COMM_WORLD, &st);
            MPI_Unpack(r_ubuf, side_len*8, &curr_pos, recv_data[0], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int source = my_rank + cluster_len, curr_pos = 0;
            MPI_Status st;
            MPI_Recv(r_dbuf, side_len*8, MPI_PACKED, source, 1, MPI_COMM_WORLD, &st);
            MPI_Unpack(r_dbuf, side_len*8, &curr_pos, recv_data[1], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int source = my_rank - 1, curr_pos = 0;
            MPI_Status st;
            MPI_Recv(r_lbuf, side_len*8, MPI_PACKED, source, 4, MPI_COMM_WORLD, &st);
            MPI_Unpack(r_lbuf, side_len*8, &curr_pos, recv_data[2], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int source = my_rank + 1, curr_pos = 0;
            MPI_Status st;
            MPI_Recv(r_rbuf, side_len*8, MPI_PACKED, source, 3, MPI_COMM_WORLD, &st);
            MPI_Unpack(r_rbuf, side_len*8, &curr_pos, recv_data[3], side_len, MPI_DOUBLE, MPI_COMM_WORLD);
        }  

        // Get the final averages for time t
        compute_halo(data, recv_data, side_len, my_rank, cluster_len, p);
    }
    ftime = MPI_Wtime();
    
    time = ftime - stime;
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!my_rank) printf ("%lf\n", maxTime);

    /***************************************************************************************************************************/
    /*****************************************USING VECTORS FOR ROW AND COLUMN TRANSFER*****************************************/
    /***************************************************************************************************************************/
    free(data);

    // re-init data
    data = initialize_data(side_len);

    MPI_Datatype row_vector, col_vector;
    MPI_Type_vector (side_len, 1, 1, MPI_DOUBLE, &row_vector);
    MPI_Type_commit (&row_vector);
    MPI_Type_vector (side_len, 1, side_len, MPI_DOUBLE, &col_vector);
    MPI_Type_commit (&col_vector);

    stime = MPI_Wtime();
    for(int t=0; t<num_time_steps; t++){

        // Perform stencil computation
        compute_stencil(data, side_len);

        // send the data in vectors in possible directions
        if(can_transfer('u', my_rank, cluster_len, p)){
            int target = my_rank - cluster_len;
            MPI_Send(&data[0], 1, row_vector, target, 1, MPI_COMM_WORLD);
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int target = my_rank + cluster_len;
            MPI_Send(&data[(side_len-1)*side_len], 1, row_vector, target, 2, MPI_COMM_WORLD);
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int target = my_rank - 1;
            MPI_Send(&data[0], 1, col_vector, target, 3, MPI_COMM_WORLD);
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int target = my_rank + 1;
            MPI_Send(&data[0*side_len + side_len-1], 1, col_vector, target, 4, MPI_COMM_WORLD);
        }

        // recieve from possible directions, for col recieve in MPI_DOUBLE datatype
        if(can_transfer('u', my_rank, cluster_len, p)){      
            int source = my_rank - cluster_len;
            MPI_Status st;
            MPI_Recv(recv_data[0], 1, row_vector, source, 2, MPI_COMM_WORLD, &st);
        }
        if(can_transfer('d', my_rank, cluster_len, p)){
            int source = my_rank + cluster_len; 
            MPI_Status st;
            MPI_Recv(recv_data[1], 1, row_vector, source, 1, MPI_COMM_WORLD, &st);
        }
        if(can_transfer('l', my_rank, cluster_len, p)){
            int source = my_rank - 1;
            MPI_Status st;
            MPI_Recv(recv_data[2], side_len, MPI_DOUBLE, source, 4, MPI_COMM_WORLD, &st);
        }
        if(can_transfer('r', my_rank, cluster_len, p)){
            int source = my_rank + 1;
            MPI_Status st;
            MPI_Recv(recv_data[3], side_len, MPI_DOUBLE, source, 3, MPI_COMM_WORLD, &st);
        }

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