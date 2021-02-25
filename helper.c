#include <stdlib.h>

int can_transfer(char c, int rank, int domain_side_len, int np){
    if(c == 'u'){
        if(rank < (int)(np/domain_side_len)) return 0;
        else return 1;
    }
    else if(c == 'd'){
        if(rank >= (np-domain_side_len)) return 0;
        else return 1;
    }
    else if(c == 'l'){
        if((rank+1) % domain_side_len == 1) return 0;
        else return 1;
    }
    else if(c == 'r'){
        if((rank+1) % domain_side_len == 0) return 0;
        else return 1;
    }
}

double** initialize_data(int n){
    double** ptr = (double**) malloc(n * sizeof(double*));
    for(int i=0; i<n; i++){
        ptr[i] = (double*) malloc(n * sizeof(double));
    }
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            ptr[i][j] = (double)rand()/RAND_MAX;
        }
    }
    return ptr;
}


void compute_stencil(double** data, int side_len){
    for(int i=1; i<side_len-1; i++){
        for(int j=1; j<side_len-1; j++){
            data[i][j] = (data[i-1][j] + data[i+1][j] + data[i][j-1] + data[i][j+1])/4;
        }
    }
    return;
}

void compute_halo(double** data, double** recv, int side_len, int rank, int domain_side_len, int p){
    // for (0, 0)
    int n=2;
    double sum = data[0][1] + data[1][0]; 
    if(can_transfer('u', rank, domain_side_len, p)) {
        sum += recv[0][0];
        n++;
    }
    if(can_transfer('l', rank, domain_side_len, p)) {
        sum += recv[2][0];
        n++;
    }
    data[0][0] = sum/n;

    // for (0, side_len-1)
    n=2;
    sum = data[0][side_len-2] + data[1][side_len-1];
    if(can_transfer('u', rank, domain_side_len, p)) {
        sum += recv[0][side_len-1];
        n++;
    }
    if(can_transfer('r', rank, domain_side_len, p)) {
        sum += recv[3][side_len-1];
        n++;
    }
    data[0][side_len-1] = sum/n;

    // for (side_len-1, 0)
    n=2;
    sum = data[side_len-2][0]+ data[side_len-1][1];
    if(can_transfer('d', rank, domain_side_len, p)) {
        sum += recv[3][0];
        n++;
    }
    if(can_transfer('l', rank, domain_side_len, p)) {
        sum += recv[2][side_len-1];
        n++;
    }
    data[side_len-1][0] = sum/n;

    // for (0, side_len-1)
    n=2;
    sum = data[side_len-1][side_len-2] + data[side_len-2][side_len-1];
    if(can_transfer('d', rank, domain_side_len, p)) {
        sum += recv[1][side_len-1];
        n++;
    }
    if(can_transfer('r', rank, domain_side_len, p)) {
        sum += recv[3][side_len-1];
        n++;
    }
    data[side_len-1][side_len-1] = sum/n;

    for(int i=1; i<side_len-1; i++){
        if(can_transfer('u', rank, domain_side_len, p)){
            data[0][i] = (data[0][i-1] + data[0][i+1] + data[1][i] + recv[0][i])/4;
        }else{
            data[0][i] = (data[0][i-1] + data[0][i+1] + data[1][i])/3;
        }

        if(can_transfer('d', rank, domain_side_len, p)){
            data[side_len-1][i] = (data[side_len-1][i-1] + data[side_len-1][i+1] + data[side_len-2][i] + recv[1][i])/4;
        }else{
            data[side_len-1][i] = (data[side_len-1][i-1] + data[side_len-1][i+1] + data[side_len-2][i])/3;
        }

        if(can_transfer('l', rank, domain_side_len, p)){
            data[i][0] = (data[i+1][0] + data[i-1][0] + data[i][1] + recv[2][i])/4;
        }else{
            data[i][0] = (data[i+1][0] + data[i-1][0] + data[i][1])/3;
        }

        if(can_transfer('r', rank, domain_side_len, p)){
            data[i][side_len-1] = (data[i+1][side_len-1] + data[i-1][side_len-1] + data[i][side_len-2] + recv[3][i])/4;
        }else{
            data[i][side_len-1] = (data[i+1][side_len-1] + data[i-1][side_len-1] + data[i][side_len-2])/3;
        }
    }
    return;
}
