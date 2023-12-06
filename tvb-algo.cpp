#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "time.h"
#include <chrono>
#include "math.h"
#include <random>

using namespace std;

typedef std::chrono::high_resolution_clock Time;

/* Defining simulation parameters */
#define N           76                                          // Number of nodes
#define M           2                                           // Number of state variables per node
#define dt          0.05                                        // Timestep    
#define tf          150.0                                       // Final time of the simulation
#define speed       4.0                                         // Speed of conductance
#define freq        1.0                                         // frequency
#define k           0.001                                       // Constant used in post function
#define lam         0.1                                         // LAM in colored noise
#define E           exp(-lam * dt)                              // Constant E in colored noise
#define PRE(i, j)   (j - 1.0)                                   // pre-synapse function
#define POST(gx)    (k * gx)                                    // post-synapse function
#define FDX(x, y)   freq * (x - (pow(x, 3)/3.0) + y) * 3.0      // Local dynamic of first state variable
#define FDY(x, c)   freq * (1.01 - x + c) / 3.0                 // Local dynamic of second state variable
#define G(i, x)     sqrt(1e-9)                                  // Additive Linear Noise
/* END */

#define USE_SPARSE  true                                        // Whether to use sparse calculation for coupling

float Xs[(int) (tf/dt)][N][M] = {0.0};  // The values of state variables of nodes throughout the simulation
float e[N][M];                          // Noise value for each state variable

void calculate_coupling(int i, int n, float W[N], int D[N], float& coupling); // naive coupling calculation
void calculate_coupling_sparse(int i, int n, int w_size, int nzr_size, float* w, int* d, int* r, int* col, int* nzr, int* lri, float& coupling); // coupling calculation using sparse characteristic
void step(int i, int n, float coupling);

int main(){

    /* Extract the Weight and Distance data from files */
    auto tick = Time::now();

    ifstream file_w("./data/tvb76_w.txt");
    ifstream file_d("./data/tvb76_d.txt");

    float W[N][N]; // Weight matrix
    int D[N][N]; // Delay matrix in timestep -> Extracted from (distance matrix / speed) / dt
    int nzr_W = 0; // Number of non-zero elements in W
    int nzr_c = 0; // Number of rows with non-zero element

    for(int i = 0; i < N; i++){
        string line_w, line_d;
        getline(file_w, line_w);
        getline(file_d, line_d);

        string temp_w, temp_d;
        stringstream stream_w(line_w);
        stringstream stream_d(line_d);

        bool nzr_c_flag = false;
        for(int j = 0; j < N; j++){
            getline(stream_w, temp_w, ' ');
            getline(stream_d, temp_d, ' ');

            W[i][j] = atof(temp_w.c_str());
            D[i][j] = (int) ((atof(temp_d.c_str()) / speed)/dt);

            if(W[i][j] != 0.0){
                nzr_W++;
                nzr_c_flag = true;
            }
        }

        if(nzr_c_flag) nzr_c++;
    }

    float* w = (float *) malloc(nzr_W * sizeof(float)); // non-zero weights
    int* d = (int *) malloc(nzr_W * sizeof(int)); // delays associated with non-zero weights
    int* r = (int *) malloc(nzr_W * sizeof(int)); // rows of the non-zero weights
    int* c = (int *) malloc(nzr_W * sizeof(int)); // columns of the non-zero weights
    int* nzr = (int *) malloc(N * sizeof(int)); // for each node n, lri[nzr[n]] is the beginning of corresponding weights of n in w array
    int* lri = (int *) malloc(nzr_c * sizeof(int)); // local reduction indices -> indexes of r in which the value is different from the value of previous index

    int index = 0;
    int index_nzr = 0;
    int i_deduct = 0;
    for(int i = 0; i < N; i++){
        bool nzr_c_flag = false; // if the row has non-zero element
        for(int j = 0; j < N; j++){
            if(W[i][j] != 0.0){
                w[index] = W[i][j];
                d[index] = D[i][j];
                r[index] = i;
                c[index] = j;
                index++;
                nzr_c_flag = true;
            }
        }
        if(nzr_c_flag){
            nzr[index_nzr] = i - i_deduct;
            index_nzr++;
        } else{
            nzr[index_nzr] = -1;
            index_nzr++;
            i_deduct++;
        }
    }

    int temp = -1;
    index_nzr = 0;
    for(int i = 0; i < nzr_W; i++){
        if(r[i] != temp){
            lri[index_nzr] = i;
            index_nzr++;
            temp = r[i];
        }
    }

    std::cout << "Pre-processing: " << chrono::duration<double, std::milli>(Time::now() - tick).count() << " ms" << std::endl;
    /* END: Extract the Weight and Distance data from files */

    /* Simulation */
    tick = Time::now();
    srand((unsigned) time(NULL));
    default_random_engine gen;
    random_device rd;
    gen.seed(rd());
    normal_distribution randn{0.0, 1.0};
    
    for(int i = 0; i < (int)(tf/dt); i++){
        if(i == 0){
            for(int n = 0; n < N; n++){
                Xs[i][n][0] = -1.0;
                Xs[i][n][1] = -1.0;
                e[n][0] = sqrt(G(i, Xs[i][n][0]) * lam) * randn(gen);
                e[n][1] = sqrt(G(i, Xs[i][n][1]) * lam) * randn(gen);
            }
        }
        else if(i == 1){
            for(int n = 0; n < N; n++){
                Xs[i][n][0] = ((((float) rand()/RAND_MAX))/5.0) + 1.0;
                Xs[i][n][1] = ((((float) rand()/RAND_MAX))/5.0) - 0.6;
                float h[M];
                h[0] = sqrt(G(i, Xs[i][n][0]) * lam * (1.0 - pow(E, 2))) * randn(gen);
                h[1] = sqrt(G(i, Xs[i][n][1]) * lam * (1.0 - pow(E, 2))) * randn(gen);
                e[n][0] = (e[n][0] * E) + h[0];
                e[n][1] = (e[n][1] * E) + h[1];
            }
        }
        else{
            for(int n = 0; n < N; n++){
                float coupling;
                if(USE_SPARSE)  calculate_coupling_sparse(i, n, nzr_W, nzr_c, w, d, r, c, nzr, lri, coupling);
                else            calculate_coupling(i, n, W[n], D[n], coupling);
                step(i, n, coupling);
                float h[M];
                h[0] = sqrt(G(i, Xs[i][n][0]) * lam * (1.0 - pow(E, 2))) * randn(gen);
                h[1] = sqrt(G(i, Xs[i][n][1]) * lam * (1.0 - pow(E, 2))) * randn(gen);
                e[n][0] = (e[n][0] * E) + h[0];
                e[n][1] = (e[n][1] * E) + h[1];
            }
        }
    }

    std::cout << "Simulation: " << chrono::duration<double, std::milli>(Time::now() - tick).count() << " ms" << std::endl;
    /* END: Simulation */

    ofstream file_xs("xs.txt");
    for(int n = 0; n < N; n++){
        for(int i = 0; i < (int) (tf/dt); i++){
            file_xs << Xs[i][n][0] << endl;
        }
        file_xs << "-" << endl;
    }

    file_w.close();
    file_d.close();
    file_xs.close();
}

void calculate_coupling(int i, int n, float W[N], int D[N], float& coupling){
    float c = 0.0;
    for(int j = 0; j < N; j++){
        if(W[j] != 0){
            if(i < D[j]){
                c += W[j] * PRE(Xs[i - 1][n][0], 0.0);
            } else{
                if(n == j)  c += W[j] * PRE(Xs[i - 1][n][0], Xs[i - 1][j][0]);
                else        c += W[j] * PRE(Xs[i - 1][n][0], Xs[i - D[j]][j][0]);
            }
        }
    }
    coupling = POST(c);
}

void calculate_coupling_sparse(int i, int n, int w_size, int nzr_size, float* w, int* d, int* r, int* col, int* nzr, int* lri, float& coupling){
    int lri_index = nzr[n];

    if(lri_index == -1){
        coupling = 0;
        return;
    }
    
    int index_start = lri[lri_index];
    int index_stop;
    if(lri_index == (nzr_size - 1)) index_stop = w_size;
    else                            index_stop = lri[lri_index + 1];


    float c = 0.0;
    for(int j = index_start; j < index_stop; j++){
        if(i < d[j]){
            c += w[j] * PRE(Xs[i - 1][n][0], 0.0);
        } else{
            if(d[j] == 0)   c += w[j] * PRE(Xs[i - 1][n][0], Xs[i - 1][col[j]][0]);
            else            c += w[j] * PRE(Xs[i - 1][n][0], Xs[i - d[j]][col[j]][0]);
        }
    }

    coupling = POST(c);
}

void step(int i, int n, float coupling){
    float x = Xs[i-1][n][0];
    float y = Xs[i-1][n][1];
    float dx = dt * (FDX(x, y)); //dt * (FDX(x, y) + e[n][0]);
    float dy = dt * (FDY(x, coupling) + e[n][1]);
    Xs[i][n][0] = x + dx;
    Xs[i][n][1] = y + dy;
}