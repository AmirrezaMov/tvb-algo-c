#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include "time.h"
#include <chrono>
#include "math.h"
#include <random>
#include "fpm/fixed.hpp"
#include "fpm/math.hpp"
#include "fpm/ios.hpp"

using namespace std;

typedef std::chrono::high_resolution_clock Time;

/* Defining simulation parameters */
#define N           76                                          // Number of nodes
#define M           2                                           // Number of state variables per node
#define dt          0.05f                                       // Timestep    
#define tf          150.0f                                      // Final time of the simulation
#define speed       2.0f                                        // Speed of conductance
#define freq        1.0f                                        // frequency
#define k           static_cast<f24>(0.001)                                      // Constant used in post function
#define lam         0.1f                                        // LAM in colored noise
#define E           exp(-lam * dt)                              // Constant E in colored noise
#define PRE(i, j)   (j - static_cast<f24>(1.0))                                  // pre-synapse function
#define POST(gx)    (k * gx)                                    // post-synapse function
#define FDX(x, y)   freq * (x - (pow(x, 3)/3.0f) + y) * 3.0f    // Local dynamic of first state variable
#define FDY(x, c)   freq * (1.01f - x + c) / 3.0f               // Local dynamic of second state variable
#define G(i, x)     sqrt(1e-9f)                                 // Additive Linear Noise
/* END */

#define USE_SPARSE  true                                        // Whether to use sparse calculation for coupling
#define BENCHMARK   false                                        // Whether to do benchmarking

typedef fpm::fixed_8_24 f24;

f24 Xs[(int) (tf/dt)][N][M];  // The values of state variables of nodes throughout the simulation
f24 e[N][M];                          // Noise value for each state variable

/* Benchmark variables */
float coupling_time = 0.0f;
int   coupling_time_n = 0;
float step_time = 0.0f;
int   step_time_n = 0; 
/* END */

void calculate_coupling(int i, int n, f24 W[N], int D[N], f24& coupling); // naive coupling calculation
void calculate_coupling_sparse(int i, int n, int w_size, int nzr_size, f24* w, int* d, int* r, int* col, int* nzr, int* lri, f24& coupling); // coupling calculation using sparse characteristic
void step(int i, int n, f24 coupling);

int main(){

    /* Extract the Weight and Distance data from files */
    auto tick = Time::now();

    ifstream file_w("./data/tvb76_w.txt");
    ifstream file_d("./data/tvb76_d.txt");

    f24 W[N][N]; // Weight matrix
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

            W[i][j]  = static_cast<f24>(atof(temp_w.c_str()));
            D[i][j] = (int) ((atof(temp_d.c_str()) / speed)/dt);

            if(W[i][j] != static_cast<f24>(0.0)){
                nzr_W++;
                nzr_c_flag = true;
            }
        }

        if(nzr_c_flag) nzr_c++;
    }

    f24* w = (f24 *) malloc(nzr_W * sizeof(f24)); // non-zero weights
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
            if(W[i][j] != static_cast<f24>(0.0)){
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

    cout << "Pre-processing: " << chrono::duration<double, std::milli>(Time::now() - tick).count() << " ms" << endl;
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
                Xs[i][n][0] = static_cast<f24>(-1.0);
                Xs[i][n][1] = static_cast<f24>(-1.0);
                e[n][0] = fpm::sqrt(static_cast<f24>(G(i, Xs[i][n][0])) * static_cast<f24>(lam)) * static_cast<f24>(randn(gen));
                e[n][1] = fpm::sqrt(static_cast<f24>(G(i, Xs[i][n][1])) * static_cast<f24>(lam)) * static_cast<f24>(randn(gen));
            }
        }
        else if(i == 1){
            for(int n = 0; n < N; n++){
                Xs[i][n][0] = ((((f24) rand()/RAND_MAX))/static_cast<f24>(5.0)) + static_cast<f24>(1.0);
                Xs[i][n][1] = ((((f24) rand()/RAND_MAX))/static_cast<f24>(5.0)) - static_cast<f24>(0.6);
                f24 h[M];
                h[0] = fpm::sqrt(static_cast<f24>(G(i, Xs[i][n][0]))) * static_cast<f24>(lam) * (static_cast<f24>(1.0) - fpm::pow(static_cast<f24>(E), 2)) * static_cast<f24>(randn(gen));
                h[1] = fpm::sqrt(static_cast<f24>(G(i, Xs[i][n][1]))) * static_cast<f24>(lam) * (static_cast<f24>(1.0) - fpm::pow(static_cast<f24>(E), 2)) * static_cast<f24>(randn(gen));
                e[n][0] = (e[n][0] * static_cast<f24>(E)) + h[0];
                e[n][1] = (e[n][1] * static_cast<f24>(E)) + h[1];
            }
        }
        else{
            for(int n = 0; n < N; n++){
                f24 coupling;
                if(USE_SPARSE)  calculate_coupling_sparse(i, n, nzr_W, nzr_c, w, d, r, c, nzr, lri, coupling);
                else            calculate_coupling(i, n, W[n], D[n], coupling);
                step(i, n, coupling);
                f24 h[M];
                h[0] = fpm::sqrt(static_cast<f24>(G(i, Xs[i][n][0]))) * static_cast<f24>(lam) * (static_cast<f24>(1.0) - fpm::pow(static_cast<f24>(E), 2)) * static_cast<f24>(randn(gen));
                h[1] = fpm::sqrt(static_cast<f24>(G(i, Xs[i][n][1]))) * static_cast<f24>(lam) * (static_cast<f24>(1.0) - fpm::pow(static_cast<f24>(E), 2)) * static_cast<f24>(randn(gen));
                e[n][0] = (e[n][0] * static_cast<f24>(E)) + h[0];
                e[n][1] = (e[n][1] * static_cast<f24>(E)) + h[1];
            }
        }
    }

    cout << "Simulation: " << chrono::duration<double, std::milli>(Time::now() - tick).count() << " ms" << endl;
    /* END: Simulation */

    
    /* Print Benchmark Results */

    if(BENCHMARK){
        cout << "Average time spent in COUPLING function: " << (float)(coupling_time/coupling_time_n) << " ms" << endl;
        cout << "Average time spent in STEP function: " << (float)(step_time/step_time_n) << " ms" << endl;
    }

    /* END: Print Benchmark Results */


    
    /* Write the results to the file */

    ofstream file_xs("xs.txt");
    file_xs << fixed << setprecision(10);
    for(int n = 0; n < N; n++){
        for(int i = 0; i < (int) (tf/dt); i++){
            file_xs << Xs[i][n][0] << endl;
        }
        file_xs << "-" << endl;
    }

    /* END: Write the results to the file */

    file_w.close();
    file_d.close();
    file_xs.close();
}

void calculate_coupling(int i, int n, f24 W[N], int D[N], f24& coupling){
    auto tick = Time::now();
    
    f24 c {0.0};
    for(int j = 0; j < N; j++){
        if(W[j] != static_cast<f24>(0.0)){
            if(i < D[j]){
                c += W[j] * static_cast<f24>(PRE(Xs[i - 1][n][0], static_cast<f24>(0.0)));
            } else{
                if(n == j)  c += W[j] * static_cast<f24>(PRE(Xs[i - 1][n][0], Xs[i - 1][j][0]));
                else        c += W[j] * static_cast<f24>(PRE(Xs[i - 1][n][0], Xs[i - D[j]][j][0]));
            }
        }
    }
    coupling = static_cast<f24>(POST(c));
    
    coupling_time += chrono::duration<double, std::milli>(Time::now() - tick).count();
    coupling_time_n++;
}

void calculate_coupling_sparse(int i, int n, int w_size, int nzr_size, f24* w, int* d, int* r, int* col, int* nzr, int* lri, f24& coupling){
    auto tick = Time::now();
    
    int lri_index = nzr[n];

    if(lri_index == -1){
        coupling = 0;
        return;
    }
    

    int index_start = lri[lri_index];
    int index_stop;
    if(lri_index == (nzr_size - 1)) index_stop = w_size;
    else                            index_stop = lri[lri_index + 1];


    f24 c = 0.0;
    for(int j = index_start; j < index_stop; j++){
        if(i < d[j]){
            c += w[j] * PRE(Xs[i - 1][n][0], 0.0);
        } else{
            if(d[j] == 0)   c += w[j] * PRE(Xs[i - 1][n][0], Xs[i - 1][col[j]][0]);
            else            c += w[j] * PRE(Xs[i - 1][n][0], Xs[i - d[j]][col[j]][0]);
        }
    }

    coupling = POST(c);
    
    coupling_time += chrono::duration<double, std::milli>(Time::now() - tick).count();
    coupling_time_n++;
}

void step(int i, int n, f24 coupling){
    auto tick = Time::now();

    f24 x = Xs[i-1][n][0];
    f24 y = Xs[i-1][n][1];
    f24 dx = dt * (FDX(x, y) + e[n][0]);
    f24 dy = dt * (FDY(x, coupling) + e[n][1]);
    Xs[i][n][0] = x + dx;
    Xs[i][n][1] = y + dy;

    step_time += chrono::duration<double, std::milli>(Time::now() - tick).count();
    step_time_n++;
}