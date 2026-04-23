#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Body ibody[N/size], send_body[N/size], jbody[N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = send_body[i].x = drand48();
    ibody[i].y = send_body[i].y = drand48();
    ibody[i].m = send_body[i].m = drand48();
    ibody[i].fx = send_body[i].fx = ibody[i].fy = send_body[i].fy = 0;
    jbody[i].x = jbody[i].y = jbody[i].m = jbody[i].fx = jbody[i].fy = 0;
  }
  int send_to = (rank - 1 + size) % size;
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);
  MPI_Win win;
  MPI_Win_create(jbody, (N/size)*sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  for(int irank=0; irank<size; irank++) {
    MPI_Win_fence(0, win);
    MPI_Put(send_body, N/size, MPI_BODY, send_to, 0, N/size, MPI_BODY, win);
    MPI_Win_fence(0, win);
    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
    for(int i=0; i<N/size; i++) {
      send_body[i] = jbody[i];
    }
  }
  MPI_Win_free(&win);
  MPI_Type_free(&MPI_BODY);
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  MPI_Finalize();
}
