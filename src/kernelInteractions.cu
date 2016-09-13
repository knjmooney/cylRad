/*********************************************************
 * Name    : kernelInteractions.cu
 * Author  : Kevin Mooney
 * Created : 12/08/16
 * Updated : 
 *
 * Description:
 *
 * Notes:
 *********************************************************/

#include "cudaErrors.cuh"

void transfer2Host (float * begin, float * end, float * begin2 ) {
  size_t sizeOfTransfer = sizeof(float)*std::distance(begin,end);
  gpuErrchk ( cudaMemcpy ( begin2, begin, sizeOfTransfer, cudaMemcpyDeviceToHost ) );
}
