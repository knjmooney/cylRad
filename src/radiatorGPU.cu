/*****************************************************
 * Name    : radiatorGPU.cpp
 * Author  : Kevin Mooney
 * Date    : 22/03/16
 * Updated : 15/08/16
 *  
 * Description:
 *   Definitions of the radiatorGPU class
 *
 * Notes:
 *****************************************************/

#include <chrono>

#include "cudaErrors.cuh"
#include "radiatorGPU.cuh"

// Parameters for defining the block dimensions 
#define XBLOCK 32
#define YBLOCK 32

//=========================================================================
//======================= SHARED MEMORY KERNEL ============================
//=========================================================================

// only update idx > 1 and ignore 1st two columns, could spawn less threads
// dynamic allocation complains when compiled twice with templates
__global__ void update_shared_d( int N, int M, int nIters, float *matrix, float *updatedMatrix ) {
  int idx    = blockIdx.x*blockDim.x+threadIdx.x;
  int idy    = blockIdx.y*blockDim.y+threadIdx.y;
  int index  = idx + M*idy;
  int lindex = 2 + threadIdx.x + threadIdx.y * ( blockDim.x + 4 );

  // size should be (dim.x+4)*dim.y
  // defining shared memory with templates,
  // shamelessly stolen from stack exchange 
  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float *local_matrix = reinterpret_cast<float *>(my_smem);

  if ( idx < M && idy < N && idx > 1) {

    // everyone load two to their left and two to their right
    // need a minimum x block size of 8
    local_matrix [ lindex - 2 ] = matrix [ index - 2 ]; 
    local_matrix [ lindex + 2 ] = matrix [ M*idy + ( ( idx + 2 ) % M ) ]; 

    __syncthreads();

    updatedMatrix[index] = ( + 0.37 * local_matrix[lindex - 2]
			     + 0.28 * local_matrix[lindex - 1]
			     + 0.20 * local_matrix[lindex    ]
			     + 0.12 * local_matrix[lindex + 1]
			     + 0.03 * local_matrix[lindex + 2]
			     );
  }
}

// update on the gpu
// two far left columns don't need threads, but this is not implemented
void RadiatorGPU::compute(const size_t &nIters ) {  
  int sizeOfTransfer = sizeof(float)*_nRows*_nCols;
  float * oldData;
  gpuErrchk ( cudaMalloc ( (void **) &oldData, sizeOfTransfer ) ); 
  gpuErrchk ( cudaMemcpy ( oldData, _data, sizeOfTransfer, cudaMemcpyDeviceToDevice ) );
  
  // set block and grid dimensions
  dim3 dimBlock(XBLOCK,YBLOCK);
  dim3 dimGrid ( (_nCols/dimBlock.x) + (!(_nCols%dimBlock.x)?0:1) 
		 , (_nRows/dimBlock.y) + (!(_nRows%dimBlock.y)?0:1));
  int sharedMemorySize = ( XBLOCK + 4 ) * YBLOCK * sizeof(float);

  for ( int i=0; i < nIters; i++ ) {
    update_shared_d<<<dimGrid,dimBlock,sharedMemorySize>>> ( _nRows, _nCols, nIters
							     , oldData, _data );
    std::swap ( _data, oldData );
  }
  gpuErrchk ( cudaGetLastError() );
  gpuErrchk ( cudaFree(oldData)  );
}

//=========================================================================
//========================== MEMORY TRANSFERS =============================
//=========================================================================

// Constructs pointer to device memory
RadiatorGPU::RadiatorGPU(const size_t &nRows, const size_t &nCols) : Radiator{nRows,nCols} {
  int sizeOfTransfer = sizeof(float)*_nRows*_nCols;
  gpuErrchk ( cudaMalloc ( (void **) &_data,     sizeOfTransfer ) ); 
}

// Frees device memory when RadiatorGPU falls out of scope
RadiatorGPU::~RadiatorGPU() {
  cudaFree(_data);
}

// Copies depending on transfer policy defined in radiator.hpp
void RadiatorGPU::copy ( enum transferPolicy tp, float* begin, float* end ) {
  size_t sizeOfTransfer = sizeof(float)*std::distance(begin,end);
  switch ( tp ) {
  case fromHost:    
    gpuErrchk ( cudaMemcpy ( _data, begin, sizeOfTransfer, cudaMemcpyHostToDevice ) );
    break;
  case fromDevice:
    gpuErrchk ( cudaMemcpy ( _data, begin, sizeOfTransfer, cudaMemcpyDeviceToDevice ) );
    break;
  default:
    throw (std::invalid_argument("Unrecognised transfer policy") );
  }
}

// Checks integrity of input
void RadiatorGPU::checkInput() {
  if ( _nRows % YBLOCK != 0 || _nCols % XBLOCK != 0 )
    throw std::logic_error ( "Implementation requires your problem size to be a multiple of the "
			     "block dimensions" );

  if ( XBLOCK < 8 )
    throw std::logic_error ( "Implementation requires your x block size to be greater than 8" );  
}
