/***********************************************************************
 * Name    : cudaErrors.cuh
 * Author  : Kevin Mooney
 * Created : 20/03/16
 * Updated : 11/08/16
 *
 * Description:
 *   Macro for checking CUDA errors, idea modified from stack exchange
 *   gpuErrchk throws an exception if error code is not a success
 *
 * Notes:
 ***********************************************************************/

#pragma once

#include <stdexcept>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line )
{
  if (code != cudaSuccess) {
    std::ostringstream oss;
    oss << cudaGetErrorString(code) << " " <<  file << " " <<  line;
    throw std::logic_error ( oss.str() );
  }
}
