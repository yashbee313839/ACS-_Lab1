// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// OpenCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "acsmatmult/matmult.h"

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")

/*************************************/

// Macro to check clFunction outputs.
// Throw an error if not successful, to make debugging easier.
#define CHECK(err) if (err != CL_SUCCESS) { \
  throw std::runtime_error("OpenCL error: " + std::to_string(err) + \
  " in " + __FILE__ + " line " + std::to_string(__LINE__) ); \
}

///@brief A little enum class to help us parse clDeviceInfo
enum class ClInfoType {
  CHAR, SIZE_T, //... add your own info types
};

/// @brief Function to discover OpenCL devices and print some info on stdout.
static std::vector<cl_device_id> discoverDevices(cl_platform_id platform_id) {
  std::vector<cl_device_id> devices;
  // Discover devices for each platform
  cl_uint num_devices = 0;
  // Get number of devices of this type, we will only discover GPU devices for now.
  // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceIDs.html
  int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

  std::cout << "\tDevices: " << num_devices << std::endl;

  if ((err != CL_DEVICE_NOT_FOUND) || (num_devices != 0)) {
    // Get the devices of this type and insert them into the final list
    std::vector<cl_device_id> platform_type_devices(num_devices);
    CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, platform_type_devices.data(), &num_devices));
    // Insert the found devices into the final result
    devices.insert(std::end(devices), std::begin(platform_type_devices), std::end(platform_type_devices));

    // Many infos exist for devices. Also see:
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
    //
    // DISCLAIMER: IT IS HIGHLY RECOMMENDED TO DISCOVER SOME MORE STUFF ABOUT YOUR DEVICE WHEN YOU ARE GOING TO
    // USE IT MORE INTENSELY

    for (auto platform_type_device : platform_type_devices) {
      std::vector<cl_device_info> info_queries = {CL_DEVICE_NAME, CL_DEVICE_MAX_WORK_GROUP_SIZE};
      std::vector<ClInfoType> info_types = {ClInfoType::CHAR, ClInfoType::SIZE_T};
      size_t info_size = 0;
      for (unsigned int i = 0; i < info_queries.size(); i++) {
        // Get the query size
        CHECK(clGetDeviceInfo(platform_type_device, info_queries[i], 0, nullptr, &info_size));
        auto query = new char[info_size];
        CHECK(clGetDeviceInfo(platform_type_device, info_queries[i], info_size, query, &info_size));
        switch (info_types[i]) {
          case ClInfoType::SIZE_T: std::cout << *reinterpret_cast<size_t *>(query) << std::endl;
            break;
          default:std::cout << query << std::endl;
            break;
        }
        delete[] query;

      }
    }
  }
  return devices;
}

/// @brief Function to discover OpenCL platforms and print some info on stdout.
static std::vector<cl_platform_id> discoverPlatforms() {
  cl_uint num_platforms = 0;

  // Obtain the number of OpenCL platforms
  // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetPlatformIDs.html
  CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));

  // OpenCL sometimes outputs some stuff on cerr. Flush this stuff from the stream.
  std::cerr.flush();

  std::cout << "Found " << num_platforms << " OpenCL platform(s)." << std::endl;

  // Create an array to hold platform IDs.
  auto platform_ids = std::vector<cl_platform_id>(num_platforms);

  // Get the actual platform IDs
  CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), &num_platforms));

  // Identify the platform info that we would like to discover (more infos exist, but they are not interesting for us)
  const std::vector<cl_platform_info> platform_queries = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION};

  // Iterate over all platforms
  for (unsigned int p = 0; p < num_platforms; p++) {
    std::cout << "Platform " << p << std::endl;

    // Iterate over all platform infos we want to inquire
    for (auto platform_query : platform_queries) {
      size_t query_size = 0;

      // Get the current platform info length
      CHECK(clGetPlatformInfo(platform_ids[p], platform_query, 0, nullptr, &query_size));
      auto query = new char[query_size];

      // Get the actual info
      CHECK(clGetPlatformInfo(platform_ids[p], platform_query, query_size, query, &query_size));

      std::cout << '\t' << query << std::endl;

      delete[] query;
    }
  }

  return platform_ids;
}

Matrix<float> multiplyMatricesOCL(Matrix<float> a,
                                  Matrix<float> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENCL HERE */

  /* DISCLAIMER: this example code is written using the default C interface or OpenCL. There are also C++ bindings,
   * but we choose to use the C interfaces, because the C++ bindings might not always be available for every platform
   * you might use in the future. You are free, however, to use the C++ bindings that are usually in CL/cl2.hpp.
   */

  /* Example code partially inspired by: https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/ */
  std::cout << "OpenCL test." << std::endl;
  int err;
  auto c = Matrix<float>(a.rows, b.columns);  
  auto platforms = discoverPlatforms();
  if (platforms.empty()) {
	  throw std::runtime_error("No OpenCL platforms detected.");
  }
  auto devices = discoverDevices(platforms[0]);
  if (devices.empty()) {
	  throw std::runtime_error("No OpenCL devices detected.");
  }
  auto context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &err);
  auto queue = clCreateCommandQueue(context, devices[0], 0, &err);
  auto kernel_source =
	  "__kernel void matrixmult_kernel(__global float *A,\n " \
  		"__global float *B,\n " \
		"__global float *C,\n " \
		"int HeightA,\n" \
		"int WidthA, \n" \
		"int HeightB,\n" \
		"int WidthB) \n" \
  "{\n"\  

 	"// Obtain thread ID for 2D.\n" \
	"int row = get_global_id(1);\n" \
	"int col = get_global_id(0);\n" \
 	"\n " \
	" // Store the computed values by the thread\n" \
	"float result = 0.0f;\n" \
	"for (int k = 0; k < WidthA; k++) { \n" 	\
	"result  += A[row * WidthA + k] * B[k * WidthB +col]; \n" \
	"}\n" \
  "\n " \
 	" // Store it ot matrix c. Each thread writes one element.\n " \
	"C[row * WidthB + col] = result; \n" \

	"}\n";
  auto program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, nullptr, &err);
  clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
  auto kernel = clCreateKernel(program, "matrixmult_kernel", &err);
  int host_size_a = a.rows * a.columns*sizeof(float);
  int host_size_b = b.rows *b.columns*sizeof(float);
  int host_size_c = a.rows *b.columns*sizeof(float);
  auto device_A  = clCreateBuffer(context, CL_MEM_READ_ONLY, host_size_a, nullptr, nullptr);
  auto device_B  = clCreateBuffer(context, CL_MEM_READ_ONLY, host_size_b, nullptr, nullptr);
  auto device_C  = clCreateBuffer(context, CL_MEM_READ_WRITE, host_size_c, nullptr, nullptr);
  CHECK(clEnqueueWriteBuffer(queue, device_A, CL_TRUE, 0, host_size_a, &a(0,0), 0, nullptr, nullptr));
  CHECK(clEnqueueWriteBuffer(queue, device_B, CL_TRUE, 0, host_size_b, &b(0,0), 0, nullptr, nullptr));
  CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*) &device_A));
  CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_B));
  CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_C));
  CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void*)&a.rows));//host_size_a en b
  CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void*)&a.columns));
  CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void*)&b.rows));
  CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void*)&b.columns));
  const int TS = 1;
  const size_t local_size[2] = { TS, TS };
  const size_t global_size[2] = {c.columns, c.rows};
  CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr));
  clFinish(queue);
  CHECK(clEnqueueReadBuffer(queue, device_C, CL_TRUE, 0, host_size_c, &c(0,0), 0, nullptr, nullptr));
  clReleaseMemObject(device_A);
  clReleaseMemObject(device_B);
  clReleaseMemObject(device_C);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return c;
}

Matrix<double> multiplyMatricesOCL(Matrix<double> a,
                                   Matrix<double> b) {
  int err;
  auto c = Matrix<double>(a.rows, b.columns); 
  auto platforms = discoverPlatforms();
  if (platforms.empty()) {
	  throw std::runtime_error("No OpenCL platforms detected.");
  }
  auto devices = discoverDevices(platforms[0]);
  if (devices.empty()) {
	  throw std::runtime_error("No OpenCL devices detected.");
  }
  auto context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &err);
  auto queue = clCreateCommandQueue(context, devices[0], 0, &err);
  auto kernel_source =
	  "__kernel void matrixmult_kernel(__global double *A,\n " \
  		"__global double *B,\n " \
		"__global double *C,\n " \
		"int HeightA,\n" \
		"int WidthA, \n" \
		"int HeightB,\n" \
		"int WidthB) \n" \
 "{\n"\  

 	"// Obtain thread ID for 2D.\n" \
	"int row = get_global_id(1);\n" \
	"int col = get_global_id(0);\n" \
 	"\n " \
	" // Store the computed values by the thread\n" \
	"double result = 0.0;\n" \
	"for (int k = 0; k < WidthA; k++) { \n" 	\
	"result  += A[row * WidthA + k] * B[k * WidthB +col]; \n" \
	"}\n" \
 "\n " \
 	" // Store it ot matrix c. Each thread writes one element.\n " \
	"C[row * WidthB + col] = result; \n" \

	"}\n";
  auto program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, nullptr, &err);
  clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
  auto kernel = clCreateKernel(program, "matrixmult_kernel", &err);
  int host_size_a = a.rows * a.columns*sizeof(double);
  int host_size_b = b.rows *b.columns*sizeof(double);
  int host_size_c = a.rows *b.columns*sizeof(double);
  auto device_A  = clCreateBuffer(context, CL_MEM_READ_ONLY, host_size_a, nullptr, nullptr);
  auto device_B  = clCreateBuffer(context, CL_MEM_READ_ONLY, host_size_b, nullptr, nullptr);
  auto device_C  = clCreateBuffer(context, CL_MEM_READ_WRITE, host_size_c, nullptr, nullptr);
  CHECK(clEnqueueWriteBuffer(queue, device_A, CL_TRUE, 0, host_size_a, &a(0,0), 0, nullptr, nullptr));
  CHECK(clEnqueueWriteBuffer(queue, device_B, CL_TRUE, 0, host_size_b, &b(0,0), 0, nullptr, nullptr));
  CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*) &device_A));
  CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_B));
  CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_C));
  CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void*)&a.rows));
  CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void*)&a.columns));
  CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void*)&b.rows));
  CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void*)&b.columns));
  const int TS = 1;
  const size_t local_size[2] = { TS, TS };
  const size_t global_size[2] = {c.columns, c.rows};
  CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr));
  clFinish(queue);
  CHECK(clEnqueueReadBuffer(queue, device_C, CL_TRUE, 0, host_size_c, &c(0,0), 0, nullptr, nullptr));
  clReleaseMemObject(device_A);
  clReleaseMemObject(device_B);
  clReleaseMemObject(device_C);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return c;
}

/*************************************/
#pragma GCC pop_options
/**********i***************************/
