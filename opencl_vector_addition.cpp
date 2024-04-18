#include <iostream>
#include <vector>
#include <CL/cl.h>

using namespace std;

void performVectorAddition(vector<int>& vecA, vector<int>& vecB, vector<int>& vecC, int size) {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, vecA.data(), NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, vecB.data(), NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * size, NULL, NULL);

    const char* kernelSourceCode =
        "__kernel void vector_addition(__global const int* A, __global const int* B, __global int* C, int size) {\n"
        "   int index = get_global_id(0);\n"
        "   if (index < size) {\n"
        "       C[index] = A[index] + B[index];\n"
        "   }\n"
        "}\n";
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCode, NULL, NULL);

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "vector_addition", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    size_t globalSize = size;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int) * size, vecC.data(), 0, NULL, NULL);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    int size = 5;

    vector<int> vecA = {1, 2, 3, 4, 5};
    vector<int> vecB = {6, 7, 8, 9, 10};
    vector<int> vecC(size);

    performVectorAddition(vecA, vecB, vecC, size);

    cout << "Result of vector addition:" << endl;
    for (int i = 0; i < size; ++i) {
        cout << vecC[i] << " ";
    }
    cout << endl;

    return 0;
}
