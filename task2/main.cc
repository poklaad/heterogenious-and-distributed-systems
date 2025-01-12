#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << " us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << " GB/s";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_reduce(int n, OpenCL& opencl) 
{
    auto a = random_vector<float>(n);
    float result = 0, expected_result = 0;
    int last_reducing_size = n;

    opencl.queue.flush();
    cl::Kernel kernel_reduce(opencl.program, "kernel_reduce");

    auto t0 = clock_type::now();
    expected_result = reduce(a);

    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    opencl.queue.finish();
    auto t2 = clock_type::now();

    // Standart reduce steps
    for (int reducing_size = n/1024; reducing_size > 0; reducing_size /= 1024) 
    {
        cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, reducing_size*sizeof(float));

        kernel_reduce.setArg(0, d_a);
        kernel_reduce.setArg(1, d_result);
        kernel_reduce.setArg(2, reducing_size*1024);

        opencl.queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(reducing_size*1024), cl::NDRange(1024));

        // Preparing for the next reduce step
        d_a = d_result;
        last_reducing_size = reducing_size;
    }

    // Last reduce step on GPU
    if (last_reducing_size > 1) 
    {
        kernel_reduce.setArg(0, d_a);

        cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, sizeof(float));

        kernel_reduce.setArg(1, d_result);
        kernel_reduce.setArg(2, last_reducing_size);

        opencl.queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(last_reducing_size), cl::NDRange(last_reducing_size));

        d_a = d_result;
    }
    opencl.queue.finish();
    auto t3 = clock_type::now();

    opencl.queue.enqueueReadBuffer(d_a, CL_TRUE, 0, sizeof(float), &result);
    auto t4 = clock_type::now();

    // Difference between openmp and opencl versions
    std::cout << "Difference: "  << std::abs(result - expected_result) << "\n";
    print("reduce",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
}

void profile_scan_inclusive(int n, OpenCL& opencl) 
{
    auto a = random_vector<float>(n);
    Vector<float> result(a), expected_result(a);
    opencl.queue.flush();
    cl::Kernel kernel_scan(opencl.program, "kernel_scan");
    cl::Kernel kernel_scan_end(opencl.program, "kernel_scan_end");

    auto t0 = clock_type::now();
    scan_inclusive(expected_result);

    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), false);
    opencl.queue.finish();

    // Standart scan steps
    auto t2 = clock_type::now();
    int NDRange = 1024;
    for (int scan_size = 1; scan_size <= 1024*1024; scan_size *= 1024)
    {
        kernel_scan.setArg(0, d_a);
        kernel_scan.setArg(1, d_a);
        kernel_scan.setArg(2, scan_size);
        NDRange = (scan_size == 1024*1024) ? 10 : NDRange;
        opencl.queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(n/scan_size), cl::NDRange(NDRange));
    }

    // Last scan step
    kernel_scan_end.setArg(0, d_a);
    kernel_scan_end.setArg(1, 1024);
    opencl.queue.enqueueNDRangeKernel(kernel_scan_end, cl::NullRange, cl::NDRange(n/1024 - 1), cl::NullRange);
    opencl.queue.finish();

    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_a, begin(result), end(result));

    auto t4 = clock_type::now();

    // Final sum and difference between methods
    std::cout << "Difference: " << std::abs(result[1024*1024*10 - 1] - expected_result[1024*1024*10 -1]) << '\n';

    print("scan-inclusive",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n*n+n*n, t0, t1), bandwidth(n*n+n*n+n*n, t2, t3)});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024*1024*10, opencl);
    profile_scan_inclusive(1024*1024*10, opencl);
}

const std::string src = R"(
kernel void kernel_reduce(global float* a, global float* result, int n) 
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    local float sub_vec[1024];
    if (n >= 1024) 
    {
        // Standart reduce steps
        sub_vec[lid] = a[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Sum local sums
        for (int offset = get_local_size(0)/2; offset > 0; offset /= 2) {
            if (lid < offset) 
            {
                sub_vec[lid] += sub_vec[lid + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    else
    {
        // Last reduce step. Sum all results
        if (gid == 0) 
        {
            sub_vec[0] = 0;
            for (int k = 0; k < n; k++) 
            {
                sub_vec[0] += a[k];
            }
        }
    }
    if (lid == 0) result[get_group_id(0)] = sub_vec[0];
}

// Standart scan steps
kernel void kernel_scan(global float* a, global float* result, int step) 
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    local float sub_vec[1024];

    sub_vec[lid] = a[(step-1) + gid*step];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = sub_vec[lid];
    for (int offset = 1; offset < local_size; offset *= 2) {
        if (lid >= offset) 
        {
            sum += sub_vec[lid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        sub_vec[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    a[(step-1) + gid*step] = sub_vec[lid];
}

// Last scan step
kernel void kernel_scan_end(global float* a, int step) 
{
    const int gid = get_global_id(0);
    for (int i = 0; i < step-1; i++) 
    {
        a[(gid+1)*step + i] += a[(gid+1)*step - 1];
    }
}
)";

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
