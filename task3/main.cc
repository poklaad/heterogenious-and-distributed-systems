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

#include "filter.hh"
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

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
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
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL& opencl) 
{
    auto input = random_std_vector<float>(n);
    std::vector<float> result(n), expected_result;
    expected_result.reserve(n);
    opencl.queue.flush();
    
    cl::Kernel kernel_scan(opencl.program, "kernel_scan");
    cl::Kernel kernel_scan_end(opencl.program, "kernel_scan_end");
    cl::Kernel kernel_positive_numbers(opencl.program, "kernel_positive_numbers");
    cl::Kernel kernel_scatter(opencl.program, "kernel_scatter");

    auto t0 = clock_type::now();
    filter(input, expected_result, [] (float x) { return x > 0; }); // filter positive numbers

    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(input), end(input), true);
    cl::Buffer d_mask(opencl.context, CL_MEM_READ_WRITE, n*sizeof(float));
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, n*sizeof(float));

    auto t2 = clock_type::now();
    kernel_positive_numbers.setArg(0, d_a);
    kernel_positive_numbers.setArg(1, d_mask);
    opencl.queue.finish();
    opencl.queue.enqueueNDRangeKernel(kernel_positive_numbers, cl::NullRange, cl::NDRange(n), cl::NullRange);

    for(int i = 0; i < 2; ++i)
    {
        kernel_scan.setArg(0, d_mask);
        kernel_scan.setArg(1, d_mask);
        int size = (i == 0) ? 1 : 1024;
        kernel_scan.setArg(2, size);
        opencl.queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(n/size), cl::NDRange(1024));
    }

    kernel_scan_end.setArg(0, d_mask);
    kernel_scan_end.setArg(1, 1024);
    opencl.queue.enqueueNDRangeKernel(kernel_scan_end, cl::NullRange, cl::NDRange(n/1024-1), cl::NullRange);

    kernel_scatter.setArg(0, d_a);
    kernel_scatter.setArg(1, d_mask);
    kernel_scatter.setArg(2, d_result);
    opencl.queue.enqueueNDRangeKernel(kernel_scatter, cl::NullRange, cl::NDRange(n), cl::NullRange);
    opencl.queue.finish();

    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_result, begin(result), end(result));

    auto t4 = clock_type::now();
    for (int k = 0; k < n; k++) {
        if (result[k] == 0) 
        {
            result.resize(k);
            break;
        }
    }
    verify_vector(expected_result, result);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024*1024, opencl);
}

const std::string src = R"(
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

// Positive numbers mask
kernel void kernel_positive_numbers(global float* a, global float* result) 
{
    const int gid = get_global_id(0);
    result[gid] = convert_float(a[gid] > 0);
}

// Get positive numbers
kernel void kernel_scatter(global float* a, global float* mask, global float* result) 
{
    const int gid = get_global_id(0);
    int last_positive = mask[gid];
    if (last_positive > 0) 
    {
        if (gid == 0) 
        {
            result[last_positive - 1] = a[gid];
        } else 
        {
            int previous_positive = mask[gid-1];
            if (last_positive != previous_positive) 
            {
                result[last_positive - 1] = a[gid];
            }
        }
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
