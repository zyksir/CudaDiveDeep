#include "utils.h"
#include <functional>

using std::clog;
using std::endl;

class BaseTest {
public:
    BaseTest(size_t _float_calculation_num, const string& _kernel_name, uint _repeat_nums): 
		float_calculation_num(_float_calculation_num), kernel_name(_kernel_name), repeat_nums(_repeat_nums) {
        cudaEventCreate(&start);
		cudaEventCreate(&stop);
    }
    ~BaseTest() {
        cudaEventDestroy(start);
		cudaEventDestroy(stop);
    }
  	virtual void run_baseline() = 0;

protected:
    cudaEvent_t start;
    cudaEvent_t stop;
    const string kernel_name;
	const uint repeat_nums;
    const uint64_t float_calculation_num;
    // print the performance result for the test
    void print_performance_result(const float elapsed_ms, const string& name) {
        float flops = repeat_nums * float_calculation_num /(elapsed_ms * 1e9); // TFLOPS means 1e12, while ms means 1e-3
        clog << "[" << kernel_name + name << "]Pass\tFLOPS:" << flops << "TFLOPS\tTimeCost:" << elapsed_ms / repeat_nums << "ms" << std::endl;
    }

    bool is_diff_single_item(float a, float b) {
        return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
    }

    bool is_diff_matrix(float* a, float* b, size_t msize) {
        for(int i = 0; i < msize; ++i) {
            if (is_diff_single_item(a[i], b[i])) {
                clog << "[FAIL]Find different is pos " << i << ": Expected: " << a[i] << ", Get: " << b[i] << endl;
                return true;
            }
        }
        return false;
    }

    void cuda_event_record() {
        cudaEventRecord(start);
    }

    float cuda_event_stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		float elapsed_ms = 0;
		cudaEventElapsedTime(&elapsed_ms, start, stop);
        return elapsed_ms;
    }

};