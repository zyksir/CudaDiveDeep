#include "utils.h"
#include <functional>

using std::clog;
using std::endl;

class BaseTest {
public:
    BaseTest(size_t _float_calculation_num, const string& _kernel_name, uint _repeat_nums): 
		float_calculation_num(_float_calculation_num), kernel_name(_kernel_name), repeat_nums(_repeat_nums) {}
  	virtual void run_baseline() = 0;

protected:
    const string kernel_name;
	const uint repeat_nums;
    const uint64_t float_calculation_num;
    // print the performance result for the test
    void print_performance_result(const duration<double>& duration, const string& name) {
        uint64_t run_time_ms = duration_cast<microseconds>(duration).count();
        float gflops = repeat_nums * float_calculation_num /(run_time_ms * 1e3);
        clog << "[" << kernel_name + name << "]Pass\tGFLOPS:" << gflops << "gflops\tTimeCost:" << duration.count()*1e6 / repeat_nums << "us" << std::endl;
    }

    bool is_diff_single_item(float a, float b) {
        return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
    }

    bool is_diff_matrix(float* a, float* b, size_t msize) {
        for(int i = 0; i < msize; ++i) {
            if (is_diff_single_item(a[i], b[i])) {
                clog << "Find different is pos " << i << ": Expected: " << a[i] << ", Get: " << b[i] << endl;
                return true;
            }
        }
        return false;
    }

};