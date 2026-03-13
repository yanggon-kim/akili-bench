#include <iostream>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6
#define RT(x) if (x) { std::cout<<"ERR "<<#x<<"\n"; exit(1); }

template <typename T>
static bool cmp(T a, T b) {
    union { float f; int32_t i; } fa{a}, fb{b};
    return std::abs(fa.i - fb.i) <= FLOAT_ULP;
}

vx_device_h device=nullptr;
vx_buffer_h X_buf=nullptr;
vx_buffer_h bin_buf=nullptr;
vx_buffer_h arg_buf=nullptr;

void cleanup(){
    if(device){
        vx_mem_free(X_buf);
        vx_mem_free(bin_buf);
        vx_mem_free(arg_buf);
        vx_dev_close(device);
    }
}

int main(){
    RT(vx_dev_open(&device));

    int total = 1024;
    std::vector<float> h_X(total);
    std::vector<float> ref(total);

    for(int i=0;i<total;i++){
        h_X[i] = (rand() / float(RAND_MAX))*2 - 1;
        ref[i] = h_X[i] < 0 ? 0 : h_X[i];
    }

    RT(vx_mem_alloc(device,total*4,VX_MEM_READ|VX_MEM_WRITE,&X_buf));

    uint64_t X_addr;
    vx_mem_address(X_buf,&X_addr);

    RT(vx_copy_to_dev(X_buf,h_X.data(),0,total*4));

    RT(vx_upload_kernel_file(device,"kernel.vxbin",&bin_buf));

    kernel_arg_relu_t arg{};
    arg.X_addr = X_addr;
    arg.total  = total;
    arg.grid_dim[0] = total;

    RT(vx_upload_bytes(device,&arg,sizeof(arg),&arg_buf));
    RT(vx_start(device,bin_buf,arg_buf));
    RT(vx_ready_wait(device,VX_MAX_TIMEOUT));

    RT(vx_copy_from_dev(h_X.data(),X_buf,0,total*4));

    int errors=0;
    for(int i=0;i<total;i++){
        if(!cmp(h_X[i],ref[i])){
            if(errors<20)
                printf("Mismatch [%d] GPU=%f CPU=%f\n",i,h_X[i],ref[i]);
            errors++;
        }
    }

    cleanup();

    if(errors==0) printf("ReLU PASSED\n");
    else printf("ReLU FAILED %d errors\n",errors);

    return errors;
}
