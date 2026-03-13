#include <iostream>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define RT(x) if(x){printf("ERR %s\n",#x);exit(1);}

vx_device_h device=nullptr;
vx_buffer_h I_buf=nullptr;
vx_buffer_h O_buf=nullptr;
vx_buffer_h bin_buf=nullptr;
vx_buffer_h arg_buf=nullptr;

void cleanup(){
    if(device){
        vx_mem_free(I_buf);
        vx_mem_free(O_buf);
        vx_mem_free(bin_buf);
        vx_mem_free(arg_buf);
        vx_dev_close(device);
    }
}

int main(){
    RT(vx_dev_open(&device));

    int C=4, H=32, W=32;
    int H2=H/2, W2=W/2;

    std::vector<float> I(C*H*W);
    std::vector<float> O(C*H2*W2);
    std::vector<float> ref(C*H2*W2);

    for(auto& v : I) v = rand()/float(RAND_MAX);

    // CPU
    for(int oc=0;oc<C;oc++){
        for(int oy=0;oy<H2;oy++){
            for(int ox=0;ox<W2;ox++){
                float m=-1e30f;
                for(int ky=0;ky<2;ky++){
                    for(int kx=0;kx<2;kx++){
                        int iy=2*oy+ky, ix=2*ox+kx;
                        float v=I[oc*H*W+iy*W+ix];
                        if(v>m) m=v;
                    }
                }
                ref[oc*H2*W2+oy*W2+ox]=m;
            }
        }
    }

    RT(vx_mem_alloc(device,I.size()*4,VX_MEM_READ,&I_buf));
    RT(vx_mem_alloc(device,O.size()*4,VX_MEM_WRITE,&O_buf));

    uint64_t I_addr,O_addr;
    vx_mem_address(I_buf,&I_addr);
    vx_mem_address(O_buf,&O_addr);

    RT(vx_copy_to_dev(I_buf,I.data(),0,I.size()*4));
    RT(vx_upload_kernel_file(device,"kernel.vxbin",&bin_buf));

    kernel_arg_pool_t arg{};
    arg.I_addr=I_addr;
    arg.O_addr=O_addr;
    arg.C=C; arg.H=H; arg.W=W;

    arg.grid_dim[0]=W2;
    arg.grid_dim[1]=H2;
    arg.grid_dim[2]=C;

    RT(vx_upload_bytes(device,&arg,sizeof(arg),&arg_buf));
    RT(vx_start(device,bin_buf,arg_buf));
    RT(vx_ready_wait(device,VX_MAX_TIMEOUT));

    RT(vx_copy_from_dev(O.data(),O_buf,0,O.size()*4));

    int errors=0;
    printf("Size of Output: %d\n",O.size());
    for(size_t i=0;i<O.size();i++){
        if(fabs(O[i]-ref[i])>1e-5){
            if(errors<20)
                printf("ERR[%zu]: GPU=%f CPU=%f\n",i,O[i],ref[i]);
            errors++;
        }
    }

    cleanup();
    if(errors==0) printf("POOL PASSED\n");
    else printf("POOL FAILED %d errors\n",errors);
    return errors;
}
