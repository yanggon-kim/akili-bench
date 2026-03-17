// hack to silence host.bc referencing missing kernel wrapper
extern "C" {
    void _Z14forward_kernelPKfS0_S0_iiiiiifPfS1_S1__wrapper(void) { }
}
