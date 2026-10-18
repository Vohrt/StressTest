#include <cstdio>
namespace cst { int verify_crypto_kernels(); }
int main() {
    const int rc = cst::verify_crypto_kernels();
    std::printf("crypto selftest: %s (%d)\n", rc == 0 ? "PASS" : "FAIL", rc);
    return rc;
}
