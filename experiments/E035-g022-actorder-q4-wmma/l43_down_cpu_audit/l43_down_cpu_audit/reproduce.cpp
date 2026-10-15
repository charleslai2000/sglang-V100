#include <cmath>
#include <cstdint>
#include <cfenv>

extern "C" int rounding_mode() { return std::fegetround(); }

// Explicit single-rounded float FMA; no fast-math / no reassociation.
extern "C" void partials(const float* x, const float* w, float* p32, double* p64,
                          int M, int K, int N, int tile) {
    int P = (K+tile-1)/tile;
    for (int p=0; p<P; ++p) {
        int end = (p+1)*tile < K ? (p+1)*tile : K;
        for (int m=0; m<M; ++m) {
            float* a = p32 + (p*M+m)*N;
            double* d = p64 + (p*M+m)*N;
            for(int n=0;n<N;++n) {a[n]=0.0f;d[n]=0.0;}
            for(int k=p*tile;k<end;++k) {
                const float xv = x[m*K+k];
                const float* wk = w+k*N;
                for(int n=0;n<N;++n) {
                    a[n] = std::fma(xv,wk[n],a[n]);
                    d[n] += (double)xv * (double)wk[n];
                }
            }
        }
    }
}
extern "C" void reduce32(const float* p, float* out, int P, int count) {
    for(int i=0;i<count;++i)out[i]=0.f;
    for(int j=0;j<P;++j)
        for(int i=0;i<count;++i)out[i]+=p[j*count+i];
}
extern "C" void reduce64(const float* p, double* out, int P, int count) {
    for(int i=0;i<count;++i)out[i]=0.;
    for(int j=0;j<P;++j)
        for(int i=0;i<count;++i)out[i]+=(double)p[j*count+i];
}
extern "C" void reduce_kahan(const float* p, float* out, float* c, int P, int count) {
    for(int i=0;i<count;++i){out[i]=0.f;c[i]=0.f;}
    for(int j=0;j<P;++j)
        for(int i=0;i<count;++i) {
            float y=p[j*count+i]-c[i];
            float t=out[i]+y;
            c[i]=(t-out[i])-y;
            out[i]=t;
        }
}

// Arithmetic counterfactual, not a GPU implementation or performance prediction.
extern "C" void partial_split4(const float* x, const float* w, float* out,
                               int M, int K, int N, int mode) {
    int P=K/128;
    for(int p=0;p<P;++p)for(int m=0;m<M;++m) {
        for(int n=0;n<N;++n){
            float a[4]={0.f,0.f,0.f,0.f};
            for(int j=0;j<128;++j) {
                int lane=mode==0 ? j/32 : j%4;
                a[lane]=std::fma(x[m*K+p*128+j], w[(p*128+j)*N+n],a[lane]);
            }
            out[(p*M+m)*N+n]=(a[0]+a[1])+(a[2]+a[3]);
        }
    }
}

// Ogita-Rump-Oishi Dot2: TwoProduct using FMA, then Knuth TwoSum.
// This retains the rounding residual of both multiplication and addition.
extern "C" void partial_dot2(const float* x, const float* w, float* out,
                              int M,int K,int N) {
    int P=K/128;
    for(int p=0;p<P;++p)for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
        float sm=0.f,c=0.f;
        for(int j=0;j<128;++j) {
            float a=x[m*K+p*128+j],b=w[(p*128+j)*N+n];
            float h=a*b;
            float r=std::fma(a,b,-h);
            float t=sm+h;
            float z=t-sm;
            float e=(sm-(t-z))+(h-z);
            c=c+(e+r);
            sm=t;
        }
        out[(p*M+m)*N+n]=sm+c;
    }
}
