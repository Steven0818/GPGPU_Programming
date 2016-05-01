#include "lab3.h"
#include <cstdio>
#include "iostream"


using namespace std;
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__host__ __device__ float saturator(float num){
	if(num>=255) return 255.0;
	else if (num<=0) return 0;
	else 
		return num;
} 

__global__ void PoissonImageEditing(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,int i
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		float  count =4;
		float  countb =4;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb){
			float tmpTN1 = (yt-1>=0)?target[(curt-wt)*3+0]:0;
			float tmpTN2 = (yt-1>=0)?target[(curt-wt)*3+1]:0;
			float tmpTN3 = (yt-1>=0)?target[(curt-wt)*3+2]:0;
			count = (yt-1>=0)?count:count-1;
			float tmpTS1 = (yt+1<ht)?target[(curt+wt)*3+0]:0;
			float tmpTS2 = (yt+1<ht)?target[(curt+wt)*3+1]:0;
			float tmpTS3 = (yt+1<ht)?target[(curt+wt)*3+2]:0;
			count = (yt+1<ht)?count:count-1;
			float tmpTW1 = (xt-1>=0)?target[(curt-1)*3+0]:0;
			float tmpTW2 = (xt-1>=0)?target[(curt-1)*3+1]:0;
			float tmpTW3 = (xt-1>=0)?target[(curt-1)*3+2]:0;
			count = (xt-1>=0)?count:count-1;
			float tmpTE1 = (xt+1<wt)?target[(curt+1)*3+0]:0;
			float tmpTE2 = (xt+1<wt)?target[(curt+1)*3+1]:0;
			float tmpTE3 = (xt+1<wt)?target[(curt+1)*3+2]:0;
			count = (xt+1<wt)?count:count-1;
			float tmpBN1 = (yb-1>=0)?background[(curb-wb)*3+0]:0;
			float tmpBN2 = (yb-1>=0)?background[(curb-wb)*3+1]:0;
			float tmpBN3 = (yb-1>=0)?background[(curb-wb)*3+2]:0;
			countb = (yb-1>=0)?countb:countb-1;
			float tmpBS1 = (yb+1<hb)?background[(curb+wb)*3+0]:0;
			float tmpBS2 = (yb+1<hb)?background[(curb+wb)*3+1]:0;	
			float tmpBS3 = (yb+1<hb)?background[(curb+wb)*3+2]:0;
			countb = (yb+1<hb)?countb:countb-1;
			float tmpBW1 = (xb-1>=0)?background[(curb-1)*3+0]:0;
			float tmpBW2 = (xb-1>=0)?background[(curb-1)*3+1]:0;
			float tmpBW3 = (xb-1>=0)?background[(curb-1)*3+2]:0;
			countb = (xb-1>=0)?countb:countb-1;
			float tmpBE1 = (xb+1<wb)?background[(curb+1)*3+0]:0;
			float tmpBE2 = (xb+1<wb)?background[(curb+1)*3+1]:0;
			float tmpBE3 = (xb+1<wb)?background[(curb+1)*3+2]:0;
			countb = (xb+1<wb)?countb:countb-1;
			output[(curb)*3+0] = (count*target[curt*3+0]-(tmpTN1+tmpTS1+tmpTW1+tmpTE1)+(tmpBE1+tmpBW1+tmpBS1+tmpBN1))/countb;
			output[(curb)*3+1] = (count*target[curt*3+1]-(tmpTN2+tmpTS2+tmpTW2+tmpTE2)+(tmpBE2+tmpBW2+tmpBS2+tmpBN2))/countb;
			output[(curb)*3+2] = (count*target[curt*3+2]-(tmpTN3+tmpTS3+tmpTW3+tmpTE3)+(tmpBE3+tmpBW3+tmpBS3+tmpBN3))/countb;
		}
	}

}
void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	int iteration = 20000;
	//simplClone<<<dim3(CeilDiv(wt+2,32), CeilDiv(ht+2,16)), dim3(32+2,16+2)>>>()
	for(int i=0;i<iteration;i++){
		float* referOutput;
		cudaMalloc((void**)&referOutput,wb*hb*sizeof(float)*3);
		cudaMemcpy(referOutput,output,wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
		
		PoissonImageEditing<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
			referOutput, target, mask, output,
			wb, hb, wt, ht, oy, ox,i
		);
		cudaFree(referOutput);
	}
}
