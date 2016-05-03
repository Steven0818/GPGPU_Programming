#include "lab3.h"
#include <cstdio>
#include "iostream"


using namespace std;
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void backgroundCopy(
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
	if (yt < ht && xt < wt ) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curt*3+0] = background[curb*3+0];
			output[curt*3+1] = background[curb*3+1];
			output[curt*3+2] = background[curb*3+2];
		}
	}
}

__global__ void SimpleClone(
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
		if (0 <= yb & yb < hb && 0 <= xb && xb < wb) {
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
	int curt = yt*wt+xt;
	const int yb = yt+oy;
	const int xb = xt+ox;
	const int curb = yb*wb+xb; 
	if (yt>=0 &&yt < ht && xt>=0 && xt < wt){
	 	if(mask[curt] > 127.0f) {
			float  count =4;
			float  countb =4;
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
			output[(curb)*3+0] = (count*target[curt*3+0]-(tmpTN1+tmpTS1+tmpTW1+tmpTE1)+(tmpBE1+tmpBW1+tmpBS1+tmpBN1))/4;
			output[(curb)*3+1] = (count*target[curt*3+1]-(tmpTN2+tmpTS2+tmpTW2+tmpTE2)+(tmpBE2+tmpBW2+tmpBS2+tmpBN2))/4;
			output[(curb)*3+2] = (count*target[curt*3+2]-(tmpTN3+tmpTS3+tmpTW3+tmpTE3)+(tmpBE3+tmpBW3+tmpBS3+tmpBN3))/4;
		}
	}

}

__global__ void upSampling2(
	float* output,float* bufferOutput2,
	const int wb, const int hb,
	int wb2,int hb2,
	bool flagx,bool flagy
)
{
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int xb2 = xb / 2;  
	const int yb2 = yb / 2; 
	int curb = yb*wb+xb;
	int curb2 = yb2*wb2+xb2;
	if (yb < hb && xb < wb){
		output[curb*3+0] = bufferOutput2[curb2*3+0];
		output[curb*3+1] = bufferOutput2[curb2*3+1];
		output[curb*3+2] = bufferOutput2[curb2*3+2]; 
	}
}
__global__ void downSampling2(
	const float *oriImg,
	float *downSampleImg,
	const int w2, const int h2,
	const int w,  const int h
)
{
	const int y2 = blockIdx.y * blockDim.y + threadIdx.y;
	const int x2 = blockIdx.x * blockDim.x + threadIdx.x;
	const int cur2 = w2*y2+x2;
	if (y2 < h2 && x2 < w2){
		downSampleImg[cur2*3+0] = (oriImg[(w*(y2*2)+x2*2)*3+0] + oriImg[(w*(y2*2)+x2*2+1)*3+0] + oriImg[(w*(y2*2+1)+x2*2)*3+0] +oriImg[(w*(y2*2+1)+x2*2+1)*3+0])/4;	
		downSampleImg[cur2*3+1] = (oriImg[(w*(y2*2)+x2*2)*3+1] + oriImg[(w*(y2*2)+x2*2+1)*3+1] + oriImg[(w*(y2*2+1)+x2*2)*3+1] +oriImg[(w*(y2*2+1)+x2*2+1)*3+1])/4;
		downSampleImg[cur2*3+2] = (oriImg[(w*(y2*2)+x2*2)*3+2] + oriImg[(w*(y2*2)+x2*2+1)*3+2] + oriImg[(w*(y2*2+1)+x2*2)*3+2] +oriImg[(w*(y2*2+1)+x2*2+1)*3+2])/4;
	}

		
}

__global__ void maskDownSampling2(
	const float *oriImg,
	float *downSampleImg,
	const int w2, const int h2,
	const int w,  const int h
)
{
	const int y2 = blockIdx.y * blockDim.y + threadIdx.y;
	const int x2 = blockIdx.x * blockDim.x + threadIdx.x;
	const int cur2 = w2*y2+x2;
	if (y2 < h2 && x2 < w2){
		downSampleImg[cur2] = (oriImg[(w*(y2*2)+x2*2)] + oriImg[(w*(y2*2)+x2*2+1)] + oriImg[(w*(y2*2+1)+x2*2)] +oriImg[(w*(y2*2+1)+x2*2+1)])/4;	
	}

		
}

__global__ void copy2output(
	float *output,
	const float *tmp,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox

)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt;
	const int xb = ox+xt;
	const int curb = wb*yb+xb;
	if (yt < ht && xt < wt){
		output[curb*3+0] = tmp[curt*3+0];
		output[curb*3+1] = tmp[curt*3+1];
		output[curb*3+2] = tmp[curt*3+2];
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
	bool flagx = (wb%2==0)?false:true;
	bool flagy = (hb%2==0)?false:true;
	//cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	int wb2 = wb/2;
	int hb2 = hb/2;
	int wt2 = wt/2;
	int ht2 = ht/2;
	int ox2 = ox/2;
	int oy2 = oy/2;
	int compleW = wb2+2;
	int compleH = hb2+2;


	float * background2;
	float * newbackground;
	float * newComplementbackground;
	float * complebackground2;
	float * target2;
	float * mask2;
	float * bufferOutput2;
	float * bufferOutput2_second;
	
	//cudaMalloc((void**)&bufferOutput,wt*ht*sizeof(float)*3);
	cudaMalloc((void**)&bufferOutput2,wb2*hb2*sizeof(float)*3);
	
	cudaMalloc((void**)&bufferOutput2_second,wb2*hb2*sizeof(float)*3);
	cudaMalloc((void**)&background2,hb2*wb2*sizeof(float)*3);
	
	cudaMalloc((void**)&newComplementbackground,compleW*compleH*sizeof(float)*3);
	cudaMalloc((void**)&complebackground2,compleW/2*compleH/2*sizeof(float)*3);
	cudaMalloc((void**)&target2,wt2*ht2*sizeof(float)*3);
	cudaMalloc((void**)&mask2,wt2*ht2*sizeof(float));


	// down sample or upsample wrong 

	downSampling2<<<dim3(CeilDiv(wb2,32), CeilDiv(hb2,16)), dim3(32,16)>>>(
		background,background2,wb2,hb2,wb,hb
	);



	maskDownSampling2<<<dim3(CeilDiv(wt2,32), CeilDiv(ht2,16)), dim3(32,16)>>>(
		mask,mask2,wt2,ht2,wt,ht
	);

	downSampling2<<<dim3(CeilDiv(wt2,32), CeilDiv(ht2,16)), dim3(32,16)>>>(
		target,target2,wt2,ht2,wt,ht
	);
	
	
	
	
	
	//cudaMemcpy(output,target2,ht2*wt2*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	 SimpleClone<<<dim3(CeilDiv(wt/2,32), CeilDiv(ht/2,16)), dim3(32,16)>>>(
	 	target2, mask2,background2,
	 	wb2, hb2, wt2, ht2,oy2, ox2
	 );

	cudaMemcpy(bufferOutput2,background2,hb2*wb2*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bufferOutput2_second,background2,hb2*wb2*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	
	 

	int iteration = 6000;
	for(int i=0;i<iteration;i++){
		if(i%2 ==0){
			PoissonImageEditing<<<dim3(CeilDiv(wt2,32), CeilDiv(ht2,16)), dim3(32,16)>>>(
			bufferOutput2, target2, mask2, bufferOutput2_second,
			wb2,hb2,
			wt2,ht2,
			oy2,ox2,i
			);
		}
		else{
			PoissonImageEditing<<<dim3(CeilDiv(wt2,32), CeilDiv(ht2,16)), dim3(32,16)>>>(
			bufferOutput2_second, target2,mask2,bufferOutput2,
			wb2,hb2,
			wt2,ht2,
			oy2,ox2,i
			);
		}
	}
	cudaMemcpy(output,mask,ht*wt*sizeof(float),cudaMemcpyDeviceToDevice);
	if(iteration%2 == 0){
		upSampling2<<<dim3(CeilDiv(wb,32), CeilDiv(hb,16)), dim3(32,16)>>>(
			output,bufferOutput2,wb,hb,wb2,hb2,flagx,flagy
		);
		
	}
	else{
		float* tmp;
		cudaMalloc((void**)&tmp,wt*ht*sizeof(float)*3);
		upSampling2<<<dim3(CeilDiv(wb,32), CeilDiv(hb,16)), dim3(32,16)>>>(
			output,bufferOutput2,wb,hb,wb2,hb2,flagx,flagy
		);	
	 }
	cudaFree(bufferOutput2);
	cudaFree(bufferOutput2_second);
}
