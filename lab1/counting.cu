#include "counting.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <math.h>

using namespace std;
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__device__ int LC(int i) {return i*2;}
__device__ int RC(int i) {return i*2+1;}
__device__ bool IsRightNode(int nodeNum){return (nodeNum%2!=0);}
__device__ bool IsLeftNode(int nodeNum){return (nodeNum%2==0);}
__device__ int GetLeftNodeFromRight(int nodeNum){return nodeNum-1;}
__device__ int GetLeftNodeRCFromRight(int nodeNum){return (nodeNum-1)*2+1;}
__device__ int GetLeftUpNodeFromLeft(int nodeNum){return nodeNum/2-1;}
__device__ bool Is2Power(int nodeNum){return !(nodeNum&nodeNum-1);}
__device__ void traceNode(int nodeNum,int* segTree,int& length,bool IsUp,int treeSize){
    if(nodeNum>=treeSize) {return;}
    if(IsUp){
        if(segTree[nodeNum] != 0 && IsLeftNode(nodeNum)){
            length+=segTree[nodeNum];
            if(Is2Power(nodeNum)){return;}
            else{
                traceNode(GetLeftUpNodeFromLeft(nodeNum),segTree,length,IsUp,treeSize);
            }
        }
        else if(segTree[nodeNum] != 0 && IsRightNode(nodeNum)){
            length+=segTree[nodeNum];
            traceNode(GetLeftNodeFromRight(nodeNum),segTree,length,IsUp,treeSize);
        }
        else{
            IsUp = false;
            traceNode(RC(nodeNum),segTree,length,IsUp,treeSize);
        } 
    }
    else{
        if(segTree[nodeNum] != 0){
            length += segTree[nodeNum];
            traceNode(GetLeftNodeRCFromRight(nodeNum),segTree,length,IsUp,treeSize);
        }
        else{
            traceNode(RC(nodeNum),segTree,length,IsUp,treeSize);
        }
    }
    
}
__global__ void Build(const char *text,int *segTree,int text_size,int level,int treeHeight,int upperNode,int columnNum){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int nodeNum = upperNode + idx;
    if(idx<columnNum){
        if(level == 0) {
            if(idx>=text_size){
                    segTree[nodeNum] = 0;
            }
            else{
                if(text[idx] != '\n'){segTree[nodeNum] = 1;}
                else{segTree[nodeNum] = 0;}
            }
        }
        else{
            if(segTree[LC(nodeNum)]==0 || segTree[RC(nodeNum)]==0){
                   segTree[nodeNum] = 0; 
            }
            else{
                segTree[nodeNum] = segTree[LC(nodeNum)] + segTree[RC(nodeNum)];
            }    
        }
    }
}

 __global__ void Trace(const char *text,int* pos,int* segTree,int upperNode,int treeSize){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int nodeNum = upperNode+idx;
    if(text[idx] == '\n'){
        pos[idx] = 0;
    }
    else{
        int length = 0;
        traceNode(nodeNum,segTree,length,true,treeSize);
        pos[idx] = length;
    }
}

__global__ void updatePos(const char *text,int* pos){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(text[idx] == '\n'&& pos[idx-1] %2 == 1&& idx>=1){
        pos[idx-1] = 0;
    }
}

__global__ void copyText(char* text,char* tmpText){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    tmpText[idx] = text[idx];
}

__global__ void exchange(char* text,char* tmpText,int* pos){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(pos[idx]!=0){
        if(pos[idx]%2 == 0){
            text[idx] = tmpText[idx-1];
        }
        else{
            text[idx] = tmpText[idx+1];
        }
    }
}

void CountPosition(const char *text, int *pos, int text_size)
{
    int* segTree;
    int treeSize; // total 1~ 
    int treeHeight;
    if(log2(text_size) - (int)log2(text_size)  == 0){
        // power of 2
        treeSize = 2*text_size;
    }
    else{
        treeSize= 2*pow(2,(int)log2(text_size)+1); 
    }
    treeHeight = log2(treeSize);
    //int *segTree_cpu = (int*)malloc(treeSize);
    cudaMalloc(&segTree,(treeSize)*sizeof(int));
    //////      build tree      //////
    for(int i = 0;i<treeHeight; i++){
         int column = pow(2,treeHeight-1-i); //level(0,treeheight-1) from bottom to top
         int upperNode = pow(2,treeHeight-1-i);
         Build<<<column/1024+1,1024>>>(text,segTree,text_size,i,treeHeight,upperNode,column);
    }
    //////      trace node      //////
    int upperNode = pow(2,treeHeight-1);
    Trace<<<text_size/1024+1,1024>>>(text,pos,segTree,upperNode,treeSize);
}

struct equalOne
{
    __host__ __device__
    int operator()(int  value,int ref){
        if(value == 1) {
            return ref;
        }
        else {
            return 0;
        }  
    }
};
struct amount{
     __host__ __device__
     int operator()(int  last ,int pre){
        if(last != 0) {
            printf("hi\n");
            return pre+1;
        }else{
            printf("no\n");
            return pre;
        }
     }
};
struct minusOne{
     __host__ __device__
     int operator()(int  value){
        return value-1 ;
     }
};
struct changeTo1{
     __host__ __device__
     int operator()(int value){
        if(value != 0) {
            return 1;
        }else{
            return 0;
        }
     }
};
int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
    int *buffer_cpu = (int*)malloc(text_size*2);
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);
    thrust::sequence(flag_d,flag_d+text_size,1);
    thrust::transform(pos_d,pos_d+text_size,flag_d,flag_d,equalOne());
    thrust::inclusive_scan(flag_d, flag_d + text_size,cumsum_d,thrust::maximum<int>());
    thrust::transform(flag_d,flag_d+text_size,flag_d,changeTo1());
    nhead = thrust::reduce(flag_d,flag_d+text_size) ;
    thrust::transform(cumsum_d,cumsum_d+text_size,cumsum_d,minusOne());
    thrust::unique_copy(cumsum_d,cumsum_d+text_size,head_d);
    cudaFree(buffer);
	return nhead;

}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
    char* tmpText;
    updatePos<<<text_size/1024+1,1024>>>(text,pos);
    cudaMalloc(&tmpText, sizeof(char)*text_size);
    copyText<<<text_size/1024+1,1024>>>(text,tmpText);
    exchange<<<text_size/1024+1,1024>>>(text,tmpText,pos);
    cudaFree(tmpText);
}
