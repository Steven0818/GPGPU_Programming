#include "counting.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
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
    cout<<"treeHeight"<<treeHeight<<endl;
    cout<<"textSize"<<text_size<<endl;
    cout<<"treeSize"<<treeSize<<endl;
    //int *segTree_cpu = (int*)malloc(treeSize);
    cudaMalloc(&segTree,(treeSize)*sizeof(int));
    //////      build tree      //////
    for(int i = 0;i<treeHeight; i++){
         int column = pow(2,treeHeight-1-i); //level(0,treeheight-1) from bottom to top
         cout<<"column"<<column<<endl;
         int upperNode = pow(2,treeHeight-1-i);
         Build<<<column/256+1,256>>>(text,segTree,text_size,i,treeHeight,upperNode,column);
    }
    /*cudaMemcpy(segTree_cpu,segTree,(treeSize)*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=1;i<treeSize;i++){
        printf("data%d %d \n",i,segTree_cpu[i]);
    }*/
    //////      trace node      //////
    int upperNode = pow(2,treeHeight-1);
    Trace<<<text_size/256+1,256>>>(text,pos,segTree,upperNode,treeSize);
}




int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{}
