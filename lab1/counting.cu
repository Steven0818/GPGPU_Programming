#include "counting.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

using namespace std;
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__global__ void bulid(const char *text,int *segTree,int text_size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x
    if(idx<text_size && text[idx] != '\n'){
        segTree[id] = text[idx]
    }
}
 
struct Node {
    int value;   
};
int LC(int i) {return i*2;}
int RC(int i) {return i*2+1;}

 
 
void build(int L, int R, int i,Node* node,const char *text)
{

    if (L == R)
    {
        // 設定樹葉的數值
    
        if(text[L]!='\n'){
        	node[i].value = 1;
        }
        else{
        	node[i].value=0;
        }
        return;
    }
 
    int M = (L + R) / 2;
    build(L  , M, LC(i),node,text); 
    build(M+1, R, RC(i),node,text);
    if(node[LC(i)].value!=0&&node[RC(i)].value!=0){   
       	node[i].value = node[LC(i)].value + node[RC(i)].value;
    }
    else{
    	node[i].value = 0;
    }
}

void CountPosition(const char *text, int *pos, int text_size)
{
    int* segTree;
    int treeSize;
    int treeHeight;
    if(text_size%2 == 0){
        treeSize = 2*text_size-1;
    }
    else{
        treeSize = 2*text_size;
    }

    for(int i = 0;i < )
    cudaMalloc(&segTree,(treeSize)*sizeof(int));

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
