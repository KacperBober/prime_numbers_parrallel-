
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <stdio.h>
#include <math.h>
#include <chrono>

bool check_parity(unsigned long long *number) {
	return *number & 1;	//oszczêdzone 3 instrukcje w porównaniu do modulo 2
}

bool isPrimaryNumber(unsigned long long *number) {

	bool isPrime = true;

	if (!check_parity(number))
		return false;
	else {
		unsigned long squared_number = sqrt(*number);

		for (unsigned long divider = 3; divider <= squared_number; divider += 2) {
			if (!(*number % divider)) {
				isPrime = false;
				break;
			}
		}
	}
	return isPrime;
}

__global__ void isPrime(bool *is_prime, unsigned long long *number, double max_divider) {

	unsigned long divider = (threadIdx.x + blockIdx.x * blockDim.x) * 2 + 3;

	if (divider <= max_divider) {
		if (!(*number % divider)) {
			*is_prime = false;
		}
	}
}

int main()
{
	unsigned long long number;
	unsigned long long *p_number = &number;

	std::cout << "Podaj liczbe do sprawdzenia: ";
	std::cin >> number;
	
	std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();
	bool is_prime = isPrimaryNumber(p_number);
	std::chrono::steady_clock::time_point endCPU = std::chrono::steady_clock::now();
	int duration = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - beginCPU).count();

	if (is_prime) {
		std::cout << "Sprawdzana liczba " << number << ", jest liczba pierwsza. \nUplynelo " <<
			duration << " milisekund.";
	}
	else {
		std::cout << "Sprawdzana liczba " << number << ", nie jest liczba pierwsza. \nUplynelo " <<
			duration << " milisekund.";
	}
	std::chrono::steady_clock::time_point beginGPU = std::chrono::steady_clock::now();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool is_primeGPU = true;
	bool *p_is_prime = &is_primeGPU;

	double max_divider = sqrt(number);	

	unsigned long total_threads = max_divider / 2;  //check for errors?
	const int threads_per_block = 128;
	
	int blocks_number = (total_threads + threads_per_block - 1) / threads_per_block;


	unsigned long long *d_number;
	int size = sizeof(unsigned long long);

	bool *d_is_prime;

	cudaMalloc((void**)&d_number, size);
	cudaMalloc((void**)&d_is_prime, sizeof(bool));

	cudaMemcpy(d_number, p_number, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_is_prime, p_is_prime, sizeof(bool), cudaMemcpyHostToDevice);

	if (!check_parity(p_number)) {
		is_primeGPU = false;
	}
	else {
		isPrime <<<blocks_number, threads_per_block >>> (d_is_prime, d_number, max_divider);
	}
	cudaMemcpy(p_is_prime, d_is_prime, sizeof(bool), cudaMemcpyDeviceToHost);

	std::chrono::steady_clock::time_point endGPU = std::chrono::steady_clock::now();
	int durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - beginGPU).count();
	if (is_primeGPU) {
		std::cout << "Sprawdzana liczba " << number << ", jest liczba pierwsza. \nUplynelo " <<
			durationGPU << " milisekund.";
	}
	else {
		std::cout << "Sprawdzana liczba " << number << ", nie jest liczba pierwsza. \nUplynelo " <<
			durationGPU << " milisekund.";
	}

    return 0;
}
