
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
	cl_int length = 1024;
	//cl_int length = 1 << 20;

	// Trabajamos con la primera plataforma (driver) en el sistema
	cl_platform_id platform;
	clGetPlatformIDs(1, &platform, NULL);

	// Crea el contexto para una GPU
	cl_context context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

	// Obtiene el dispositivo
	cl_device_id     device;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);

	// Crea la cola de comandos (cola para las kernels)
	cl_command_queue commandQueue;
	commandQueue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

	// Crea los buffer en el host, para los datos de entrada
	cl_int* inputA = (cl_int*)malloc(sizeof(cl_int) * length);
	cl_int* inputB = (cl_int*)malloc(sizeof(cl_int) * length);

	// Inicializa los buffer del host con los valores de entrada
	for (cl_int i = 0; i < length; i++) {
		inputA[i] = i; //0,1,2,...,1023
		inputB[i] = length - i; //1023,1022,...,0
	}

	// Crea los buffer en la GPU, tanto para valores de entrada como para valores de salida
	cl_mem gpuBuffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * length, NULL, NULL);
	cl_mem gpuBuffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * length, NULL, NULL);
	cl_mem gpuBuffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * length, NULL, NULL);

	// Copia los valores de entrada desde los buffer del host a los buffer de la GPU
	clEnqueueWriteBuffer(commandQueue, gpuBuffer_A, CL_TRUE, 0, sizeof(cl_int) * length, inputA, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, gpuBuffer_B, CL_TRUE, 0, sizeof(cl_int) * length, inputB, 0, NULL, NULL);

	// Lee el código fuente de la kernel desde el archivo
	char* source = NULL;
	size_t sourceSize = 0;
	FILE* fp = NULL;
	fopen_s(&fp, "AddArraysKernel.cl", "rb");
	fseek(fp, 0, SEEK_END);
	sourceSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	source = new char[sourceSize];
	fread(source, 1, sourceSize, fp);

	// Crea el programa OpenCL y lo compila, a partir del código fuente de la kernel
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, NULL);
	clBuildProgram(program, 1, &device, "", NULL, NULL);

	// Crea la kernel que se va a ejecutar, a partir del programa ya compilado
	cl_kernel kernel = clCreateKernel(program, "addArrays", NULL);

	// Pasa los argumentos a la kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpuBuffer_A);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpuBuffer_B);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&gpuBuffer_C);

	// Ejecuta la kernel
	size_t localSize = 64; //tamaño del bloque de ejecución
	size_t globalSize = length; //número total de threads a ejecutar
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Recupera el resultado desde la GPU y lo pone en un buffer del host
	cl_int* outputC = (cl_int*)malloc(sizeof(cl_int) * length);
	clEnqueueReadBuffer(commandQueue, gpuBuffer_C, CL_TRUE, 0, sizeof(cl_int) * length, outputC, 0, NULL, NULL);

	// Presenta el resultado (si es menor o igual a 1024)
	if (length <= 1024) {
		for (cl_uint i = 0; i < length; i++) {
			printf("Resultados %d: (%d + %d = %d)\n", i, inputA[i], inputB[i], outputC[i]);
		}
	}
	printf("Listo con talla %d !!!\n", length);

	// Libera los recursos
	delete[] source;
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseMemObject(gpuBuffer_A);
	clReleaseMemObject(gpuBuffer_B);
	clReleaseMemObject(gpuBuffer_C);
	free(inputA);
	free(inputB);
	free(outputC);

	// Fin !!!
	system("pause");
	return 0;
}