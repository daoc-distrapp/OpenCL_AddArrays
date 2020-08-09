
#include <CL/cl.h>
#include <vector>

int main() {
	// Cuántas plataformas (drivers, implementaciones, ...) hay?
	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	printf("Numero de plataformas: %d\n", numPlatforms);

	// Obtiene las IDs de todas las plataformas
	std::vector<cl_platform_id> platforms(numPlatforms);
	clGetPlatformIDs(numPlatforms, &platforms[0], NULL);

	// Obtiene los nombres de todas las plataformas
	for (int i = 0; i < numPlatforms; i++) {
		std::vector<char> platformName(256);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 256, &platformName[i], NULL);
		printf("Indice y Nombre de plataforma: %d | %s\n", i, &platformName[i]);
	}

	// El usuario escoge su plataforma
	puts("Ingrese el indice (+ENTER) de la plataforma que desea: ");
	int plat;
	scanf("%d", &plat);

	// Crea el contexto
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[plat], 0 };
	cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

	// Obtiene el dispositivo
	cl_device_id     device;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);

	// Obtiene la versión de la plataforma
	std::vector<char> tmp_platformVersion(256);
	clGetPlatformInfo(platforms[plat], CL_PLATFORM_VERSION, 256, &tmp_platformVersion[0], NULL);
	float platformVersion = (strstr(&tmp_platformVersion[0], "OpenCL 2.0") != NULL) ? 2.0f : 1.2f;
	printf("Version de plataforma: %s\n", &tmp_platformVersion[0]);

	// Obtiene la versión del dispositivo
	std::vector<char> tmp_deviceVersion(256);
	clGetDeviceInfo(device, CL_DEVICE_VERSION, 256, &tmp_deviceVersion[0], NULL);
	float deviceVersion = (strstr(&tmp_deviceVersion[0], "OpenCL 2.0") != NULL) ? 2.0f : 1.2f;
	printf("Version de dispositivo: %s\n", &tmp_deviceVersion[0]);

	// Obtiene la versión del compilador
	std::vector<char> tmp_compilerVersion(256);
	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 256, &tmp_compilerVersion[0], NULL);
	float compilerVersion = (strstr(&tmp_compilerVersion[0], "OpenCL C 2.0") != NULL) ? 2.0f : 1.2f;
	printf("Version de compilador: %s\n", &tmp_compilerVersion[0]);

	// Crea la cola de comandos (cola para las kernels)
	cl_command_queue commandQueue;
	if (deviceVersion == 2.0f) {
		const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		commandQueue = clCreateCommandQueueWithProperties(context, device, properties, NULL);
	}
	else {
		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
		commandQueue = clCreateCommandQueue(context, device, properties, NULL);
	}

	// Crea los buffer en el host, para los datos de entrada
	// El buffer debe estar alineado con un tamaño de página de 4096, y su tamaño total debe ser múltiplo de 64
	cl_int* inputA = (cl_int*)_aligned_malloc(sizeof(cl_int) * 1024, 4096);
	cl_int* inputB = (cl_int*)_aligned_malloc(sizeof(cl_int) * 1024, 4096);

	// Inicializa los buffer del host con los valores de entrada
	for (cl_uint i = 0; i < 1024; i++) {
		inputA[i] = i; //0,1,2,...,1023
		inputB[i] = 1023 - i; //1023,1022,...,0
	}

	// Crea los buffer en la GPU, tanto para valores de entrada como para valores de salida
	cl_mem gpuBuffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * 1024, NULL, NULL);
	cl_mem gpuBuffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * 1024, NULL, NULL);
	cl_mem gpuBuffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * 1024, NULL, NULL);

	// Copia los valores de entrada desde los buffer del host a los buffer de la GPU
	clEnqueueWriteBuffer(commandQueue, gpuBuffer_A, CL_TRUE, 0, sizeof(cl_int) * 1024, inputA, 0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, gpuBuffer_B, CL_TRUE, 0, sizeof(cl_int) * 1024, inputB, 0, NULL, NULL);

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
	size_t globalSize = 1024; //número total de threads a ejecutar
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Recupera el resultado desde la GPU y lo pone en un buffer del host
	cl_int* outputC = (cl_int*)malloc(sizeof(cl_int) * 1024);
	clEnqueueReadBuffer(commandQueue, gpuBuffer_C, CL_TRUE, 0, sizeof(cl_int) * 1024, outputC, 0, NULL, NULL);

	// Presenta el resultado
	for (cl_uint i = 0; i < 1024; i++) {
		printf("Resultados %d: (%d + %d = %d)\n", i, inputA[i], inputB[i], outputC[i]);
	}

	// Libera los recursos
	delete[] source;
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseMemObject(gpuBuffer_A);
	clReleaseMemObject(gpuBuffer_B);
	clReleaseMemObject(gpuBuffer_C);
	_aligned_free(inputA);
	_aligned_free(inputB);
	free(outputC);

	// Fin !!!
	system("pause");
	return 0;
}