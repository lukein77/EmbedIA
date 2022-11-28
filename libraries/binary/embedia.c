/* 
 * EmbedIA 
 * LIBRERÍA ARDUINO QUE DEFINE FUNCIONES PARA LA IMPLEMENTACIÓN DE REDES NEURNOALES CONVOLUCIONALES
 * EN MICROCONTROLADORES Y LAS ESTRUCTURAS DE DATOS NECESARIAS PARA SU USO
 */

#include "embedia.h"

/* IMPLEMENTACIÓN DE FUNCIONES DE LA LIBRERÍA EmbedIA DEFINIDAS EN embedia.h */


uint8_t sign(float x){
    if(x>=0){
        return 1;
    }else{
        return 0;
    }
}



typedef struct{
	size_t size;
	void  * data;
} raw_buffer;


raw_buffer buffer1 = {0, NULL};
raw_buffer buffer2 = {0, NULL};

void * swap_alloc(size_t s){ 
	static raw_buffer * last_buff = &buffer2;
	last_buff = (last_buff==&buffer1) ? &buffer2 : &buffer1;
	
	if (last_buff->size < s){
		last_buff->data = realloc(last_buff->data, s);
		last_buff->size = s;
	}

	return last_buff->data;
}



int count_set_bits_Brian_Kernighan_algorithm(xBYTE n) {
  register int count = 0;
  while(n) {
    count++;
    n = n & (n-1);
  }

  return count;
  //return __builtin_popcount(n);
}

dato_salida POPCOUNT(xBYTE n){

    return 2*count_set_bits_Brian_Kernighan_algorithm(n) - tamano_del_bloque;

}



xBYTE XNOR(register xBYTE a,register xBYTE b){

    return ~(a^b);

}





void quant_conv2d_input_not_binary(quant_filter_t filter, data_t input, data_t * output, uint32_t delta){

    uint32_t i,j,k,l,c;
	dato_salida suma;
	register uint32_t offset_absoluto;

	for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			suma = 0;
			for (c=0; c<filter.channels; c++){
				for (k=0; k<filter.kernel_size; k++){
					for (l=0; l<filter.kernel_size; l++){
                        offset_absoluto = (c*filter.kernel_size*filter.kernel_size)+k*filter.kernel_size+l;
                        if((filter.bitarray[offset_absoluto / tamano_del_bloque] & mascara_global_bits[offset_absoluto % tamano_del_bloque])>>(tamano_del_bloque-(offset_absoluto % tamano_del_bloque)-1)){
                            suma += (input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)]);
                        }else{
                            suma += ((-1) * input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)]);
                        }
					}
				}
			}
			output->data[delta + i*output->width + j] = suma + filter.bias;
		}
	}


}


void quant_conv2d_input_not_binary_layer(quant_conv_layer_t layer, data_t input, data_t *output){

    uint32_t delta;

	output->channels = layer.n_filters; //cantidad de filtros
	output->height   = input.height - layer.filters[0].kernel_size + 1;
	output->width    = input.width - layer.filters[0].kernel_size + 1;
	output->data     = (dato_salida*)swap_alloc( sizeof(dato_salida)*output->channels*output->height*output->width );

	for(uint32_t i=0; i<layer.n_filters; i++){
		delta = i*(output->height)*(output->width);
        quant_conv2d_input_not_binary(layer.filters[i],input,output,delta);
	}


}



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
void quant_conv2d(quant_conv_layer_t layer, data_t input, data_t *output){   //sirve para entrada bin

    uint32_t i,j,k,l,c,f;

    output->channels = layer.n_filters; //cantidad de filtros
	output->height   = input.height - layer.filters[0].kernel_size + 1;
	output->width    = input.width - layer.filters[0].kernel_size + 1;
	output->data     = (dato_salida*)swap_alloc( sizeof(dato_salida)*(output->channels*output->height*output->width) );

    register uint32_t cont;
    xBYTE suma;
    size_t output_hw = output->height*output->width;
    size_t iterador;
    uint8_t long_ult_bloque = (layer.filters[0].kernel_size*layer.filters[0].channels*layer.filters[0].kernel_size) % tamano_del_bloque;

    for(iterador=0;iterador<output->channels*output_hw;iterador++){ //valores a cero
        output->data[iterador] = 0;
    }

    for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			cont = 0;
			suma = 0;
			for (c=0; c<layer.filters[0].channels; c++){
				for (k=0; k<layer.filters[0].kernel_size; k++){
					for (l=0; l<layer.filters[0].kernel_size; l++){

						if(sign(input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)])){
                            suma+= mascara_global_bits[cont%tamano_del_bloque];
						}
						if(((++cont)%tamano_del_bloque)==0){
                            //ya tengo el num
                            for(f=0;f<layer.n_filters;f++){
                                output->data[(f*output_hw) + i*output->width + j] += POPCOUNT(XNOR(suma,layer.filters[f].bitarray[(cont/tamano_del_bloque)-1]));
                            }
                            suma = 0;
						}
					}
				}
				//termino un canal
			}
			//ya tenemos una fila de la matriz de salida
            if(long_ult_bloque!=0){ //bloque extra
                for(f=0;f<layer.n_filters;f++){
                    output->data[(f*output_hw) + i*output->width + j] += POPCOUNT(XNOR(suma,layer.filters[f].bitarray[cont/tamano_del_bloque])) - (tamano_del_bloque-long_ult_bloque) + layer.filters[f].bias;
                }
            }else{  //solo bias
                for(f=0;f<layer.n_filters;f++){
                    output->data[(f*output_hw) + i*output->width + j] += layer.filters[f].bias;
                }
            }
            //ahora se corre el kernel
		}
	}
}


void quant_dense_forward(quant_dense_layer_t dense_layer, flatten_data_t input, flatten_data_t * output){

    uint32_t bloques_enteros = input.length / tamano_del_bloque;
    uint8_t long_ult = input.length % tamano_del_bloque;
    output->length = dense_layer.n_neurons;
	output->data = (dato_salida*)swap_alloc(sizeof(dato_salida)*dense_layer.n_neurons);

    uint32_t i;
    register uint32_t cont;
    uint8_t j;
    xBYTE tot=0;

    size_t iterador;
    for(iterador=0;iterador<output->length;iterador++){ //valores a cero
        output->data[iterador] = 0;
    }

    for(i=0;i<bloques_enteros;i++){
        for(j=0;j<tamano_del_bloque;j++){
            if(sign(input.data[j+(tamano_del_bloque*i)])){

                tot += mascara_global_bits[j];
            }
        }
        for(cont = 0;cont<dense_layer.n_neurons;cont++){
            output->data[cont] += POPCOUNT(XNOR(dense_layer.neurons[cont].weights[i],tot));
        }
        tot = 0;
    }
	if(long_ult!=0){
        for(j=0;j<long_ult;j++){
            if(sign(input.data[j])){
                tot += mascara_global_bits[j];
            }
        }
        for(cont = 0;cont<dense_layer.n_neurons;cont++){
            output->data[cont] += POPCOUNT(XNOR(dense_layer.neurons[cont].weights[bloques_enteros],tot)) - (tamano_del_bloque-long_ult) + dense_layer.neurons[cont].bias;
        }
    }else{  //solo bias
        for(cont=0;cont<dense_layer.n_neurons;cont++){
            output->data[cont] += dense_layer.neurons[cont].bias;
        }
    }

}





/* 
 * conv2d()
 * Función que realiza la convolución entre un filtro y un conjunto de datos.
 * Parámetros:
 *             filter_t filter  =>  estructura filtro con pesos cargados
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 *                   uint32_t delta =>  posicionamiento de feature_map dentro de output.data
 */
void conv2d(filter_t filter, data_t input, data_t * output, uint32_t delta){
	uint32_t i,j,k,l,c;
	float suma;

	for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			suma = 0;
			for (c=0; c<filter.channels; c++){
				for (k=0; k<filter.kernel_size; k++){
					for (l=0; l<filter.kernel_size; l++){
						suma += (filter.weights[(c*filter.kernel_size*filter.kernel_size)+k*filter.kernel_size+l] * input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)]);
					}
				}
			}
			output->data[delta + i*output->width + j] = suma + filter.bias;
		}
	}

}

/* 
 * conv2d_layer()
 * Función que se encarga de aplicar la convolución de una capa de filtros (conv_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          conv_layer_t layer  =>  capa convolucional con filtros cargados
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void conv2d_layer(conv_layer_t layer, data_t input, data_t * output){
	uint32_t delta;

	output->channels = layer.n_filters; //cantidad de filtros
	output->height   = input.height - layer.filters[0].kernel_size + 1;
	output->width    = input.width - layer.filters[0].kernel_size + 1;
	output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

	for(uint32_t i=0; i<layer.n_filters; i++){
		delta = i*(output->height)*(output->width);
		conv2d(layer.filters[i],input,output,delta);
	}

}

void depthwise(filter_t filter, data_t input, data_t * output){
	uint32_t i,j,k,l,c;
	float suma;

	for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			for (c=0; c<filter.channels; c++){
				suma=0;
				for (k=0; k<filter.kernel_size; k++){
					for (l=0; l<filter.kernel_size; l++){
						suma += (filter.weights[(c*filter.kernel_size*filter.kernel_size)+k*filter.kernel_size+l] * input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)]);
					}
				}
				output->data[c*output->width*output->height + i*output->width + j] = suma;
			}
		}
	}
}
void pointwise(filter_t filter, data_t input, data_t * output, uint32_t delta){
	uint32_t i,j,c;
	float suma;

	for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			suma = 0;
			for (c=0; c<filter.channels; c++){
				suma += (filter.weights[c] * input.data[(c*input.height*input.width)+i*input.width+j]);
			}
			output->data[delta + i*output->width + j] = suma + filter.bias;
		}
	}
}

/* 
 * separable_conv2d_layer()
 * Función que se encarga de aplicar la convolución de una capa de filtros (conv_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          conv_layer_t layer  =>  capa convolucional con filtros cargados
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void separable_conv2d_layer(separable_layer_t layer, data_t input, data_t * output){
	uint32_t delta;
	data_t depth_output;

	depth_output.channels = input.channels; //cantidad de canales
	depth_output.height   = input.height - layer.depth_filter.kernel_size + 1;
	depth_output.width    = input.width - layer.depth_filter.kernel_size + 1;
	depth_output.data     = (float*)swap_alloc( sizeof(float)*depth_output.channels*depth_output.height*depth_output.width );

	depthwise(layer.depth_filter,input,&depth_output);

	output->channels = layer.n_filters; //cantidad de filtros
	output->height   = depth_output.height;
	output->width    = depth_output.width;
	output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );
	
	for(uint32_t i=0; i<layer.n_filters; i++){
		delta = i*(output->height)*(output->width);
		pointwise(layer.point_filters[i],depth_output,output,delta);
	}

}

/* 
 * neuron_forward()
 * Función que realiza el forward de una neurona ante determinado conjunto de datos de entrada.
 * Parámetros:
 *             neuron_t neuron  =>  neurona con sus pesos y bias cargados
 *        flatten_data_t input  =>  datos de entrada en forma de vector (flatten_data_t)
 * Retorna:
 *                      float  =>  resultado de la operación             
 */
float neuron_forward(neuron_t neuron, flatten_data_t input){
	uint32_t i;
	float result = 0;

	for(i=0;i<input.length;i++){
		result += input.data[i]*neuron.weights[i];
	}

	return result + neuron.bias;
}

/* 
 * dense_forward()
 * Función que se encarga de realizar el forward de una capa densa (dense_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros
 *          dense_layer_t dense_layer  =>  capa densa con neuronas cargadas
 *               flatten_data_t input  =>  datos de entrada de tipo flatten_data_t
 *            flatten_data_t * output  =>  puntero a la estructura flatten_data_t donde se guardará el resultado
 */
void dense_forward(dense_layer_t dense_layer, flatten_data_t input, flatten_data_t * output){
	uint32_t i;

	output->length = dense_layer.n_neurons;
	output->data = (float*)swap_alloc(sizeof(float)*dense_layer.n_neurons);
	
	for(i=0;i<dense_layer.n_neurons;i++){
		output->data[i] = neuron_forward(dense_layer.neurons[i],input);
	}
}

/* 
 * max_pooling2d()
 * Función que se encargará de aplicar un max pooling a una entrada
 * con un tamaño de ventana de recibido por parámetro (uint32_t strides)
 * a un determinado conjunto de datos de entrada.
 * Parámetros:
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
// void max_pooling_2d(uint32_t strides, data_t input, data_t* output){
void max_pooling_2d(uint32_t pool_size, uint32_t strides, data_t input, data_t* output){
	uint32_t c,i,j,aux1,aux2;
	float max = -INFINITY;
	float num;

	// output->height = (input.height)/pool_size ;
	// output->width =  (input.width )/pool_size ;
	output->height = ((uint32_t) ((input.height - pool_size)/strides)) + 1;
	output->width  = ((uint32_t) ((input.width - pool_size)/strides)) + 1;
	output->channels = input.channels;
	output->data = (float*)swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

	for (c=0; c<output->channels; c++){
		for (i=0; i<output->height; i++){
			for (j=0; j<output->width; j++){

				max = -INFINITY;

				for(aux1=0; aux1<pool_size; aux1++){
					for(aux2=0; aux2<pool_size; aux2++){

						num = input.data[c*input.width*input.height + (i*strides + aux1)*input.width + j*strides + aux2];
						
						if(num>max){
							max = num;
						}
					}
				}

				output->data[c*output->width*output->height + i*output->width + j] = max;
			}
		}	
	}
}

/* 
 * avg_pooling_2d()
 * Función que se encargará de aplicar un average pooling a una entrada
 * con un tamaño de ventana de recibido por parámetro (uint32_t strides)
 * a un determinado conjunto de datos de entrada.
 * Parámetros:
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void avg_pooling_2d(uint32_t pool_size, uint32_t strides, data_t input, data_t* output){
	uint32_t c,i,j,aux1,aux2;
	uint32_t cant = pool_size*pool_size;
	float avg = 0;
	float num;

	// output->height = (input.height)/strides ;
	// output->width =  (input.width )/strides ;
	output->height = ((uint32_t) ((input.height - pool_size)/strides)) + 1;
	output->width  = ((uint32_t) ((input.width - pool_size)/strides)) + 1;
	output->channels = input.channels;
	output->data = (float*)swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

	for (c=0; c<output->channels; c++){
		for (i=0; i<output->height; i++){
			for (j=0; j<output->width; j++){

				avg = 0;

				for(aux1=0; aux1<pool_size; aux1++){
					for(aux2=0; aux2<pool_size; aux2++){	
						num = input.data[c*input.width*input.height + (i*strides + aux1)*input.width + j*strides + aux2];
						avg += num;
					}
				}

				output->data[c*output->width*output->height + i*output->width + j] = avg/cant;
			}
		}	
	}
}

/* 
 * softmax()
 * Función que implementa la activación softmax.
 * Convierte un vector de entrada en na distribución de probabilidades
 * (aplicado a la salida de regresión lineal para obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *         flatten_data_t data  =>  datos de entrada de tipo flatten_data_t
 */
void softmax(flatten_data_t data){
	float m = -INFINITY;
	for (size_t i = 0; i < data.length; i++) {
		if (data.data[i] > m) {
			m = data.data[i];
		}
	}

	float sum = (0.0);
	for (size_t i = 0; i < data.length; i++) {
		sum += exp(data.data[i] - m);
	}

	float offset = m + log(sum);
	for (size_t i = 0; i < data.length; i++) {
		data.data[i] = exp(data.data[i] - offset);
	}
}

/* 
 * relu(data_t)
 * Función que implementa la activación relu, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *                 data_t data  =>  datos de tipo data_t a modificar
 */
void relu(data_t data){
	uint32_t i;
	uint32_t length = data.channels*data.height*data.width;

	for (i=0;i<length;i++){
		data.data[i] = data.data[i] < 0 ? 0 : data.data[i];
	}
}

/* 
 * relu_flatten(flatten_data_t)
 * Función que implementa la activación relu, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *         flatten_data_t data  =>  datos de tipo flatten_data_t a modificar
 */
void relu_flatten(flatten_data_t data){
	uint32_t i;

	for (i=0;i<(data.length);i++){
		data.data[i] = data.data[i] < 0 ? 0 : data.data[i];
	}
}

/* 
 * tanh(data_t)
 * Función que implementa la activación tanh, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *                 data_t data  =>  datos de tipo data_t a modificar
 */
void tanh2d(data_t data){
	uint32_t i;
	uint32_t length = data.channels*data.height*data.width;

	for (i=0;i<length;i++){
		// data.data[i] = tanh(data.data[i]); 
		data.data[i] = 2/(1+exp(-2*data.data[i])) - 1;
	}
}

/* 
 * tanh_flatten(flatten_data_t)
 * Función que implementa la activación tanh, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *         flatten_data_t data  =>  datos de tipo flatten_data_t a modificar
 */
void tanh_flatten(flatten_data_t data){
	uint32_t i;

	for (i=0;i<(data.length);i++){
		// data.data[i] = tanh(data.data[i]); 
		data.data[i] = 2/(1+exp(-2*data.data[i])) - 1;
	}
}


/* 
 * flatten_layer()
 * Realiza un cambio de tipo de variable. 
 * Convierte el formato de los datos en formato de matriz data_t en vector flatten_data_t.
 * (prepara los datos para ingresar en una capa de tipo dense_layer_t)
 * Parámetros:
 *                data_t input  =>  datos de entrada de tipo data_t
 *     flatten_data_t * output  =>  puntero a la estructura flatten_data_t donde se guardará el resultado
 */
void flatten_layer(data_t input, flatten_data_t * output){
	uint32_t c,i,j;
	uint32_t cantidad = 0;

	output->length = input.channels * input.height * input.width;
	output->data = (float*)swap_alloc(sizeof(float)*output->length);

	for(i=0;i<input.height;i++){
		for(j=0;j<input.width;j++){
			for(c=0;c<input.channels;c++){
				output->data[cantidad++] = input.data[(c*input.width*input.height)+(i*input.width)+j];
			}
		}
	}
}

/* 
 * argmax()
 * Busca el indice del valor mayor dentro de un vector de datos (flatten_data_t)
 * Parámetros:
 *         flatten_data_t data  =>  datos de tipo flatten_data_t a buscar máximo
 * Retorna
 *                         uint32_t  =>  resultado de la búsqueda - indice del valor máximo
 */
uint32_t argmax(flatten_data_t data){
	float max = data.data[0];
	uint32_t pos = 0;

	for(uint32_t i=1;i<data.length;i++){
		if(data.data[i]>max){
			max = data.data[i];
			pos = i;
		} 
	}
	
	return pos;
}



/*
 * batch_normalization()
 * Normaliza la salida de una capa anterior
 * Parámetros:
 *      batchnorm_layer_t layer =>  capa BatchNormalization con sus respectivos parámetros
 *      data_t *input           =>  datos de entrada de tipo data_t
 * 		data_t *output			=>	puntero a la estructura data_t donde se guardará el resultado
 */

void batch_normalization(batchnorm_layer_t layer, data_t input, data_t *output) {
	uint32_t i, j;
	uint32_t length = (input.height)*(input.width);
	uint32_t ilen;

	output->height = input.height;
	output->width  = input.width;
	output->channels = input.channels;
	output->data = (float*) swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

	for (i = 0; i < output->channels; i++) {
		ilen = i*length;
		for (j = 0; j < length; j++) {
			output->data[ilen+j] = input.data[ilen+j] * layer.standard_gamma[i] + layer.standard_beta[i];
		}
	}
}

/*
 * batch_normalization_flatten()
 * Normaliza la salida proveniente de una capa densa
 * Parámetros:
 *      batchnorm_layer_t layer =>  capa BatchNormalization con sus respectivos parámetros
 *      flatten_data_t *input   =>  datos de entrada de tipo flatten_data_t
 * 		flatten_data_t *output	=>	puntero a la estructura flatten_data_t donde se guardará el resultado
 */

void batch_normalization_flatten(batchnorm_layer_t layer, flatten_data_t input, flatten_data_t *output) {
	uint32_t i;

	output->length = input.length;
	output->data = (float*) swap_alloc(sizeof(float)*output->length);

	for (i = 0; i < output->length; i++) {
		output->data[i] = input.data[i] * layer.standard_gamma[i] + layer.standard_beta[i];
	}
}

