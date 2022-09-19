import numpy as np
from tensorflow import keras
from math import sqrt
import larq as lq

from embedia.project_options import ModelDataType, ProjectType, ProjectFiles, DebugMode, BinaryBlockSize

mascara_global_bits_8 = [ 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1]
mascara_global_bits_16  = [ 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1]
mascara_global_bits_32 = [ 2147483648 , 1073741824 , 536870912 , 268435456 , 134217728 , 67108864 , 33554432 , 16777216 , 8388608 , 4194304 , 2097152 , 1048576 , 524288 , 262144 , 131072 , 65536 , 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1]


def convertir_pesos(weights):
  '''
    Used internally to transpose weights 4 dimentions array. Our library works with weights with the form (filter,channel,row,column)
    It goes from (row,column,channel,filter) to (filter,channel,row,column)
    For example: It goes from (3,3,1,8) to (8,1,3,3)
    Receives: weights from keras/tf model (model.get_weights return)
    Returns: weights our library can work with
  '''
  
  _fila,_col,_can,_filt = weights.shape
  arr = np.zeros((_filt,_can,_fila,_col))
  for fila,elem in enumerate(weights):
    for columna,elem2 in enumerate(elem):
      for canal,elem3 in enumerate(elem2):
        for filtros,valor in enumerate(elem3):
          #print("F:{0}, C:{1}, Canal:{2}, Filtro:{3} -> Valor: {4}".format(fila,columna,canal,filtro,valor))
          arr[filtros,canal,fila,columna] = valor   
  return arr
  



def get_weights_separable(layer):
  '''
    Function to return model weights from cnn layer in a way our model can work with
    Params:
    layer -> convolutional layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (filters,channels,rows,columns)
    biases -> array with dimention: (filters)

    example of usage: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'separable_conv2d' in layer.name #asserting I didn't receive a non convolutional layer
  depth_weights=layer.get_weights()[0]
  point_weights=layer.get_weights()[1]
  biases=layer.get_weights()[2]
  
  depth_weights=convertir_pesos(depth_weights)
  point_weights=convertir_pesos(point_weights)
  
  return depth_weights,point_weights,biases


def get_weights_cnn(layer):
  '''
    Function to return model weights from cnn layer in a way our model can work with
    Params:
    layer -> convolutional layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (filters,channels,rows,columns)
    biases -> array with dimention: (filters)

    example of usage: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'conv2d' in layer.name #asserting I didn't receive a non convolutional layer
  weights=layer.get_weights()[0]
  biases=layer.get_weights()[1]
  weights=convertir_pesos(weights)
  return weights,biases

#NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
def get_weights_bnn(layer):
  '''
    Function to return model weights from cnn layer in a way our model can work with
    Params:
    layer -> convolutional layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (filters,channels,rows,columns)
    biases -> array with dimention: (filters)

    example of usage: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'quant_conv2d' in layer.name #asserting I didn't receive a non convolutional layer
  with lq.context.quantized_scope(True):
      weights=layer.get_weights()[0]
      biases=layer.get_weights()[1]
  
  weights=convertir_pesos(weights)
  return weights,biases



def exportar_separable_a_c(layer,nro, macro_converter, data_type):
  '''
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model separable weights
  '''
  
  depth_weights,point_weights,biases=get_weights_separable(layer)


  depth_filtros,depth_canales,depth_filas,depth_columnas=depth_weights.shape #Getting layer info from it's weights
  assert depth_filas==depth_columnas #WORKING WITH SQUARE KERNELS FOR NOW
  depth_kernel_size=depth_filas #Defining kernel size

  point_filtros,point_canales,_,_=point_weights.shape #Getting layer info from it's weights

  ret=""

  init_conv_layer=f'''

separable_layer_t init_separable_layer{nro}(void){{

  '''
  o_weights=""
  for ch in range(depth_canales):
    for f in range(depth_filas):
      o_weights+='\n    '
      for c in range(depth_columnas):
        o_weights+=f'''{macro_converter(depth_weights[0,ch,f,c])}, '''
      #o_weights+='\n'
    o_weights+='\n  '

  o_code=f'''
  static const {data_type} depth_weights[]={{{o_weights}
  }};
  static filter_t depth_filter = {{{depth_canales}, {depth_kernel_size}, depth_weights, 0}};

  static filter_t point_filters[{point_filtros}];
  '''
  init_conv_layer+=o_code
  
  for i in range(point_filtros):
    o_weights=""
    for ch in range(point_canales):
      o_weights+=f'''{macro_converter(point_weights[i,ch,0,0])},'''
    
    o_code=f'''
  static const {data_type} point_weights{i}[]={{{o_weights}
  }};
  static filter_t point_filter{i} = {{{point_canales}, 1, point_weights{i}, {macro_converter(biases[i])}}};
  point_filters[{i}]=point_filter{i};
  '''
    init_conv_layer+=o_code
  
  init_conv_layer+=f'''
  separable_layer_t layer = {{{point_filtros},depth_filter,point_filters}};
  return layer;
}}
  '''

  ret+=init_conv_layer

  return ret


def exportar_cnn_a_c(layer,nro, macro_converter, data_type):
  '''
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model cnn weights
  '''
  
  pesos,biases=get_weights_cnn(layer)


  filtros,canales,filas,columnas=pesos.shape #Getting layer info from it's weights

  assert filas==columnas #WORKING WITH SQUARE KERNELS FOR NOW
  kernel_size=filas #Defining kernel size

  ret=""

  init_conv_layer=f'''

conv_layer_t init_conv_layer{nro}(void){{

  static filter_t filtros[{filtros}];
  '''
  for i in range(filtros):
    o_weights=""
    for ch in range(canales):
      for f in range(filas):
        o_weights+='\n    '
        for c in range(columnas):
          o_weights+=f'''{macro_converter(pesos[i,ch,f,c])},'''
        #o_weights+='\n'
      o_weights+='\n  '
 
    o_code=f'''
  static const {data_type} weights{i}[]={{{o_weights}
  }};
  static filter_t filter{i} = {{{canales}, {kernel_size}, weights{i}, {macro_converter(biases[i])}}};
  filtros[{i}]=filter{i};
    '''
    init_conv_layer+=o_code
  init_conv_layer+=f'''
  conv_layer_t layer = {{{filtros},filtros}};
  return layer;
}}
  '''

  ret+=init_conv_layer

  return ret


# NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
def exportar_bnn_a_c(layer,nro, macro_converter, data_type,block_type,xBits,input_bin):
  '''
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons
    input_bin      --> 1 si es full binaria

    Returns:
    String with c code representing the function with model cnn weights
  '''
  
  pesos,biases=get_weights_bnn(layer)


  filtros,canales,filas,columnas=pesos.shape #Getting layer info from it's weights

  assert filas==columnas #WORKING WITH SQUARE KERNELS FOR NOW
  kernel_size=filas #Defining kernel size
  
  largo_total = (columnas)*(canales)*(filas)
  
  

  ret=""

  init_conv_layer=f'''

quant_conv_layer_t init_conv_binary_{input_bin}_layer{nro}(void){{

  static quant_filter_t filtros_b[{filtros}];
  '''
  for i in range(filtros):
    cont = 0
    suma = 0
    o_weights=""
    for ch in range(canales):
      for f in range(filas):
        for c in range(columnas):
            num = pesos[i,ch,f,c]
            if xBits==16:
                if num == 1.0:  
                    suma += mascara_global_bits_16[cont]
            elif xBits==32:
                if num == 1.0: 
                    suma += mascara_global_bits_32[cont]
            else:
                if num == 1.0: 
                    suma += mascara_global_bits_8[cont]
            
            if cont == xBits-1 or ((ch+1)*(f+1)*(c+1) == largo_total):
                
                o_weights+=f'''{macro_converter(suma)},'''
                cont = 0
                suma = 0
                
            else:
                cont+=1
    
    o_weights=o_weights[:-1] #remuevo la ultima coma
    o_code=f'''
  static const {block_type} weights{i}[]={{{o_weights}
  }};
  static quant_filter_t filter{i} = {{{canales}, {kernel_size}, weights{i}, {macro_converter(biases[i])}}};
  filtros_b[{i}]=filter{i};
    '''
    init_conv_layer+=o_code
  init_conv_layer+=f'''
  quant_conv_layer_t layer = {{{filtros},filtros_b}};
  return layer;
}}
  '''

  ret+=init_conv_layer

  return ret





def get_weights_dense(layer):
  '''
    Function to return model weights from dense layer in a way our model can work with
    Params:
    layer -> dense layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (input,neurons)
    biases -> array with dimention: (filters)

    Example: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'dense' in layer.name #Get sure it is a dense layer
  weights=layer.get_weights()[0]
  biases=layer.get_weights()[1]
  return weights,biases


def get_weights_dense_binary(layer):
  '''
    Function to return model weights from dense layer in a way our model can work with
    Params:
    layer -> dense layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (input,neurons)
    biases -> array with dimention: (filters)

    Example: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'quant_dense' in layer.name #Get sure it is a dense layer
  with lq.context.quantized_scope(True):
      weights=layer.get_weights()[0]
      biases=layer.get_weights()[1]
  return weights,biases



def exportar_densa_binaria_a_c(layer, nro, macro_converter, data_type,block_type,xBits):
  '''
    Builds embedia's init_dense_layer function
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model dense weights
  '''
  pesos,biases=get_weights_dense_binary(layer)
  input,neuronas=pesos.shape
  toti = pesos[:,0].size
  suma = 0 
  cont = 0
  cont2 = 0
  ret=""

  init_dense_layer=f'''
quant_dense_layer_t init_dense_binary_layer{nro}(){{
  // Cantidad de variables weights = numero de neuronas
  // Cantidad de pesos por weights = numero de entradas

  static quant_neuron_t neuronas_b[{neuronas}];
  '''
  o_code=""
  for neurona in range(neuronas):
    suma = 0
    cont = 0
    cont2 = 0
    o_weights=""
    #for p in pesos[neurona,:]:
    for p in pesos[:,neurona]:
        cont2+=1
        if xBits==16:
            if p == 1.0:  
                suma += mascara_global_bits_16[cont]
        elif xBits==32:
            if p == 1.0: 
                suma += mascara_global_bits_32[cont]
        else:
            if p == 1.0: 
                suma += mascara_global_bits_8[cont]
        
        if cont == xBits-1 or (cont2 == toti):
            
            o_weights+=f'''{macro_converter(suma)},'''
            cont = 0
            suma = 0
            
        else:
            cont+=1
      
    o_weights=o_weights[:-1] #remuevo la ultima coma
   
    o_code+=f'''
  static const {block_type} weights{neurona}[]={{
    {o_weights}
  }};
  static quant_neuron_t neuron{neurona} = {{weights{neurona}, {macro_converter(biases[neurona])}}};
  neuronas_b[{neurona}]=neuron{neurona};
    '''
  init_dense_layer+=o_code

  init_dense_layer+=f'''
  quant_dense_layer_t layer= {{{neuronas}, neuronas_b}};
  return layer;
}}
'''
  return init_dense_layer



def exportar_densa_a_c(layer,nro, macro_converter, data_type):
  '''
    Builds embedia's init_dense_layer function
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model dense weights
  '''
  pesos,biases=get_weights_dense(layer)
  input,neuronas=pesos.shape
  ret=""

  init_dense_layer=f'''
dense_layer_t init_dense_layer{nro}(){{
  // Cantidad de variables weights = numero de neuronas
  // Cantidad de pesos por weights = numero de entradas

  static neuron_t neuronas[{neuronas}];
  '''
  o_code=""
  for neurona in range(neuronas):
    o_weights=""
    #for p in pesos[neurona,:]:
    for p in pesos[:,neurona]:
      o_weights+=f'''{macro_converter(p)},'''
    o_weights=o_weights[:-1] #remuevo la ultima coma
    #o_weights+='\n'
    o_code+=f'''
  static const {data_type} weights{neurona}[]={{
    {o_weights}
  }};
  static neuron_t neuron{neurona} = {{weights{neurona}, {macro_converter(biases[neurona])}}};
  neuronas[{neurona}]=neuron{neurona};
    '''
  init_dense_layer+=o_code

  init_dense_layer+=f'''
  dense_layer_t layer= {{{neuronas}, neuronas}};
  return layer;
}}
'''
  return init_dense_layer




def get_parameters_batchnorm(layer):
  '''
    Function to return model parameters from BatchNormalization layer in a way our model can work with
    Params:
    layer -> BatchNormalization layer
    
    Returns:
    Tuple with values: gamma, beta, moving_mean, moving_variance
    
  '''
  assert 'batch_normalization' in layer.name #Get sure it is a batchnorm layer
  gamma = layer.get_weights()[0]
  beta = layer.get_weights()[1]
  moving_mean = layer.get_weights()[2]
  moving_variance = layer.get_weights()[3]
  epsilon = layer.epsilon

  '''Calculate a new parameter (we'll call it gamma_variance)
    This way we don't need to do division and calculate square root in the microcontroller
    epsilon = 0.001 (to avoid division by zero)'''
  gamma_variance = np.array([(gamma[i] / sqrt(moving_variance[i] + epsilon)) for i in range(gamma.size)])

  return gamma,beta,moving_mean,moving_variance,gamma_variance

def exportar_batchnorm_a_c(layer, nro, macro_converter, data_type):
  '''
    Builds embedia's init_batchnorm_layer function
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function 
  '''
  gamma,beta,mean,variance,gamma_variance=get_parameters_batchnorm(layer)
  ret=""

  init_batchnorm_layer=f'''
batchnorm_layer_t init_batchnorm_layer{nro}(void){{
  '''
  o_gamma = ""
  for i in range(gamma.size):
    o_gamma += f'''{macro_converter(gamma[i])}, '''

  o_beta = ""
  for i in range(beta.size):
    o_beta += f'''{macro_converter(beta[i])}, '''

  o_mean = ""
  for i in range(mean.size):
    o_mean += f'''{macro_converter(mean[i])}, '''

  o_variance = ""
  for i in range(variance.size):
    o_variance += f'''{macro_converter(variance[i])}, '''

  o_gamma_variance = ""
  for i in range(gamma_variance.size):
    o_gamma_variance += f'''{macro_converter(gamma_variance[i])}, '''

  init_batchnorm_layer += f'''

  static const {data_type} beta[] = {{ {o_beta} 
  }};

  static const {data_type} moving_mean[] = {{ {o_mean} 
  }};

  static const {data_type} gamma_variance[] = {{ {o_gamma_variance} 
  }};

  /* gamma and moving_variance parameters from keras layer:
  static const {data_type} gamma[] = {{ {o_gamma} 
  }};

  static const {data_type} moving_variance[] = {{ {o_variance} 
  }};
  */

  batchnorm_layer_t layer;
  layer.moving_mean = moving_mean;
  layer.beta = beta;
  layer.gamma_variance = gamma_variance; 

  return layer;
}}
  '''
  ret += init_batchnorm_layer
  return ret





#=================================

#CREATE MODEL INIT FUNCTION
def create_model_init(cantConv,cantDensas,cantSeparable,cantConvBinary,cantDensasBinary,cantConvBinaryInputNotBinary,cantBatchNorm):
  #Begin model_init function string
  model_init = f'''
void model_init(){{
'''
  for i in range(cantSeparable):
    model_init+=f'''    separable_layer{i} = init_separable_layer{i}(); // Capa depthwise separable conv {i+1}\n'''

  for i in range(cantConv):
    model_init+=f'''    conv_layer{i} = init_conv_layer{i}(); // Capa convolucional {i+1}\n'''

  for i in range(cantDensas):
    model_init+=f'''    dense_layer{i} = init_dense_layer{i}(); //Capa densa {i+1}\n'''
  for i in range(cantConvBinary):
    model_init+=f'''    conv_binary_layer{i} = init_conv_binary_1_layer{i}(); //Capa conv binary {i+1}\n'''
  for i in range(cantDensasBinary):
    model_init+=f'''    dense_binary_layer{i} = init_dense_binary_layer{i}(); //Capa densa binaria{i+1}\n'''
  for i in range(cantConvBinaryInputNotBinary):
    model_init+=f'''    conv_binary_input_not_binary_layer{i} = init_conv_binary_0_layer{i}(); //Capa conv binaria input not binary {i+1}\n'''
  for i in range(cantBatchNorm):
    model_init+=f'''    batchnorm_layer{i} = init_batchnorm_layer{i}(); // Capa BatchNormalization {i+1}\n'''
  
  model_init+=f'''}}\n'''
  #End of model_init function string
  return model_init

#==============================

#MODEL PREDICT FUNCTION UTILS

def model_predict_separable(layer,index,isFirst,layerNumber,options):

  #me aseguro que tenga una activacion implementada
  assert layer.activation==keras.activations.relu or layer.activation==keras.activations.tanh

  ret=f'''
  // Capa {layerNumber}: Depthwise Separable Conv2D
  separable_conv2d_layer(separable_layer{index},input,&output);
    '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Output matrix layer {layerNumber} (Depthwise Separable Conv2D): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  if layer.activation==keras.activations.relu:
    ret+=f'''// Activation Layer {layerNumber}: relu
  relu(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Relu): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.tanh:
    ret+=f'''// Activation Layer {layerNumber}: tanh
  tanh2d(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Tanh): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  ret+='''input=output;
  '''
  return ret



def model_predict_conv(layer,index,isFirst,layerNumber,options):

  #me aseguro que tenga una activacion implementada
  assert layer.activation==keras.activations.relu or layer.activation==keras.activations.tanh

  ret=f'''
  // Capa {layerNumber}: Conv 2D
  conv2d_layer(conv_layer{index},input,&output);
    '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Output matrix layer {layerNumber} (Conv 2D): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  if layer.activation==keras.activations.relu:
    ret+=f'''// Activation Layer {layerNumber}: relu
  relu(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Relu): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.tanh:
    ret+=f'''// Activation Layer {layerNumber}: tanh
  tanh2d(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Tanh): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  ret+='''input=output;
  '''
  return ret



def model_predict_maxPool(pool_size, stride,layerNumber,options):
  ret = f'''
  // Capa {layerNumber}: MaxPooling2D
  max_pooling_2d({pool_size},{stride},input,&output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
    #if EMBEDIA_DEBUG > 0
    print_data_t("Output Layer  {layerNumber} (MaxPooling 2D): ", output);
    #endif // EMBEDIA_DEBUG
    '''
  ret+='''input=output;
  '''
  return ret

def model_predict_avgPool(pool_size, stride,layerNumber,options):
  ret = f'''
  // Capa {layerNumber}: AveragePooling2D
  avg_pooling_2d({pool_size},{stride},input,&output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
    #if EMBEDIA_DEBUG > 0
    print_data_t("Output Layer  {layerNumber} (AveragePooling 2D): ", output);
    #endif // EMBEDIA_DEBUG
    '''
  ret+='''input=output;
  '''
  return ret

def model_predict_flatten(layerNumber,options):
  ret = f'''
  // Capa {layerNumber}: Flatten
  flatten_data_t f_output;
  flatten_layer(output, &f_output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Output Vector Layer {layerNumber} (Flatten): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  ret += '''f_input=f_output;
  '''
  return ret

def model_predict_dense(layer,index,layerNumber,options,isLastLayer=False):
  assert layer.activation==keras.activations.relu or layer.activation==keras.activations.softmax or layer.activation==keras.activations.tanh
  ret = f'''
  // Capa {layerNumber}: Dense
  dense_forward(dense_layer{index},f_input,&f_output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Output Vector Layer {layerNumber} (Dense): ",f_output);
      #endif // EMBEDIA_DEBUG
    '''
  if layer.activation==keras.activations.relu:
    ret+=f'''
  //Activación Capa {layerNumber}: relu
  relu_flatten(f_output);
  '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Activation Layer {layerNumber} (Relu): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.softmax:
    ret+=f'''
  //Activación Capa {layerNumber}: softmax
  softmax(f_output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Activation Layer {layerNumber} (Softmax): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.tanh:
    ret+=f'''
  //Activación Capa {layerNumber}: tanh
  tanh_flatten(f_output);
  '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Activation Layer {layerNumber} (Tanh): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  if (not isLastLayer):
    ret+='''f_input = f_output;
    '''
  return ret




#NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
def model_predict_conv_binary_input_not_binary(layer,index,isFirst,layerNumber,options):
    
    #me aseguro que tenga una activacion implementada
    assert layer.activation==keras.activations.relu or layer.activation==keras.activations.tanh

    ret=f'''
    // Capa {layerNumber}: Conv 2D  binary input not binary
    quant_conv2d_input_not_binary_layer(conv_binary_input_not_binary_layer{index},input,&output);
      '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Output matrix layer {layerNumber} (Conv 2D): ", output);
        #endif // EMBEDIA_DEBUG
        '''
    if layer.activation==keras.activations.relu:
      ret+=f'''// Activation Layer {layerNumber}: relu
    relu(output);
      '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Activation Layer {layerNumber} (Relu): ", output);
        #endif // EMBEDIA_DEBUG
        '''
    elif layer.activation==keras.activations.tanh:
      ret+=f'''// Activation Layer {layerNumber}: tanh
    tanh2d(output);
      '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Activation Layer {layerNumber} (Tanh): ", output);
        #endif // EMBEDIA_DEBUG
        '''
    ret+='''input=output;
    '''
    return ret


def model_predict_conv_binary(layer,index,isFirst,layerNumber,options):

    #me aseguro que tenga una activacion implementada
    assert layer.activation==keras.activations.relu or layer.activation==keras.activations.tanh

    ret=f'''
    // Capa {layerNumber}: Conv 2D full binary
    quant_conv2d(conv_binary_layer{index},input,&output);
      '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Output matrix layer {layerNumber} (Conv 2D): ", output);
        #endif // EMBEDIA_DEBUG
        '''
    if layer.activation==keras.activations.relu:
      ret+=f'''// Activation Layer {layerNumber}: relu
    relu(output);
      '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Activation Layer {layerNumber} (Relu): ", output);
        #endif // EMBEDIA_DEBUG
        '''
    elif layer.activation==keras.activations.tanh:
      ret+=f'''// Activation Layer {layerNumber}: tanh
    tanh2d(output);
      '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Activation Layer {layerNumber} (Tanh): ", output);
        #endif // EMBEDIA_DEBUG
        '''
    ret+='''input=output;
    '''
    return ret



def model_predict_dense_binary(layer,index,layerNumber,options,isLastLayer=False):
    
    assert layer.activation==keras.activations.relu or layer.activation==keras.activations.softmax or layer.activation==keras.activations.tanh
    ret = f'''
    // Capa {layerNumber}: Dense full binary
    quant_dense_forward(dense_binary_layer{index},f_input,&f_output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_flatten_data_t("Output Vector Layer {layerNumber} (Dense): ",f_output);
        #endif // EMBEDIA_DEBUG
      '''
    if layer.activation==keras.activations.relu:
      ret+=f'''
    //Activación Capa {layerNumber}: relu
    relu_flatten(f_output);
    '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_flatten_data_t("Activation Layer {layerNumber} (Relu): ", f_output);
        #endif // EMBEDIA_DEBUG
        '''
    elif layer.activation==keras.activations.softmax:
      ret+=f'''
    //Activación Capa {layerNumber}: softmax
    softmax(f_output);
      '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_flatten_data_t("Activation Layer {layerNumber} (Softmax): ", f_output);
        #endif // EMBEDIA_DEBUG
        '''
    elif layer.activation==keras.activations.tanh:
      ret+=f'''
    //Activación Capa {layerNumber}: tanh
    tanh_flatten(f_output);
    '''
      if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_flatten_data_t("Activation Layer {layerNumber} (Tanh): ", f_output);
        #endif // EMBEDIA_DEBUG
        '''
    if (not isLastLayer):
      ret+='''f_input = f_output;
      '''
    return ret



def model_predict_batchnorm(layer, index, layerNumber, options, flatten,isLastLayer=False):
  ret = f'''
  // Capa {layerNumber}: BatchNormalization
  '''
  if (not flatten):
    # Estamos trabajando con data_t
    ret += f'''
    batch_normalization(batchnorm_layer{index}, input,&output);  
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
        #if EMBEDIA_DEBUG > 0
        print_data_t("Output matrix layer {layerNumber} (BatchNormalization): ", input);
        #endif // EMBEDIA_DEBUG
        '''
    if (not isLastLayer):
        ret+='''input = output;
        '''
  else:
    # Estamos trabajando con flatten_data_t
    ret += f'''
    batch_normalization_flatten(batchnorm_layer{index}, f_input,&f_output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
        ret+=f'''
          #if EMBEDIA_DEBUG > 0
          print_flatten_data_t("Output Vector Layer {layerNumber} (BatchNormalization): ", f_output);
          #endif // EMBEDIA_DEBUG
          '''
    if (not isLastLayer):
        ret+='''f_input = f_output;
        '''
  return ret







#CREATE MODEL PREDICT FUNCTION recibe modelo binario o no (1 si es binario y utiliza larq)
def create_model_predict(model,options,binario):
  ret='''
int model_predict(data_t input, flatten_data_t * results){
  data_t output;
  flatten_data_t f_input;
  '''
  if options.debug_mode != DebugMode.DISCARD:
      ret+='''
      // Input
          #if EMBEDIA_DEBUG > 0
          print_data_t("Input data:", input);
          #endif // EMBEDIA_DEBUG
      '''

  cantSeparable=0
  cantConv=0
  cantDensas=0
  cantConvBinary = 0
  cantDensasBinary = 0
  cantConvBinaryInputNotBinary = 0
  cantBatchNorm=0
  
  flatten_data = False
  
  if(not binario):

      for i,layer in enumerate(model.layers):
        if 'separable_conv2d' in layer.name:
          ret+=model_predict_separable(layer,cantSeparable,i==0,i+1,options)
          cantSeparable+=1
        elif 'conv2d' in layer.name:
          ret+=model_predict_conv(layer,cantConv,i==0,i+1,options)
          cantConv+=1
        elif 'dense' in layer.name:
          ret+=model_predict_dense(layer,cantDensas,i+1,options,i+1==len(model.layers))
          cantDensas+=1
          flatten_data = True
        elif 'max_pooling2d' in layer.name:
          pool_size,_=layer.pool_size
          stride,_=layer.strides
          ret+=model_predict_maxPool(pool_size,stride,i+1,options)
        elif 'average_pooling2d' in layer.name:
          pool_size,_=layer.pool_size
          stride,_=layer.strides
          ret+=model_predict_avgPool(pool_size,stride,i+1,options)
        elif 'flatten' in layer.name:
          ret+=model_predict_flatten(i+1,options)
          flatten_data = True
        elif 'batch_normalization' in layer.name:
          ret+=model_predict_batchnorm(layer, cantBatchNorm, i+1, options, flatten_data,i+1==len(model.layers))
          cantBatchNorm += 1
          flatten_data = False
        
        else:
          return f"Error: No support for layer {layer} which is of type {layer.activation}"
    
      ret+='''
      int result= argmax(f_output);
      *results = f_output;
      return result;
    }
      '''
      return ret
  else:
      with lq.context.quantized_scope(True):
          for i,layer in enumerate(model.layers):
            if 'separable_conv2d' in layer.name:
              ret+=model_predict_separable(layer,cantSeparable,i==0,i+1,options)
              cantSeparable+=1
            elif 'quant_conv2d' in layer.name:
                if (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] == None):
                    #es una conv normal
                    ret+=model_predict_conv(layer,cantConv,i==0,i+1,options)
                    cantConv+=1
                elif (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] != None):        
                    if (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                       #entrada no binaria
                       ret+=model_predict_conv_binary_input_not_binary(layer,cantConvBinaryInputNotBinary,i==0,i+1,options)
                       cantConvBinaryInputNotBinary+=1
                    else:
                        print(f"Error: No support for layer {layer} with this arguments")
                        raise f"Error: No support for layer {layer} with this arguments"
                elif (layer.get_config()['input_quantizer'] != None) and (layer.get_config()['kernel_quantizer'] != None):
                    if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                        #conv pura binaria
                        ret+=model_predict_conv_binary(layer,cantConvBinary,i==0,i+1,options)
                        cantConvBinary+=1
                    else:
                        print(f"Error: No support for layer {layer} with this arguments")
                        raise f"Error: No support for layer {layer} with this arguments"
                else:
                    print(f"Error: No support for layer {layer} with this arguments")
                    raise f"Error: No support for layer {layer} with this arguments"
              
            elif 'quant_dense' in layer.name:
                if (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] == None):
                    #es una desnse normal
                    ret+=model_predict_dense(layer,cantDensas,i+1,options,i+1==len(model.layers))
                    cantDensas+=1
                    flatten_data = True
                elif (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] != None):        
                    
                       #entrada no binaria
                       print(f"Error: No support for layer {layer} with this arguments")
                       raise f"Error: No support for layer {layer} with this arguments"
                    
                elif (layer.get_config()['input_quantizer'] != None) and (layer.get_config()['kernel_quantizer'] != None):
                    if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                        #dnse pura binaria
                        ret+=model_predict_dense_binary(layer,cantDensasBinary,i+1,options,i+1==len(model.layers))
                        cantDensasBinary+=1
                        flatten_data = True
                    else:
                        print(f"Error: No support for layer {layer} with this arguments")
                        raise f"Error: No support for layer {layer} with this arguments"
                else:
                    print(f"Error: No support for layer {layer} with this arguments")
                    raise f"Error: No support for layer {layer} with this arguments"
             
            
            elif 'conv2d' in layer.name:
              ret+=model_predict_conv(layer,cantConv,i==0,i+1,options)
              cantConv+=1
            elif 'dense' in layer.name:
              ret+=model_predict_dense(layer,cantDensas,i+1,options,i+1==len(model.layers))
              cantDensas+=1
              flatten_data = True
            elif 'max_pooling2d' in layer.name:
              pool_size,_=layer.pool_size
              stride,_=layer.strides
              ret+=model_predict_maxPool(pool_size,stride,i+1,options)
            elif 'average_pooling2d' in layer.name:
              pool_size,_=layer.pool_size
              stride,_=layer.strides
              ret+=model_predict_avgPool(pool_size,stride,i+1,options)
            elif 'flatten' in layer.name:
              ret+=model_predict_flatten(i+1,options)
              flatten_data = True
            elif 'batch_normalization' in layer.name:
              ret+=model_predict_batchnorm(layer, cantBatchNorm, i+1, options, flatten_data,i+1==len(model.layers))
              cantBatchNorm += 1
              flatten_data = False
            
            else:
              return f"Error: No support for layer {layer}"
        
          ret+='''
          int result= argmax(f_output);
          *results = f_output;
          return result;
        }
          '''
          return ret

def create_codeblock_project(project_name,files):
    output=f'''
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<CodeBlocks_project_file>
	<FileVersion major="1" minor="6" />
	<Project>
		<Option title="{project_name}" />
		<Option pch_mode="2" />
		<Option compiler="gcc" />
		<Build>
			<Target title="Debug">
				<Option output="bin/Debug/{project_name}" prefix_auto="1" extension_auto="1" />
				<Option object_output="obj/Debug/" />
				<Option type="1" />
				<Option compiler="gcc" />
				<Compiler>
					<Add option="-g" />
				</Compiler>
			</Target>
			<Target title="Release">
				<Option output="bin/Release/{project_name}" prefix_auto="1" extension_auto="1" />
				<Option object_output="obj/Release/" />
				<Option type="1" />
				<Option compiler="gcc" />
				<Compiler>
					<Add option="-O2" />
				</Compiler>
				<Linker>
					<Add option="-s" />
				</Linker>
			</Target>
		</Build>
		<Compiler>
			<Add option="-Wall" />
		</Compiler>'''
    for filename in files:
        if filename[-2:].lower()=='.c':
            output+=f'''
		<Unit filename="{filename}">
			<Option compilerVar="CC" />
		</Unit>'''
        elif filename[-2:].lower()=='.h':
            output+=f'''
		<Unit filename="{filename}" />'''

    output+='''
		<Extensions>
			<code_completion />
			<envvars />
			<debugger />
		</Extensions>
	</Project>
</CodeBlocks_project_file>'''
    return output
