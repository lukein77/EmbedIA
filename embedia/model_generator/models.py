# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:32:04 2021

@author: cesar
"""

#descomentar si se corre sola la celda
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from embedia.project_options import ProjectType, ModelDataType, DebugMode, BinaryBlockSize
from embedia.model_generator.utils.generator_utils import exportar_batchnorm_a_c,exportar_densa_binaria_a_c,exportar_separable_a_c,exportar_cnn_a_c,exportar_bnn_a_c,exportar_densa_a_c,create_model_predict,create_model_init

import larq as lq



block_type = 'uint8_t'
xBits = 8


def format_model_name(model):
    model_name = model.name.lower()
    if not model_name.endswith('model'):
       model_name+= '_model'
       
    return model_name
    
def macro_fixed_converter(v):
    return f'''FL2FX({v})''' 

def macro_float_converter(v):
    return v    

def create_model_template_c(model, model_binary,options):
  '''Receives: model and options object
  Returns: code, header and filename with model weights and the library
  function (model_predict) to make predictions
  '''
  if options.tamano_bloque == BinaryBlockSize.Bits8:
      
      block_type = 'uint8_t'
      xBits = 8
      
  elif options.tamano_bloque == BinaryBlockSize.Bits16:
      
      block_type = 'uint16_t'
      xBits = 16
      
  else:
      
      block_type = 'uint32_t'
      xBits = 32
  

    
  if options.data_type != ModelDataType.BINARY:
    
      h_ext = 'h'
      
      if options.data_type == ModelDataType.FLOAT :
        macro_converter= macro_float_converter
        data_type ='float'
      else:
        macro_converter = macro_fixed_converter
        data_type = 'fixed'      
      
      model_name = format_model_name(model)
      model_name_h = model_name.upper()
      
      _,in_height,in_width,in_chan = model.input_shape #(none,h,w,ch)
      c_header = ""
      c_code = ""
      
      #=========================#
      #Begin Make of header (.h)
    
      #Libs imported on header
      
      libs= f'''
    /* EmbedIA model - Autogenerado */
    #ifndef {model_name_h}_H
    #define {model_name_h}_H
    
    #include "embedia.{h_ext}"
    #define {model_name_h}_CHANNELS {in_chan}
    #define {model_name_h}_HEIGHT {in_height}
    #define {model_name_h}_WIDTH {in_width}'''
      
      c_header+=libs
    
      c_header+='''
    void model_init();
    
    int model_predict(data_t input, flatten_data_t * results);
    '''
      c_header+='''
    #endif
    '''
      #End of header
      #=========================#
      #Begin make .c file
      c_code += f'''#include "{model_name}.{h_ext}"\n'''
      
      if options.debug_mode != DebugMode.DISCARD:
          c_code+= '#include "embedia_debug.h"\n'
      c_code+="\n"
      #Counting ammount of conv layers and dense layers found
      cantSeparable=0
      cantConv=0
      cantDensas=0
      cantBatchNorm=0
      func_batchnorm=""
      func_separable=""
      func_conv=""
      func_dens=""
      #Adding functions with model weights inicialization
      #(TO-DO use another method to iterate through layer types)
      for layer in model.layers:
        if 'separable_conv2d' in layer.name:
          func_separable+=exportar_separable_a_c(layer, cantSeparable, macro_converter, data_type)
          cantSeparable+=1
        elif 'conv2d' in layer.name:
          func_conv+=exportar_cnn_a_c(layer, cantConv, macro_converter, data_type)
          cantConv+=1
        elif 'dense' in layer.name:
          func_dens+=exportar_densa_a_c(layer,cantDensas, macro_converter, data_type)
          cantDensas+=1
        elif 'max_pooling2d' in layer.name:
          #do nothing yet
          continue
        elif 'average_pooling2d' in layer.name:
          #do nothing yet
          continue
        elif 'flatten' in layer.name:
          #do nothing here
          continue
        elif 'batch_normalization' in layer.name:
          func_batchnorm+=exportar_batchnorm_a_c(layer, cantBatchNorm, macro_converter, data_type)
          cantBatchNorm += 1
        else:
          print(f"Error: No support for layer {layer} which is of type {layer.activation} (dont change it's name)")
          raise f"Error: No support for layer {layer} which is of type {layer.activation}"
    
    
      #Private global variables declaration
      var_decl = '' 
      for i in range(cantSeparable):
        var_decl+=f'''separable_layer_t separable_layer{i}; // Capa depthwise separable conv {i+1}\n'''
      for i in range(cantConv):
        var_decl+=f'''conv_layer_t conv_layer{i}; // Capa convolucional {i+1}\n'''
      for i in range(cantDensas):
        var_decl+=f'''dense_layer_t dense_layer{i}; // Capa densa {i+1}\n'''
      for i in range(cantBatchNorm):
        var_decl+=f'''batchnorm_layer_t batchnorm_layer{i}; // Capa BatchNormalization {i+1}\n'''

      c_code+= var_decl #Adds local (.c) variables (layers)
    
      c_code+= func_separable #Adds function to initialize conv layers
      c_code+= func_conv #Adds function to initialize conv layers
      c_code+= func_dens #Adds function to initialize dense layers
      c_code+= func_batchnorm #Adds function to initialize batch normalization layers
      #Adds function which calls layers initializators
      c_code+= create_model_init(cantConv,cantDensas,cantSeparable,0,0,0,cantBatchNorm)
      
      #Adds model_predict function
      c_code+= '\n\n'+create_model_predict(model,options,0)
    
      return (c_code, c_header, model_name)
  else:
      
      h_ext = 'h'
      data_type ='float'
      
      
      macro_converter= macro_float_converter
  
      
      model_name = format_model_name(model_binary)
      model_name_h = model_name.upper()
      
      _,in_height,in_width,in_chan = model_binary.input_shape #(none,h,w,ch)
      c_header = ""
      c_code = ""
      
      #=========================#
      #Begin Make of header (.h)
    
      #Libs imported on header
      
      libs= f'''
    /* EmbedIA model_binary - Autogenerado */
    #ifndef {model_name_h}_H
    #define {model_name_h}_H
    
    #include "embedia.{h_ext}"
    #define {model_name_h}_CHANNELS {in_chan}
    #define {model_name_h}_HEIGHT {in_height}
    #define {model_name_h}_WIDTH {in_width}'''
      
      c_header+=libs
    
      c_header+='''
    void model_init();
    
    int model_predict(data_t input, flatten_data_t * results);
    '''
      c_header+='''
    #endif
    '''
      #End of header
      #=========================#
      #Begin make .c file
      c_code += f'''#include "{model_name}.{h_ext}"\n'''
      
      if options.debug_mode != DebugMode.DISCARD:
          c_code+= '#include "embedia_debug.h"\n'
      c_code+="\n"
      #Counting ammount of conv layers and dense layers found
      cantSeparable=0
      cantConv=0
      cantDensas=0
      func_separable=""
      func_conv=""
      func_dens=""
      cantConvBinary = 0
      cantDensasBinary = 0
      cantBatchNorm=0
      cantConvBinaryInputNotBinary = 0
      func_batchnorm=""
      func_conv_binary =""
      func_conv_binary_input_not_binary =""
      func_dens_binary = ""
      #Adding functions with model_binaryel weights inicialization
      #(TO-DO use another method to iterate through layer types)
      for layer in model_binary.layers:
        if 'separable_conv2d' in layer.name:
          func_separable+=exportar_separable_a_c(layer, cantSeparable, macro_converter, data_type)
          cantSeparable+=1
        elif 'quant_conv2d' in layer.name:
            with lq.context.quantized_scope(True):
                if (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] == None):
                    #es una conv normal
                    func_conv+=exportar_cnn_a_c(layer, cantConv, macro_converter, data_type)
                    cantConv+=1
                elif (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] != None):        
                    if (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                       #entrada no binaria
                       func_conv_binary_input_not_binary+=exportar_bnn_a_c(layer, cantConvBinary, macro_converter, data_type,block_type,xBits,0)
                       cantConvBinaryInputNotBinary+=1
                    else:
                        print(f"Error: No support for layer {layer} with this arguments")
                        raise f"Error: No support for layer {layer} with this arguments"
                elif (layer.get_config()['input_quantizer'] != None) and (layer.get_config()['kernel_quantizer'] != None):
                    if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                        #conv pura binaria
                        func_conv_binary+=exportar_bnn_a_c(layer, cantConvBinary, macro_converter, data_type,block_type,xBits,1)
                        cantConvBinary+=1
                    else:
                        print(f"Error: No support for layer {layer} with this arguments")
                        raise f"Error: No support for layer {layer} with this arguments"
                else:
                    print(f"Error: No support for layer {layer} with this arguments")
                    raise f"Error: No support for layer {layer} with this arguments"
        elif 'quant_dense' in layer.name:
            with lq.context.quantized_scope(True):
                if (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] == None):
                    #es una desnse normal
                    func_dens+=exportar_densa_a_c(layer,cantDensas, macro_converter, data_type)
                    cantDensas+=1
                elif (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['kernel_quantizer'] != None):        
                    
                       #entrada no binaria
                       print(f"Error: No support for layer {layer} with this arguments")
                       raise f"Error: No support for layer {layer} with this arguments"
                    
                elif (layer.get_config()['input_quantizer'] != None) and (layer.get_config()['kernel_quantizer'] != None):
                    if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['kernel_quantizer']['class_name'] == 'SteSign'):
                        #dnse pura binaria
                        func_dens_binary+=exportar_densa_binaria_a_c(layer, cantDensasBinary, macro_converter, data_type,block_type,xBits)
                        cantDensasBinary+=1
                    else:
                        print(f"Error: No support for layer {layer} with this arguments")
                        raise f"Error: No support for layer {layer} with this arguments"
                else:
                    print(f"Error: No support for layer {layer} with this arguments")
                    raise f"Error: No support for layer {layer} with this arguments"
                
        
        elif 'conv2d' in layer.name:
          func_conv+=exportar_cnn_a_c(layer, cantConv, macro_converter, data_type)
          cantConv+=1
        elif 'dense' in layer.name:
          func_dens+=exportar_densa_a_c(layer,cantDensas, macro_converter, data_type)
          cantDensas+=1
        elif 'max_pooling2d' in layer.name:
          #do nothing yet
          continue
        elif 'average_pooling2d' in layer.name:
          #do nothing yet
          continue
        elif 'flatten' in layer.name:
          #do nothing here
          continue
        elif 'batch_normalization' in layer.name:
          func_batchnorm+=exportar_batchnorm_a_c(layer, cantBatchNorm, macro_converter, data_type)
          cantBatchNorm += 1
        
        else:
          print(f"Error: No support for layer {layer} which is of type {layer.activation} (dont change it's name)")
          raise f"Error: No support for layer {layer} which is of type {layer.activation}"
    
        
      #Private global variables declaration
      var_decl = '' 
      for i in range(cantSeparable):
        var_decl+=f'''separable_layer_t separable_layer{i}; // Capa depthwise separable conv {i+1}\n'''
      for i in range(cantConv):
        var_decl+=f'''conv_layer_t conv_layer{i}; // Capa convolucional {i+1}\n'''
      for i in range(cantDensas):
        var_decl+=f'''dense_layer_t dense_layer{i}; // Capa densa {i+1}\n'''
      for i in range(cantConvBinary):
        var_decl+=f'''quant_conv_layer_t conv_binary_layer{i}; // Capa convolucional binaria {i+1}\n'''
      for i in range(cantDensasBinary):
        var_decl+=f'''quant_dense_layer_t dense_binary_layer{i}; // Capa densa binaria {i+1}\n'''
      for i in range(cantConvBinaryInputNotBinary):
        var_decl+=f'''quant_conv_layer_t conv_binary_input_not_binary_layer{i}; // Capa convolucional binaria input no binaria{i+1}\n'''
      for i in range(cantBatchNorm):
        var_decl+=f'''batchnorm_layer_t batchnorm_layer{i}; // Capa BatchNormalization {i+1}\n'''

        
        
      c_code+= var_decl #Adds local (.c) variables (layers)
    
      c_code+= func_separable #Adds function to initialize conv layers
      c_code+= func_conv #Adds function to initialize conv layers
      c_code+= func_dens #Adds function to initialize dense layers
      c_code+= func_conv_binary #Adds function to initialize conv layers bin
      c_code+= func_conv_binary_input_not_binary #Adds function to initialize conv layers binary input not binary
      c_code+= func_dens_binary #Adds function to initialize dense layers binary
      c_code+= func_batchnorm #Adds function to initialize batch normalization layers
      #Adds function which calls layers initializators
      c_code+= create_model_init(cantConv,cantDensas,cantSeparable,cantConvBinary,cantDensasBinary,cantConvBinaryInputNotBinary,cantBatchNorm)
      
      #Adds model_predict function
      c_code+= '\n\n'+create_model_predict(model_binary,options,1)
    
      return (c_code, c_header, model_name)
       
        
        

def image_to_array_str(image, macro_converter, clip=120):
    output = ''
    cline = '  ' 
    for i in image.flatten():
        cline+= macro_converter(str(i)) + ', '
        if len(cline) > clip:
            output += cline +'\n';
            cline ='  '
    output += cline
    salidita =  output[:-2]
    return salidita
        
    
def create_main_template_c(model, example=None, example_comment='', options=None, baudRate=9600):
  model_name = format_model_name(model)
  model_name_h=model_name.upper()
  
  h_ext ='h'
            
  if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
    macro_converter= macro_float_converter
    data_type = 'float'
  else:
    macro_converter = macro_fixed_converter  
    data_type = 'fixed'      

  if example is not None:
      example = '= {\n' + image_to_array_str(example, macro_converter) + '}'
      fill_input = ''
  else:
      example = ''
      example = ''
      fill_input = f'''// fill input data structure
    fill_input(&input);'''
      

  if example_comment == '':
      example_comment = '// Buffer with data example'
  else:
      example_comment = f'''// Buffer with {example_comment}'''

  if options.project_type == ProjectType.ARDUINO:
      extra_include = ''
      print_block = f'''Serial.print("Prediction class id: ");
    Serial.println(prediction);'''
      main_block = f'''void setup(){{
      
    Serial.begin({baudRate});

    // model initialization
    model_init();
      
}} 
        
void loop(){{
      '''
      end_block = ''
  else:
      extra_include = '#include <stdio.h>'
      print_block = f'''printf("Prediction class id: %d\\n", prediction);'''
      
      main_block = f'''int main(void){{

  // model initialization
  model_init();'''

      end_block = '  return 0;'
     

  return f'''{extra_include}
#include "embedia.{h_ext}"
#include "{model_name}.{h_ext}"

#define INPUT_SIZE ({model_name_h}_CHANNELS*{model_name_h}_WIDTH*{model_name_h}_HEIGHT)

{example_comment}
{data_type} input_data[INPUT_SIZE]{example}; 

// Structure with input data for the inference function
data_t input = {{ {model_name_h}_CHANNELS, {model_name_h}_WIDTH, {model_name_h}_HEIGHT, input_data }};

// Structure with inference output results
flatten_data_t results;

{main_block}
    {fill_input}
  // model inference
  int prediction = model_predict(input, &results);    
    
  // print predicted class id
  {print_block} 

{end_block}
}}
  '''