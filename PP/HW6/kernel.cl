__kernel void convolution(__global float* input,                           
                         __global float* output,                          
                         __constant float* filter,                        
                         const int filterWidth,                           
                         const int imageWidth,                            
                         const int imageHeight) {                         
    int x = get_global_id(0);                                            
    int y = get_global_id(1);                                            
                                                                         
    if (x >= imageWidth || y >= imageHeight) return;                     
                                                                         
    float sum = 0.0f;                                                    
    int radius = filterWidth / 2;                                        
                                                                         
    // Iterate over filter window                                        
    for (int i = -radius; i <= radius; i++) {                           
        for (int j = -radius; j <= radius; j++) {                       
            // Get sample position                                       
            int sampleY = y + i;                                        
            int sampleX = x + j;                                        
                                                                        
            // Handle boundary conditions with clamping                  
            sampleX = max(0, min(sampleX, imageWidth - 1));             
            sampleY = max(0, min(sampleY, imageHeight - 1));            
                                                                        
            // Calculate filter and input indices                       
            int filterIndex = (i + radius) * filterWidth + (j + radius);
            int inputIndex = sampleY * imageWidth + sampleX;            
                                                                        
            // Accumulate weighted sum                                  
            sum += input[inputIndex] * filter[filterIndex];             
        }                                                               
    }                                                                   
                                                                        
    // Write output                                                     
    output[y * imageWidth + x] = sum;                                   
}