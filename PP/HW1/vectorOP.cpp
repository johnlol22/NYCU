#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
 
  __pp_vec_int exp_val;
  __pp_vec_int vec_ones = _pp_vset_int(1);
  __pp_vec_int vec_zeros = _pp_vset_int(0);
  __pp_mask tmp;
  __pp_mask remaining;
  __pp_mask mask_int_ones = _pp_init_ones();
  __pp_mask mask_int_zeros = _pp_init_ones(0);
  __pp_vec_float vec_val;
  __pp_vec_float vec_float_ones = _pp_vset_float(1.0f);
  __pp_vec_float vec_float_max = _pp_vset_float(9.999999f);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
	  __pp_vec_float res = _pp_vset_float(1.0f);
	  if (N - i >= VECTOR_WIDTH){
	  	_pp_vload_float(vec_val, values+i, mask_int_ones);
		_pp_vload_int(exp_val, exponents+i, mask_int_ones);
	  }else {
		remaining = _pp_init_ones(N-i);
	  	_pp_vload_float(vec_val, values+i, remaining);
		_pp_vload_int(exp_val, exponents+i, remaining);
		mask_int_ones = remaining;
	  }
	  _pp_vgt_int(tmp, exp_val, vec_zeros, mask_int_ones);
	  while (_pp_cntbits(tmp)){
	  	_pp_vmult_float(res, res, vec_val, tmp);
		_pp_vsub_int(exp_val, exp_val, vec_ones, tmp);
		_pp_vgt_int(tmp, exp_val, vec_zeros, mask_int_ones);
	  }
	  __pp_mask overflow;
	  _pp_vgt_float(overflow, res, vec_float_max, mask_int_ones);
	  _pp_vmove_float(res, vec_float_max, overflow);
	  _pp_vstore_float(output+i, res, mask_int_ones);  
  }
	 
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float res_vec = _pp_vset_float(0.0f);
  __pp_vec_float a;
  __pp_mask mask_int_ones = _pp_init_ones();
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
	int tmp = log2(VECTOR_WIDTH) - 1;
	_pp_vload_float(a, values + i, mask_int_ones);
	while (tmp > 0)
	{
		_pp_hadd_float(a, a);
		_pp_interleave_float(a, a);
		tmp--;
	}
	_pp_hadd_float(a, a);
	_pp_vadd_float(res_vec, res_vec, a, mask_int_ones);
  }
  return res_vec.value[0];
}
