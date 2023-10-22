#pragma once

/******** input/output size ***********/
#ifndef BatchSize
    #define BatchSize 1
#endif
/******** input/output size end ***********/

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define Val(matrix, x, y, Y) matrix[(x)*Y+(y)]
#define Val3D(matrix, x, y, z, Y, Z) matrix[((x)*Y+(y))*Z+(z)]
#define Val4D(matrix, a, b, c, d, B, C, D) matrix[(((a)*B+(b))*C+(c))*D+(d)]