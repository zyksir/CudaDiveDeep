#pragma once

const int WARPSIZE = 32; // warpSize is not constexpr
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define Val(matrix, x, y, Y) matrix[(x)*Y+(y)]
#define Val3D(matrix, x, y, z, Y, Z) matrix[((x)*Y+(y))*Z+(z)]
#define Val4D(matrix, a, b, c, d, B, C, D) matrix[(((a)*B+(b))*C+(c))*D+(d)]