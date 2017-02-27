#include <stdint.h>
#include <x86intrin.h>

namespace VarintTables {

extern const __m128i VarintG8CUSSEMasks[4][256][2] = {
	{ // state 0
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff070605040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06050403ff020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0x06050403ffffff02U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0x07060504ffff0302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0x07060504ffffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0x07060504ffffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffffff07060504U, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff070605ff040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xff070605ffff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xff070605ffff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xff070605ffffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xff070605ffffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xff070605ffffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xff070605ffffff04U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff070605040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff050403ff020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0706ff050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0706ff050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff050403ffffff02U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffff0706ffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffff0706ffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0504ffff0302U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffff0706ffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0504ffffff03U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0504ffffff03U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffff0706ffff0504U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff05ff040302U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff05ffff0403U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff05ffff0403U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff05ffffff04U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff05ffffff04U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffff05ffffff04U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff05ffffff04U, 0xffffffffffff0706U}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff0605040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06050403ff020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0x06050403ffffff02U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff07ff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff07ff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff060504ffff0302U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffff07ff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xff060504ffffff03U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xff060504ffffff03U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff07ff060504U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0605ff040302U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0605ffff0403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0605ffff0403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffff0605ffffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffff0605ffffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffff0605ffffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffff0605ffffff04U, 0xffffffffffffff07U}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffff06ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffff07ffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff0605040302U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff050403ff020100U, 0xffffff07ffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff06ff050403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff06ff050403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff050403ffffff02U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffff07ffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff06ffff0504U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff06ffff0504U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0504ffff0302U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffff06ffff0504U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0504ffffff03U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0504ffffff03U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff06ffff0504U, 0xffffffffffffff07U}
		},
		{
			{0xffffff0403020100U, 0xffffff06ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff05ff040302U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff05ffff0403U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff05ffff0403U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff0403020100U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff05ffffff04U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff05ffffff04U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff03ff020100U, 0xffffff05ffffff04U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff05ffffff04U, 0xffffff07ffffff06U}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffffff05040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06050403ff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffffff06050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffffff06050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0x06050403ffffff02U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0x07060504ffff0302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0x07060504ffffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0x07060504ffffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffffff07060504U, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff070605ff040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xff070605ffff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xff070605ffff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xff070605ffffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xff070605ffffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xff070605ffffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xff070605ffffff04U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff070605040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff050403ff020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0706ff050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0706ff050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff050403ffffff02U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffff0706ffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffff0706ffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0504ffff0302U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffff0706ffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0504ffffff03U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0504ffffff03U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffff0706ffff0504U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff05ff040302U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff05ffff0403U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff05ffff0403U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffff0706ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff05ffffff04U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff05ffffff04U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffff05ffffff04U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffff0706ffffff05U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff05ffffff04U, 0xffffffffffff0706U}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff0605040302U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06050403ff020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0x06050403ffffff02U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff07ff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff07ff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff060504ffff0302U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffff07ff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xff060504ffffff03U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xff060504ffffff03U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff07ff060504U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0605ff040302U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0605ffff0403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0605ffff0403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffff07ffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffff0605ffffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffff0605ffffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffff0605ffffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffffff07ffff0605U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffff0605ffffff04U, 0xffffffffffffff07U}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffff06ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05040302ffff0100U, 0xffffff07ffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff0605040302U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff050403ff020100U, 0xffffff07ffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff06ff050403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff06ff050403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xff050403ffffff02U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffff07ffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff06ffff0504U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff06ffff0504U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0504ffff0302U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff03ff020100U, 0xffffff06ffff0504U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffff0504ffffff03U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffff0504ffffff03U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff06ffff0504U, 0xffffffffffffff07U}
		},
		{
			{0xffffff0403020100U, 0xffffff06ffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04030201ffffff00U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff040302ffff0100U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff05ff040302U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff0403ff020100U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff05ffff0403U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff02ffff0100U, 0xffffff05ffff0403U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffff0403ffffff02U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff0403020100U, 0xffffff06ffffff05U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff030201ffffff00U, 0xffffff05ffffff04U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff0302ffff0100U, 0xffffff05ffffff04U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffffff01ffffff00U, 0xffffff04ffff0302U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff03ff020100U, 0xffffff05ffffff04U},
			{0xffffff07ffffff06U, 0xffffffffffffffffU}
		},
		{
			{0xffff0201ffffff00U, 0xffffff04ffffff03U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff02ffff0100U, 0xffffff04ffffff03U},
			{0xffffff06ffffff05U, 0xffffffffffffff07U}
		},
		{
			{0xffffff01ffffff00U, 0xffffff03ffffff02U},
			{0xffffff05ffffff04U, 0xffffff07ffffff06U}
		},
	},
	{ // state 1
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x0706050403ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xff07060504ffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xff07060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xff07060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffff070605ff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffff070605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffff070605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffff070605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffff070605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffff070605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffff070605ffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06ff050403020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffffff0706ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffffff0706ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ff050403ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffffff0706ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffffff0706ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffff0504ffff03U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffff0706ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffff0504ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffff0504ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffffff0706ffff05U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffff0706ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0xffffff0706ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffffff05ff0403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffff0706ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffffff05ffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffffff05ffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x06ffffff05ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x06ffffff05ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x06ffffff05ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x06ffffff05ffffffU, 0xffffffffffffff07U}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffffffff06050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x0706050403ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06ff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffffffff07ff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffffffff07ff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x07ff060504ffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffff07ff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x07ff060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x07ff060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffffffff07ff0605U, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffff07ffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0xffffffff07ffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x07ffff0605ff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffff07ffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x07ffff0605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x07ffff0605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x07ffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x07ffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x07ffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x07ffff0605ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffff07ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x07ffffff06050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06ff050403020100U, 0xffffffff07ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x07ffffff06ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x07ffffff06ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ff050403ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x07ffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x07ffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffff0504ffff03U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x07ffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffff0504ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffff0504ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x07ffffff06ffff05U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffff05ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0x07ffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0x07ffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffffff05ff0403U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0x07ffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffffff05ffff04U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffffff05ffff04U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x06ffffff05ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x06ffffff05ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x06ffffff05ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x06ffffff05ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x06ffffff05ffffffU, 0xffffffff07ffffffU}
		},
		{
			{0xffffffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04030201ffff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05040302ff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffffffffff050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffffffffff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffffffffff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xff06050403ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xff07060504ffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xff07060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xff07060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffffffffff070605U, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffff070605ff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffff070605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffff070605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffff070605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffff070605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffff070605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffff070605ffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffffff0706050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06ff050403020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffffff0706ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffffff0706ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ff050403ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffffff0706ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffffff0706ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffff0504ffff03U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffff0706ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffff0504ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffff0504ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffffff0706ffff05U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffff0706ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0xffffff0706ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffffff05ff0403U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffff0706ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffffff05ffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffffff05ffff04U},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x06ffffff05ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x06ffffff05ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x06ffffff05ffffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0xffffff0706ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x06ffffff05ffffffU, 0xffffffffffffff07U}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0xffffffff06050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x0706050403ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06ff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0xffffffff07ff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0xffffffff07ff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x07ff060504ffff03U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffff07ff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x07ff060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x07ff060504ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0xffffffff07ff0605U, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffff07ffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0xffffffff07ffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x07ffff0605ff0403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffff07ffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x07ffff0605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x07ffff0605ffff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x07ffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x07ffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x07ffff0605ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0xffffffff07ffff06U, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x07ffff0605ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0xffffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605040302ff0100U, 0xffffffff07ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x07ffffff06050403U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x06ff050403020100U, 0xffffffff07ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x07ffffff06ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x07ffffff06ff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ff050403ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0xffffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x07ffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x07ffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffff0504ffff03U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x07ffffff06ffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffff0504ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffff0504ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x07ffffff06ffff05U, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0xffffffff05ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504030201ffff00U, 0x07ffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ff040302ff0100U, 0x07ffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x06ffffff05ff0403U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x05ffff0403020100U, 0x07ffffff06ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x06ffffff05ffff04U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x06ffffff05ffff04U},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffff0403ffffffU},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x06ffffff05ffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ff030201ffff00U, 0x06ffffff05ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffff0302ff0100U, 0x06ffffff05ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x05ffffff04ffff03U},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x04ffffff03020100U, 0x06ffffff05ffffffU},
			{0xffffffff07ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffff0201ffff00U, 0x05ffffff04ffffffU},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x03ffffff02ff0100U, 0x05ffffff04ffffffU},
			{0x07ffffff06ffffffU, 0xffffffffffffffffU}
		},
		{
			{0x02ffffff01ffff00U, 0x04ffffff03ffffffU},
			{0x06ffffff05ffffffU, 0xffffffff07ffffffU}
		},
	},
	{ // state 2
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff0706050403ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffff07060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xffff07060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffff07060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605ff0403020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffff070605ff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0605ffff0403ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0605ffffff04ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0x0605ffffff04ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0605ffffff04ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0xffffff070605ffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605ff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffffff0706ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffff0706ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0706ff050403ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0706ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0x0706ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0706ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0706ffffff05ff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff05ffff0403ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff05ffffff04ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0605ffffff04ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff05ffffff04ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0x0706ffffff05ffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffffffff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff0706050403ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xffffffffff07ff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffff07ff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff07ff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff06ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff07ff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff07ff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xffffffffff07ff06U, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffff07ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605ff0403020100U, 0xffffffffff07ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff07ffff0605ff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0605ffff0403ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0605ffffff04ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff06ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0x0605ffffff04ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0605ffffff04ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0xff07ffff0605ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xffffffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffff07ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff07ffffff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xffffffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xff07ffffff06ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff07ffffff06ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff06ff050403ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff06ffff0504ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff06ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff06ffff0504ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff06ffff0504ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffffffff04ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff06ffffff05ff04U},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xff06ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff05ffff0403ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff05ffffff04ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xffffffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff05ffffff04ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0xff06ffffff05ffffU, 0xffffffffff07ffffU}
		},
		{
			{0xffffffffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04030201ff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffffffffff0504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xffff06050403ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffff07060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xffff07060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffff07060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xffffffffffff0706U, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605ff0403020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffff070605ff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0605ffff0403ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffff070605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0605ffffff04ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0x0605ffffff04ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0605ffffff04ffffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0xffffff070605ffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffffff07060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605ff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffffff0706ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffff0706ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0706ff050403ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0706ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0x0706ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0706ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xffffffff0706ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0706ffffff05ff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff05ffff0403ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0x0706ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff05ffffff04ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0605ffffff04ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff05ffffff04ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xffffffff0706ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0x0706ffffff05ffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xffffffffff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff0706050403ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xffffffffff07ff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffff07ff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff07ff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff06ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff07ff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff07ff060504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xffffffffff07ff06U, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504030201ff00U, 0xffffffffff07ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0605ff0403020100U, 0xffffffffff07ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff07ffff0605ff04U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xffffffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0605ffff0403ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xffffffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xff07ffff0605ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0x0605ffffff04ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff06ffff0504ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0x0605ffffff04ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0x0605ffffff04ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0xff07ffff0605ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xffffffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffff07ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff07ffffff060504U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xffffffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xff07ffffff06ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff07ffffff06ff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff06ff050403ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xffffffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ff030201ff00U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0504ffff03020100U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff06ffff0504ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff06ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff06ffff0504ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff06ffff0504ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0x0504ffffff03ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xffffffffff04ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504030201ff00U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff05ff0403020100U, 0xff07ffffff06ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff06ffffff05ff04U},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xff06ffffff05ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffff0201ff00U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0403ffffff020100U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff05ffff0403ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ff030201ff00U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff04ffff03020100U, 0xff06ffffff05ffffU},
			{0xffffffffff07ffffU, 0xffffffffffffffffU}
		},
		{
			{0x0302ffffff01ff00U, 0xff05ffffff04ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xffffffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffff0201ff00U, 0xff05ffffff04ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff03ffffff020100U, 0xff05ffffff04ffffU},
			{0xff07ffffff06ffffU, 0xffffffffffffffffU}
		},
		{
			{0xff02ffffff01ff00U, 0xff04ffffff03ffffU},
			{0xff06ffffff05ffffU, 0xffffffffff07ffffU}
		},
	},
	{ // state 3
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff0706050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504ff03020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffff07060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffff07060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffff06050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x060504ffffff03ffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504ff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffff070605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffff070605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x070605ffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffff070605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0x070605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffff050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0x070605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x060504ffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffff070605ffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffff0706ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0706ff050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffffffffff0706ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xff0706ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xff0706ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0605ffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffff0706ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xff0706ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xff0706ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xff0605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xff0706ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xff0605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffffff0605ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xff0706ffffff05ffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff0706050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504ff03020100U, 0xffffffffffff07ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff07ff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff07ff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff06ff050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x060504ffffff03ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffff07ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff07ffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffff07ffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff06ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0605ffff0403ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffff07ffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff06ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xff0605ffffff04ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xff0605ffffff04ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffff07ffff0605ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff07ffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffff07ffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff06ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff06ff050403ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffff07ffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffff06ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff06ffff0504ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff06ffff0504ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffff07ffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffff06ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff06ffffff05ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffff06ffffff05ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffff06ffffff05ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffffffff05ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffff06ffffff05ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffff06ffffff05ffU, 0xffffffffffff07ffU}
		},
		{
			{0xffffffffffffff00U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffff06050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffffff0403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504ff03020100U, 0xffffffffffffff07U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffffff06U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffff07060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffffff05U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffff07060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffff06050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x060504ffffff03ffU},
			{0xffffffffffffff07U, 0xffffffffffffffffU}
		},
		{
			{0xffffffff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x0706050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504ff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffff070605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffff070605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x070605ffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffff070605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0x070605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffff050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0x070605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x060504ffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffff070605ffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffff070605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffff0706ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0706ff050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffffffffff0706ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xff0706ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xff0706ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0605ffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffff0706ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xff0706ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xff0706ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xff0605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xff0706ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xff0605ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffffffffff0706ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffffff0605ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xff0706ffffff05ffU, 0xffffffffffffffffU}
		},
		{
			{0xffffffffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffff0706U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff0706050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x060504ff03020100U, 0xffffffffffff07ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff07ff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffffffffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff07ff060504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff06ff050403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0x060504ffffff03ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff06050403020100U, 0xffffffffffff07ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff07ffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffffffffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffff07ffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff06ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0605ffff0403ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffffffffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffff07ffff0605ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff06ffff0504ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xff0605ffffff04ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xff0605ffffff04ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffff07ffff0605ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffffffffffffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffffffffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff07ffffff0605U},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffffffffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x050403ffff020100U, 0xffff07ffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff06ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff06ff050403ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffffffffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0504ff03020100U, 0xffff07ffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffff06ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff06ffff0504ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff06ffff0504ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xff0504ffffff03ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffffffffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff050403020100U, 0xffff07ffffff06ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffff06ffffff05ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0x040302ffffff0100U, 0xffff06ffffff05ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xff0403ffff020100U, 0xffff06ffffff05ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff05ffff0403ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffffffffffffU, 0xffffffffffffffffU}
		},
		{
			{0xffff04ff03020100U, 0xffff06ffffff05ffU},
			{0xffffffffffff07ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffffffffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xff0302ffffff0100U, 0xffff05ffffff04ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffffffffffff05ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff03ffff020100U, 0xffff05ffffff04ffU},
			{0xffff07ffffff06ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffff06ffffff05ffU, 0xffffffffffffffffU}
		},
		{
			{0xffff02ffffff0100U, 0xffff04ffffff03ffU},
			{0xffff06ffffff05ffU, 0xffffffffffff07ffU}
		},
	},
};

// number of encoded bytes in the group described by the descriptor desc.
extern const uint8_t VarintG8CUOutputOffset[4][256] = {
	{ // state 0
        8, 11, 10, 14,  9, 13, 13, 17,  8, 12, 12, 16, 12, 16, 16, 20, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
       10, 14, 14, 18, 14, 18, 18, 22, 14, 18, 18, 22, 18, 22, 22, 26, 
        5,  9,  9, 13,  9, 13, 13, 17,  9, 13, 13, 17, 13, 17, 17, 21, 
        9, 13, 13, 17, 13, 17, 17, 21, 13, 17, 17, 21, 17, 21, 21, 25, 
        9, 13, 13, 17, 13, 17, 17, 21, 13, 17, 17, 21, 17, 21, 21, 25, 
       13, 17, 17, 21, 17, 21, 21, 25, 17, 21, 21, 25, 21, 25, 25, 29, 
        4,  8,  8, 12,  8, 12, 12, 16,  8, 12, 12, 16, 12, 16, 16, 20, 
        8, 12, 12, 16, 12, 16, 16, 20, 12, 16, 16, 20, 16, 20, 20, 24, 
        8, 12, 12, 16, 12, 16, 16, 20, 12, 16, 16, 20, 16, 20, 20, 24, 
       12, 16, 16, 20, 16, 20, 20, 24, 16, 20, 20, 24, 20, 24, 24, 28, 
        8, 12, 12, 16, 12, 16, 16, 20, 12, 16, 16, 20, 16, 20, 20, 24, 
       12, 16, 16, 20, 16, 20, 20, 24, 16, 20, 20, 24, 20, 24, 24, 28, 
       12, 16, 16, 20, 16, 20, 20, 24, 16, 20, 20, 24, 20, 24, 24, 28, 
       16, 20, 20, 24, 20, 24, 24, 28, 20, 24, 24, 28, 24, 28, 28, 32, 
	},
	{ // state 1
        7, 10,  9, 13,  8, 12, 12, 16,  7, 11, 11, 15, 11, 15, 15, 19, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
        5,  9,  9, 13,  9, 13, 13, 17,  9, 13, 13, 17, 13, 17, 17, 21, 
        9, 13, 13, 17, 13, 17, 17, 21, 13, 17, 17, 21, 17, 21, 21, 25, 
        4,  8,  8, 12,  8, 12, 12, 16,  8, 12, 12, 16, 12, 16, 16, 20, 
        8, 12, 12, 16, 12, 16, 16, 20, 12, 16, 16, 20, 16, 20, 20, 24, 
        8, 12, 12, 16, 12, 16, 16, 20, 12, 16, 16, 20, 16, 20, 20, 24, 
       12, 16, 16, 20, 16, 20, 20, 24, 16, 20, 20, 24, 20, 24, 24, 28, 
        3,  7,  7, 11,  7, 11, 11, 15,  7, 11, 11, 15, 11, 15, 15, 19, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
       11, 15, 15, 19, 15, 19, 19, 23, 15, 19, 19, 23, 19, 23, 23, 27, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
       11, 15, 15, 19, 15, 19, 19, 23, 15, 19, 19, 23, 19, 23, 23, 27, 
       11, 15, 15, 19, 15, 19, 19, 23, 15, 19, 19, 23, 19, 23, 23, 27, 
       15, 19, 19, 23, 19, 23, 23, 27, 19, 23, 23, 27, 23, 27, 27, 31, 
	},
	{ // state 2
        6,  9,  8, 12,  7, 11, 11, 15,  6, 10, 10, 14, 10, 14, 14, 18, 
        5,  9,  9, 13,  9, 13, 13, 17,  9, 13, 13, 17, 13, 17, 17, 21, 
        4,  8,  8, 12,  8, 12, 12, 16,  8, 12, 12, 16, 12, 16, 16, 20, 
        8, 12, 12, 16, 12, 16, 16, 20, 12, 16, 16, 20, 16, 20, 20, 24, 
        3,  7,  7, 11,  7, 11, 11, 15,  7, 11, 11, 15, 11, 15, 15, 19, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
       11, 15, 15, 19, 15, 19, 19, 23, 15, 19, 19, 23, 19, 23, 23, 27, 
        2,  6,  6, 10,  6, 10, 10, 14,  6, 10, 10, 14, 10, 14, 14, 18, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
       10, 14, 14, 18, 14, 18, 18, 22, 14, 18, 18, 22, 18, 22, 22, 26, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
       10, 14, 14, 18, 14, 18, 18, 22, 14, 18, 18, 22, 18, 22, 22, 26, 
       10, 14, 14, 18, 14, 18, 18, 22, 14, 18, 18, 22, 18, 22, 22, 26, 
       14, 18, 18, 22, 18, 22, 22, 26, 18, 22, 22, 26, 22, 26, 26, 30, 
	},
	{ // state 3
        5,  8,  7, 11,  6, 10, 10, 14,  5,  9,  9, 13,  9, 13, 13, 17, 
        4,  8,  8, 12,  8, 12, 12, 16,  8, 12, 12, 16, 12, 16, 16, 20, 
        3,  7,  7, 11,  7, 11, 11, 15,  7, 11, 11, 15, 11, 15, 15, 19, 
        7, 11, 11, 15, 11, 15, 15, 19, 11, 15, 15, 19, 15, 19, 19, 23, 
        2,  6,  6, 10,  6, 10, 10, 14,  6, 10, 10, 14, 10, 14, 14, 18, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
        6, 10, 10, 14, 10, 14, 14, 18, 10, 14, 14, 18, 14, 18, 18, 22, 
       10, 14, 14, 18, 14, 18, 18, 22, 14, 18, 18, 22, 18, 22, 22, 26, 
        1,  5,  5,  9,  5,  9,  9, 13,  5,  9,  9, 13,  9, 13, 13, 17, 
        5,  9,  9, 13,  9, 13, 13, 17,  9, 13, 13, 17, 13, 17, 17, 21, 
        5,  9,  9, 13,  9, 13, 13, 17,  9, 13, 13, 17, 13, 17, 17, 21, 
        9, 13, 13, 17, 13, 17, 17, 21, 13, 17, 17, 21, 17, 21, 21, 25, 
        5,  9,  9, 13,  9, 13, 13, 17,  9, 13, 13, 17, 13, 17, 17, 21, 
        9, 13, 13, 17, 13, 17, 17, 21, 13, 17, 17, 21, 17, 21, 21, 25, 
        9, 13, 13, 17, 13, 17, 17, 21, 13, 17, 17, 21, 17, 21, 21, 25, 
       13, 17, 17, 21, 17, 21, 21, 25, 17, 21, 21, 25, 21, 25, 25, 29, 
	},
};

// number of leading 0s in the descriptor desc.
extern const uint8_t VarintG8CUState[256] = {
	8,
	7,
	6,
	6,
	5,
	5,
	5,
	5,
	4,
	4,
	4,
	4,
	4,
	4,
	4,
	4,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	3,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
};

extern const uint8_t VarintG8CULengths[256][8] = {
	{8, 0, 0, 0, 0, 0, 0, 0},
	{1, 7, 0, 0, 0, 0, 0, 0},
	{2, 6, 0, 0, 0, 0, 0, 0},
	{1, 1, 6, 0, 0, 0, 0, 0},
	{3, 5, 0, 0, 0, 0, 0, 0},
	{1, 2, 5, 0, 0, 0, 0, 0},
	{2, 1, 5, 0, 0, 0, 0, 0},
	{1, 1, 1, 5, 0, 0, 0, 0},
	{4, 4, 0, 0, 0, 0, 0, 0},
	{1, 3, 4, 0, 0, 0, 0, 0},
	{2, 2, 4, 0, 0, 0, 0, 0},
	{1, 1, 2, 4, 0, 0, 0, 0},
	{3, 1, 4, 0, 0, 0, 0, 0},
	{1, 2, 1, 4, 0, 0, 0, 0},
	{2, 1, 1, 4, 0, 0, 0, 0},
	{1, 1, 1, 1, 4, 0, 0, 0},
	{5, 3, 0, 0, 0, 0, 0, 0},
	{1, 4, 3, 0, 0, 0, 0, 0},
	{2, 3, 3, 0, 0, 0, 0, 0},
	{1, 1, 3, 3, 0, 0, 0, 0},
	{3, 2, 3, 0, 0, 0, 0, 0},
	{1, 2, 2, 3, 0, 0, 0, 0},
	{2, 1, 2, 3, 0, 0, 0, 0},
	{1, 1, 1, 2, 3, 0, 0, 0},
	{4, 1, 3, 0, 0, 0, 0, 0},
	{1, 3, 1, 3, 0, 0, 0, 0},
	{2, 2, 1, 3, 0, 0, 0, 0},
	{1, 1, 2, 1, 3, 0, 0, 0},
	{3, 1, 1, 3, 0, 0, 0, 0},
	{1, 2, 1, 1, 3, 0, 0, 0},
	{2, 1, 1, 1, 3, 0, 0, 0},
	{1, 1, 1, 1, 1, 3, 0, 0},
	{6, 2, 0, 0, 0, 0, 0, 0},
	{1, 5, 2, 0, 0, 0, 0, 0},
	{2, 4, 2, 0, 0, 0, 0, 0},
	{1, 1, 4, 2, 0, 0, 0, 0},
	{3, 3, 2, 0, 0, 0, 0, 0},
	{1, 2, 3, 2, 0, 0, 0, 0},
	{2, 1, 3, 2, 0, 0, 0, 0},
	{1, 1, 1, 3, 2, 0, 0, 0},
	{4, 2, 2, 0, 0, 0, 0, 0},
	{1, 3, 2, 2, 0, 0, 0, 0},
	{2, 2, 2, 2, 0, 0, 0, 0},
	{1, 1, 2, 2, 2, 0, 0, 0},
	{3, 1, 2, 2, 0, 0, 0, 0},
	{1, 2, 1, 2, 2, 0, 0, 0},
	{2, 1, 1, 2, 2, 0, 0, 0},
	{1, 1, 1, 1, 2, 2, 0, 0},
	{5, 1, 2, 0, 0, 0, 0, 0},
	{1, 4, 1, 2, 0, 0, 0, 0},
	{2, 3, 1, 2, 0, 0, 0, 0},
	{1, 1, 3, 1, 2, 0, 0, 0},
	{3, 2, 1, 2, 0, 0, 0, 0},
	{1, 2, 2, 1, 2, 0, 0, 0},
	{2, 1, 2, 1, 2, 0, 0, 0},
	{1, 1, 1, 2, 1, 2, 0, 0},
	{4, 1, 1, 2, 0, 0, 0, 0},
	{1, 3, 1, 1, 2, 0, 0, 0},
	{2, 2, 1, 1, 2, 0, 0, 0},
	{1, 1, 2, 1, 1, 2, 0, 0},
	{3, 1, 1, 1, 2, 0, 0, 0},
	{1, 2, 1, 1, 1, 2, 0, 0},
	{2, 1, 1, 1, 1, 2, 0, 0},
	{1, 1, 1, 1, 1, 1, 2, 0},
	{7, 1, 0, 0, 0, 0, 0, 0},
	{1, 6, 1, 0, 0, 0, 0, 0},
	{2, 5, 1, 0, 0, 0, 0, 0},
	{1, 1, 5, 1, 0, 0, 0, 0},
	{3, 4, 1, 0, 0, 0, 0, 0},
	{1, 2, 4, 1, 0, 0, 0, 0},
	{2, 1, 4, 1, 0, 0, 0, 0},
	{1, 1, 1, 4, 1, 0, 0, 0},
	{4, 3, 1, 0, 0, 0, 0, 0},
	{1, 3, 3, 1, 0, 0, 0, 0},
	{2, 2, 3, 1, 0, 0, 0, 0},
	{1, 1, 2, 3, 1, 0, 0, 0},
	{3, 1, 3, 1, 0, 0, 0, 0},
	{1, 2, 1, 3, 1, 0, 0, 0},
	{2, 1, 1, 3, 1, 0, 0, 0},
	{1, 1, 1, 1, 3, 1, 0, 0},
	{5, 2, 1, 0, 0, 0, 0, 0},
	{1, 4, 2, 1, 0, 0, 0, 0},
	{2, 3, 2, 1, 0, 0, 0, 0},
	{1, 1, 3, 2, 1, 0, 0, 0},
	{3, 2, 2, 1, 0, 0, 0, 0},
	{1, 2, 2, 2, 1, 0, 0, 0},
	{2, 1, 2, 2, 1, 0, 0, 0},
	{1, 1, 1, 2, 2, 1, 0, 0},
	{4, 1, 2, 1, 0, 0, 0, 0},
	{1, 3, 1, 2, 1, 0, 0, 0},
	{2, 2, 1, 2, 1, 0, 0, 0},
	{1, 1, 2, 1, 2, 1, 0, 0},
	{3, 1, 1, 2, 1, 0, 0, 0},
	{1, 2, 1, 1, 2, 1, 0, 0},
	{2, 1, 1, 1, 2, 1, 0, 0},
	{1, 1, 1, 1, 1, 2, 1, 0},
	{6, 1, 1, 0, 0, 0, 0, 0},
	{1, 5, 1, 1, 0, 0, 0, 0},
	{2, 4, 1, 1, 0, 0, 0, 0},
	{1, 1, 4, 1, 1, 0, 0, 0},
	{3, 3, 1, 1, 0, 0, 0, 0},
	{1, 2, 3, 1, 1, 0, 0, 0},
	{2, 1, 3, 1, 1, 0, 0, 0},
	{1, 1, 1, 3, 1, 1, 0, 0},
	{4, 2, 1, 1, 0, 0, 0, 0},
	{1, 3, 2, 1, 1, 0, 0, 0},
	{2, 2, 2, 1, 1, 0, 0, 0},
	{1, 1, 2, 2, 1, 1, 0, 0},
	{3, 1, 2, 1, 1, 0, 0, 0},
	{1, 2, 1, 2, 1, 1, 0, 0},
	{2, 1, 1, 2, 1, 1, 0, 0},
	{1, 1, 1, 1, 2, 1, 1, 0},
	{5, 1, 1, 1, 0, 0, 0, 0},
	{1, 4, 1, 1, 1, 0, 0, 0},
	{2, 3, 1, 1, 1, 0, 0, 0},
	{1, 1, 3, 1, 1, 1, 0, 0},
	{3, 2, 1, 1, 1, 0, 0, 0},
	{1, 2, 2, 1, 1, 1, 0, 0},
	{2, 1, 2, 1, 1, 1, 0, 0},
	{1, 1, 1, 2, 1, 1, 1, 0},
	{4, 1, 1, 1, 1, 0, 0, 0},
	{1, 3, 1, 1, 1, 1, 0, 0},
	{2, 2, 1, 1, 1, 1, 0, 0},
	{1, 1, 2, 1, 1, 1, 1, 0},
	{3, 1, 1, 1, 1, 1, 0, 0},
	{1, 2, 1, 1, 1, 1, 1, 0},
	{2, 1, 1, 1, 1, 1, 1, 0},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{8, 0, 0, 0, 0, 0, 0, 0},
	{1, 7, 0, 0, 0, 0, 0, 0},
	{2, 6, 0, 0, 0, 0, 0, 0},
	{1, 1, 6, 0, 0, 0, 0, 0},
	{3, 5, 0, 0, 0, 0, 0, 0},
	{1, 2, 5, 0, 0, 0, 0, 0},
	{2, 1, 5, 0, 0, 0, 0, 0},
	{1, 1, 1, 5, 0, 0, 0, 0},
	{4, 4, 0, 0, 0, 0, 0, 0},
	{1, 3, 4, 0, 0, 0, 0, 0},
	{2, 2, 4, 0, 0, 0, 0, 0},
	{1, 1, 2, 4, 0, 0, 0, 0},
	{3, 1, 4, 0, 0, 0, 0, 0},
	{1, 2, 1, 4, 0, 0, 0, 0},
	{2, 1, 1, 4, 0, 0, 0, 0},
	{1, 1, 1, 1, 4, 0, 0, 0},
	{5, 3, 0, 0, 0, 0, 0, 0},
	{1, 4, 3, 0, 0, 0, 0, 0},
	{2, 3, 3, 0, 0, 0, 0, 0},
	{1, 1, 3, 3, 0, 0, 0, 0},
	{3, 2, 3, 0, 0, 0, 0, 0},
	{1, 2, 2, 3, 0, 0, 0, 0},
	{2, 1, 2, 3, 0, 0, 0, 0},
	{1, 1, 1, 2, 3, 0, 0, 0},
	{4, 1, 3, 0, 0, 0, 0, 0},
	{1, 3, 1, 3, 0, 0, 0, 0},
	{2, 2, 1, 3, 0, 0, 0, 0},
	{1, 1, 2, 1, 3, 0, 0, 0},
	{3, 1, 1, 3, 0, 0, 0, 0},
	{1, 2, 1, 1, 3, 0, 0, 0},
	{2, 1, 1, 1, 3, 0, 0, 0},
	{1, 1, 1, 1, 1, 3, 0, 0},
	{6, 2, 0, 0, 0, 0, 0, 0},
	{1, 5, 2, 0, 0, 0, 0, 0},
	{2, 4, 2, 0, 0, 0, 0, 0},
	{1, 1, 4, 2, 0, 0, 0, 0},
	{3, 3, 2, 0, 0, 0, 0, 0},
	{1, 2, 3, 2, 0, 0, 0, 0},
	{2, 1, 3, 2, 0, 0, 0, 0},
	{1, 1, 1, 3, 2, 0, 0, 0},
	{4, 2, 2, 0, 0, 0, 0, 0},
	{1, 3, 2, 2, 0, 0, 0, 0},
	{2, 2, 2, 2, 0, 0, 0, 0},
	{1, 1, 2, 2, 2, 0, 0, 0},
	{3, 1, 2, 2, 0, 0, 0, 0},
	{1, 2, 1, 2, 2, 0, 0, 0},
	{2, 1, 1, 2, 2, 0, 0, 0},
	{1, 1, 1, 1, 2, 2, 0, 0},
	{5, 1, 2, 0, 0, 0, 0, 0},
	{1, 4, 1, 2, 0, 0, 0, 0},
	{2, 3, 1, 2, 0, 0, 0, 0},
	{1, 1, 3, 1, 2, 0, 0, 0},
	{3, 2, 1, 2, 0, 0, 0, 0},
	{1, 2, 2, 1, 2, 0, 0, 0},
	{2, 1, 2, 1, 2, 0, 0, 0},
	{1, 1, 1, 2, 1, 2, 0, 0},
	{4, 1, 1, 2, 0, 0, 0, 0},
	{1, 3, 1, 1, 2, 0, 0, 0},
	{2, 2, 1, 1, 2, 0, 0, 0},
	{1, 1, 2, 1, 1, 2, 0, 0},
	{3, 1, 1, 1, 2, 0, 0, 0},
	{1, 2, 1, 1, 1, 2, 0, 0},
	{2, 1, 1, 1, 1, 2, 0, 0},
	{1, 1, 1, 1, 1, 1, 2, 0},
	{7, 1, 0, 0, 0, 0, 0, 0},
	{1, 6, 1, 0, 0, 0, 0, 0},
	{2, 5, 1, 0, 0, 0, 0, 0},
	{1, 1, 5, 1, 0, 0, 0, 0},
	{3, 4, 1, 0, 0, 0, 0, 0},
	{1, 2, 4, 1, 0, 0, 0, 0},
	{2, 1, 4, 1, 0, 0, 0, 0},
	{1, 1, 1, 4, 1, 0, 0, 0},
	{4, 3, 1, 0, 0, 0, 0, 0},
	{1, 3, 3, 1, 0, 0, 0, 0},
	{2, 2, 3, 1, 0, 0, 0, 0},
	{1, 1, 2, 3, 1, 0, 0, 0},
	{3, 1, 3, 1, 0, 0, 0, 0},
	{1, 2, 1, 3, 1, 0, 0, 0},
	{2, 1, 1, 3, 1, 0, 0, 0},
	{1, 1, 1, 1, 3, 1, 0, 0},
	{5, 2, 1, 0, 0, 0, 0, 0},
	{1, 4, 2, 1, 0, 0, 0, 0},
	{2, 3, 2, 1, 0, 0, 0, 0},
	{1, 1, 3, 2, 1, 0, 0, 0},
	{3, 2, 2, 1, 0, 0, 0, 0},
	{1, 2, 2, 2, 1, 0, 0, 0},
	{2, 1, 2, 2, 1, 0, 0, 0},
	{1, 1, 1, 2, 2, 1, 0, 0},
	{4, 1, 2, 1, 0, 0, 0, 0},
	{1, 3, 1, 2, 1, 0, 0, 0},
	{2, 2, 1, 2, 1, 0, 0, 0},
	{1, 1, 2, 1, 2, 1, 0, 0},
	{3, 1, 1, 2, 1, 0, 0, 0},
	{1, 2, 1, 1, 2, 1, 0, 0},
	{2, 1, 1, 1, 2, 1, 0, 0},
	{1, 1, 1, 1, 1, 2, 1, 0},
	{6, 1, 1, 0, 0, 0, 0, 0},
	{1, 5, 1, 1, 0, 0, 0, 0},
	{2, 4, 1, 1, 0, 0, 0, 0},
	{1, 1, 4, 1, 1, 0, 0, 0},
	{3, 3, 1, 1, 0, 0, 0, 0},
	{1, 2, 3, 1, 1, 0, 0, 0},
	{2, 1, 3, 1, 1, 0, 0, 0},
	{1, 1, 1, 3, 1, 1, 0, 0},
	{4, 2, 1, 1, 0, 0, 0, 0},
	{1, 3, 2, 1, 1, 0, 0, 0},
	{2, 2, 2, 1, 1, 0, 0, 0},
	{1, 1, 2, 2, 1, 1, 0, 0},
	{3, 1, 2, 1, 1, 0, 0, 0},
	{1, 2, 1, 2, 1, 1, 0, 0},
	{2, 1, 1, 2, 1, 1, 0, 0},
	{1, 1, 1, 1, 2, 1, 1, 0},
	{5, 1, 1, 1, 0, 0, 0, 0},
	{1, 4, 1, 1, 1, 0, 0, 0},
	{2, 3, 1, 1, 1, 0, 0, 0},
	{1, 1, 3, 1, 1, 1, 0, 0},
	{3, 2, 1, 1, 1, 0, 0, 0},
	{1, 2, 2, 1, 1, 1, 0, 0},
	{2, 1, 2, 1, 1, 1, 0, 0},
	{1, 1, 1, 2, 1, 1, 1, 0},
	{4, 1, 1, 1, 1, 0, 0, 0},
	{1, 3, 1, 1, 1, 1, 0, 0},
	{2, 2, 1, 1, 1, 1, 0, 0},
	{1, 1, 2, 1, 1, 1, 1, 0},
	{3, 1, 1, 1, 1, 1, 0, 0},
	{1, 2, 1, 1, 1, 1, 1, 0},
	{2, 1, 1, 1, 1, 1, 1, 0},
	{1, 1, 1, 1, 1, 1, 1, 1},
};

} // VarintTables