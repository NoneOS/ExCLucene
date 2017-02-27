#include <stdint.h>
#include <x86intrin.h>

namespace VarintTables {

extern const __m128i VarintG8IUSSEMasks[256][2] = {
	{
		{0xffffffffffffffffU, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffffffffffff00U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffffffffff0100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffffffff020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffffffffffff02U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffffff03020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffffffffff0302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffffffffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffffffffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
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
		{0xff040302ffff0100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffffffff040302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0403ff020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffffffffff0403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffffffffff0403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0403ffffff02U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff0403020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffffffffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffffffffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff04ffff0302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffffffffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff04ffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff04ffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffffffffffff04U, 0xffffffffffffffffU}
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
		{0xff050403ff020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffffffff050403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffffffff050403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xff050403ffffff02U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff050403020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffffffffff0504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffffffffff0504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0504ffff0302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffffffffff0504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffff0504ffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffff0504ffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffffffffff0504U, 0xffffffffffffffffU}
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
		{0xff040302ffff0100U, 0xffffffffffffff05U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff05ff040302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0403ff020100U, 0xffffffffffffff05U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff05ffff0403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff05ffff0403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0403ffffff02U},
		{0xffffffffffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff0403020100U, 0xffffffffffffff05U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffff05ffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffff05ffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff04ffff0302U},
		{0xffffffffffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffff05ffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff04ffffff03U},
		{0xffffffffffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff04ffffff03U},
		{0xffffffffffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffff05ffffff04U, 0xffffffffffffffffU}
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
		{0xff06050403020100U, 0xffffffffffffffffU},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffffffff060504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffffffff060504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xff060504ffff0302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffffffff060504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xff060504ffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xff060504ffffff03U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffffffff060504U, 0xffffffffffffffffU}
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
		{0xff040302ffff0100U, 0xffffffffffff0605U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0605ff040302U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0403ff020100U, 0xffffffffffff0605U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffff0605ffff0403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffff0605ffff0403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0403ffffff02U},
		{0xffffffffffff0605U, 0xffffffffffffffffU}
	},
	{
		{0xffffff0403020100U, 0xffffffffffff0605U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffff0605ffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffff0605ffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff04ffff0302U},
		{0xffffffffffff0605U, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffff0605ffffff04U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff04ffffff03U},
		{0xffffffffffff0605U, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff04ffffff03U},
		{0xffffffffffff0605U, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffff0605ffffff04U, 0xffffffffffffffffU}
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
		{0xff050403ff020100U, 0xffffffffffffff06U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff06ff050403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff06ff050403U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xff050403ffffff02U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffff050403020100U, 0xffffffffffffff06U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffff06ffff0504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffff06ffff0504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0504ffff0302U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffff06ffff0504U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffff0504ffffff03U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffff0504ffffff03U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffff06ffff0504U, 0xffffffffffffffffU}
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
		{0xff040302ffff0100U, 0xffffff06ffffff05U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff05ff040302U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffff0403ff020100U, 0xffffff06ffffff05U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff05ffff0403U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff05ffff0403U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffff0403ffffff02U},
		{0xffffff06ffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff0403020100U, 0xffffff06ffffff05U},
		{0xffffffffffffffffU, 0xffffffffffffffffU}
	},
	{
		{0xff030201ffffff00U, 0xffffff05ffffff04U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffff0302ffff0100U, 0xffffff05ffffff04U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff04ffff0302U},
		{0xffffff06ffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff03ff020100U, 0xffffff05ffffff04U},
		{0xffffffffffffff06U, 0xffffffffffffffffU}
	},
	{
		{0xffff0201ffffff00U, 0xffffff04ffffff03U},
		{0xffffff06ffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff02ffff0100U, 0xffffff04ffffff03U},
		{0xffffff06ffffff05U, 0xffffffffffffffffU}
	},
	{
		{0xffffff01ffffff00U, 0xffffff03ffffff02U},
		{0xffffff05ffffff04U, 0xffffffffffffff06U}
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
};

// number of integers whose encoding is complete in the group described by the descriptor desc.
extern const uint8_t VarintG8IUOutputOffset[256] = {
	0,
	1,
	1,
	2,
	1,
	2,
	2,
	3,
	1,
	2,
	2,
	3,
	2,
	3,
	3,
	4,
	1,
	2,
	2,
	3,
	2,
	3,
	3,
	4,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	1,
	2,
	2,
	3,
	2,
	3,
	3,
	4,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	1,
	2,
	2,
	3,
	2,
	3,
	3,
	4,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	4,
	5,
	5,
	6,
	5,
	6,
	6,
	7,
	1,
	2,
	2,
	3,
	2,
	3,
	3,
	4,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	4,
	5,
	5,
	6,
	5,
	6,
	6,
	7,
	2,
	3,
	3,
	4,
	3,
	4,
	4,
	5,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	4,
	5,
	5,
	6,
	5,
	6,
	6,
	7,
	3,
	4,
	4,
	5,
	4,
	5,
	5,
	6,
	4,
	5,
	5,
	6,
	5,
	6,
	6,
	7,
	4,
	5,
	5,
	6,
	5,
	6,
	6,
	7,
	5,
	6,
	6,
	7,
	6,
	7,
	7,
	8,
};

extern const uint8_t VarintG8IULengths[256][8] = {
	{0, 0, 0, 0, 0, 0, 0, 0},
	{1, 0, 0, 0, 0, 0, 0, 0},
	{2, 0, 0, 0, 0, 0, 0, 0},
	{1, 1, 0, 0, 0, 0, 0, 0},
	{3, 0, 0, 0, 0, 0, 0, 0},
	{1, 2, 0, 0, 0, 0, 0, 0},
	{2, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 1, 0, 0, 0, 0, 0},
	{4, 0, 0, 0, 0, 0, 0, 0},
	{1, 3, 0, 0, 0, 0, 0, 0},
	{2, 2, 0, 0, 0, 0, 0, 0},
	{1, 1, 2, 0, 0, 0, 0, 0},
	{3, 1, 0, 0, 0, 0, 0, 0},
	{1, 2, 1, 0, 0, 0, 0, 0},
	{2, 1, 1, 0, 0, 0, 0, 0},
	{1, 1, 1, 1, 0, 0, 0, 0},
	{5, 0, 0, 0, 0, 0, 0, 0},
	{1, 4, 0, 0, 0, 0, 0, 0},
	{2, 3, 0, 0, 0, 0, 0, 0},
	{1, 1, 3, 0, 0, 0, 0, 0},
	{3, 2, 0, 0, 0, 0, 0, 0},
	{1, 2, 2, 0, 0, 0, 0, 0},
	{2, 1, 2, 0, 0, 0, 0, 0},
	{1, 1, 1, 2, 0, 0, 0, 0},
	{4, 1, 0, 0, 0, 0, 0, 0},
	{1, 3, 1, 0, 0, 0, 0, 0},
	{2, 2, 1, 0, 0, 0, 0, 0},
	{1, 1, 2, 1, 0, 0, 0, 0},
	{3, 1, 1, 0, 0, 0, 0, 0},
	{1, 2, 1, 1, 0, 0, 0, 0},
	{2, 1, 1, 1, 0, 0, 0, 0},
	{1, 1, 1, 1, 1, 0, 0, 0},
	{6, 0, 0, 0, 0, 0, 0, 0},
	{1, 5, 0, 0, 0, 0, 0, 0},
	{2, 4, 0, 0, 0, 0, 0, 0},
	{1, 1, 4, 0, 0, 0, 0, 0},
	{3, 3, 0, 0, 0, 0, 0, 0},
	{1, 2, 3, 0, 0, 0, 0, 0},
	{2, 1, 3, 0, 0, 0, 0, 0},
	{1, 1, 1, 3, 0, 0, 0, 0},
	{4, 2, 0, 0, 0, 0, 0, 0},
	{1, 3, 2, 0, 0, 0, 0, 0},
	{2, 2, 2, 0, 0, 0, 0, 0},
	{1, 1, 2, 2, 0, 0, 0, 0},
	{3, 1, 2, 0, 0, 0, 0, 0},
	{1, 2, 1, 2, 0, 0, 0, 0},
	{2, 1, 1, 2, 0, 0, 0, 0},
	{1, 1, 1, 1, 2, 0, 0, 0},
	{5, 1, 0, 0, 0, 0, 0, 0},
	{1, 4, 1, 0, 0, 0, 0, 0},
	{2, 3, 1, 0, 0, 0, 0, 0},
	{1, 1, 3, 1, 0, 0, 0, 0},
	{3, 2, 1, 0, 0, 0, 0, 0},
	{1, 2, 2, 1, 0, 0, 0, 0},
	{2, 1, 2, 1, 0, 0, 0, 0},
	{1, 1, 1, 2, 1, 0, 0, 0},
	{4, 1, 1, 0, 0, 0, 0, 0},
	{1, 3, 1, 1, 0, 0, 0, 0},
	{2, 2, 1, 1, 0, 0, 0, 0},
	{1, 1, 2, 1, 1, 0, 0, 0},
	{3, 1, 1, 1, 0, 0, 0, 0},
	{1, 2, 1, 1, 1, 0, 0, 0},
	{2, 1, 1, 1, 1, 0, 0, 0},
	{1, 1, 1, 1, 1, 1, 0, 0},
	{7, 0, 0, 0, 0, 0, 0, 0},
	{1, 6, 0, 0, 0, 0, 0, 0},
	{2, 5, 0, 0, 0, 0, 0, 0},
	{1, 1, 5, 0, 0, 0, 0, 0},
	{3, 4, 0, 0, 0, 0, 0, 0},
	{1, 2, 4, 0, 0, 0, 0, 0},
	{2, 1, 4, 0, 0, 0, 0, 0},
	{1, 1, 1, 4, 0, 0, 0, 0},
	{4, 3, 0, 0, 0, 0, 0, 0},
	{1, 3, 3, 0, 0, 0, 0, 0},
	{2, 2, 3, 0, 0, 0, 0, 0},
	{1, 1, 2, 3, 0, 0, 0, 0},
	{3, 1, 3, 0, 0, 0, 0, 0},
	{1, 2, 1, 3, 0, 0, 0, 0},
	{2, 1, 1, 3, 0, 0, 0, 0},
	{1, 1, 1, 1, 3, 0, 0, 0},
	{5, 2, 0, 0, 0, 0, 0, 0},
	{1, 4, 2, 0, 0, 0, 0, 0},
	{2, 3, 2, 0, 0, 0, 0, 0},
	{1, 1, 3, 2, 0, 0, 0, 0},
	{3, 2, 2, 0, 0, 0, 0, 0},
	{1, 2, 2, 2, 0, 0, 0, 0},
	{2, 1, 2, 2, 0, 0, 0, 0},
	{1, 1, 1, 2, 2, 0, 0, 0},
	{4, 1, 2, 0, 0, 0, 0, 0},
	{1, 3, 1, 2, 0, 0, 0, 0},
	{2, 2, 1, 2, 0, 0, 0, 0},
	{1, 1, 2, 1, 2, 0, 0, 0},
	{3, 1, 1, 2, 0, 0, 0, 0},
	{1, 2, 1, 1, 2, 0, 0, 0},
	{2, 1, 1, 1, 2, 0, 0, 0},
	{1, 1, 1, 1, 1, 2, 0, 0},
	{6, 1, 0, 0, 0, 0, 0, 0},
	{1, 5, 1, 0, 0, 0, 0, 0},
	{2, 4, 1, 0, 0, 0, 0, 0},
	{1, 1, 4, 1, 0, 0, 0, 0},
	{3, 3, 1, 0, 0, 0, 0, 0},
	{1, 2, 3, 1, 0, 0, 0, 0},
	{2, 1, 3, 1, 0, 0, 0, 0},
	{1, 1, 1, 3, 1, 0, 0, 0},
	{4, 2, 1, 0, 0, 0, 0, 0},
	{1, 3, 2, 1, 0, 0, 0, 0},
	{2, 2, 2, 1, 0, 0, 0, 0},
	{1, 1, 2, 2, 1, 0, 0, 0},
	{3, 1, 2, 1, 0, 0, 0, 0},
	{1, 2, 1, 2, 1, 0, 0, 0},
	{2, 1, 1, 2, 1, 0, 0, 0},
	{1, 1, 1, 1, 2, 1, 0, 0},
	{5, 1, 1, 0, 0, 0, 0, 0},
	{1, 4, 1, 1, 0, 0, 0, 0},
	{2, 3, 1, 1, 0, 0, 0, 0},
	{1, 1, 3, 1, 1, 0, 0, 0},
	{3, 2, 1, 1, 0, 0, 0, 0},
	{1, 2, 2, 1, 1, 0, 0, 0},
	{2, 1, 2, 1, 1, 0, 0, 0},
	{1, 1, 1, 2, 1, 1, 0, 0},
	{4, 1, 1, 1, 0, 0, 0, 0},
	{1, 3, 1, 1, 1, 0, 0, 0},
	{2, 2, 1, 1, 1, 0, 0, 0},
	{1, 1, 2, 1, 1, 1, 0, 0},
	{3, 1, 1, 1, 1, 0, 0, 0},
	{1, 2, 1, 1, 1, 1, 0, 0},
	{2, 1, 1, 1, 1, 1, 0, 0},
	{1, 1, 1, 1, 1, 1, 1, 0},
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

extern const uint32_t kMask[8] = {
	0xff, 0xffff, 0xffffff, 0xffffffff, 0x00, 0x00, 0x00, 0x00
};


} // namespace VarintTables
