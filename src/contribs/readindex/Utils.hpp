
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <chrono>
#include <stdint.h>

typedef std::chrono::time_point<std::chrono::system_clock> t_clock;
typedef std::chrono::duration<double> t_duration;

inline t_clock fetchClock()
{
	return std::chrono::system_clock::now();
}

inline void getMiSecs(t_clock start, t_clock end, const char *prefix)
{
	std::cout << prefix << " is "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
	    << "ms." << std::endl;
}

inline void getSecs(t_clock start, t_clock end, const char *prefix)
{
	std::cout << prefix << " is "
	    << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
	    << "s." << std::endl;
}

//inline void getCumulate(const t_clock accumulate)
//{
//	std::cout << "Time is "
//			<< std::chrono::duration_cast<std::chrono::milliseconds>(accumulate - 0).count()
//			<< "ms." << std::endl;
//}

template<typename IntType1, typename IntType2>
inline IntType1 ceil_div(IntType1 dividend, IntType2 divisor)
{
	IntType1 d = IntType1(divisor);
	return IntType1(dividend + d - 1) / d;
}

template<typename IntType>
inline IntType log_bit(IntType base)
{
	IntType val = base, log = 0;
	while (val >>= 1)
		log++;
	return log;
}


#endif /* UTILS_HPP_ */
