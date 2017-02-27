# Install script for directory: /home/zr/nblucene/src/ext

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/version.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/memory_order.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/checked_delete.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/shared_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/current_function.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/digitalmars.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/compaq_cxx.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/metrowerks.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/sgi_mipspro.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/vacpp.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/codegear.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/gcc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/sunpro_cc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/intel.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/pgi.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/borland.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/greenhills.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/comeau.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/common_edg.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/hp_acc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/gcc_xml.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/visualc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/mpw.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/compiler" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/compiler/kai.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/suffix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/select_compiler_config.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/libstdcpp3.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/sgi.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/vacpp.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/modena.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/roguewave.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/msl.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/stlport.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/dinkumware.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/stdlib" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/stdlib/libcomo.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/select_stdlib_config.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/requires_threads.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/abi" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/abi/borland_suffix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/abi" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/abi/msvc_suffix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/abi" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/abi/borland_prefix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/abi" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/abi/msvc_prefix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/warning_disable.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/abi_suffix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/user.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/auto_link.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/select_platform_config.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/linux.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/aix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/macos.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/win32.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/bsd.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/qnxnto.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/hpux.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/cygwin.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/amigaos.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/irix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/solaris.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/beos.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/platform" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/platform/vxworks.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/abi_prefix.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/no_tr1" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/no_tr1/cmath.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/no_tr1" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/no_tr1/complex.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/no_tr1" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/no_tr1/memory.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/no_tr1" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/no_tr1/functional.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config/no_tr1" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/no_tr1/utility.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/config" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config/posix_features.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/throw_exception.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/config.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/assert.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/weak_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/shared_array.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/shared_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/bad_weak_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/enable_shared_from_this.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/enable_shared_from_this2.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/scoped_array.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/scoped_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/intrusive_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/make_shared.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/lwm_pthreads.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count_gcc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_cw_ppc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_nt.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/lwm_win32_cs.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_has_sync.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/shared_ptr_nmt.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_cw_x86.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_w32.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock_gcc_arm.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_impl.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_convertible.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_sync.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count_sync.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/operator_bool.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock_w32.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock_sync.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/quick_allocator.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_acc_ia64.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_pt.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/shared_count.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count_solaris.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_gcc_sparc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count_gcc_x86.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/shared_array_nmt.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/lightweight_mutex.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock_nt.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_gcc_ia64.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_gcc_mips.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count_win32.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_gcc_ppc.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/yield_k.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/atomic_count_pthreads.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_spin.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_solaris.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base_gcc_x86.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock_pool.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/spinlock_pt.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/lwm_nop.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/smart_ptr/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/smart_ptr/detail/sp_counted_base.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/lcast_precision.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/interlocked.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/catch_exceptions.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/iterator.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/endian.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/identifier.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/is_incrementable.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/is_xxx.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/reference_content.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/container_fwd.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/named_template_params.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/lightweight_thread.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/limits.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/binary_search.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/ob_compressed_pair.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/scoped_enum_emulation.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/templated_streams.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/dynamic_bitset.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/indirect_traits.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/has_default_constructor.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/workaround.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/quick_allocator.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/allocator_utilities.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/algorithm.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/sp_typeinfo.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/numeric_traits.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/is_function_ref_tester.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/no_exceptions_support.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/ob_call_traits.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/none_t.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/lightweight_mutex.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/call_traits.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/select_type.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/utf8_codecvt_facet.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/atomic_count.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/compressed_pair.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/detail/lightweight_test.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_nested_exception.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/to_string.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/exception.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/enable_error_info.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/all.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/info_tuple.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_file_open_mode.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_type_info_name.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/current_exception_cast.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/get_error_info.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/enable_current_exception.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_file_handle.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/error_info.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_errno.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/diagnostic_information.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_api_function.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/info.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/to_string_stub.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_at_line.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/errinfo_file_name.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/detail/exception_ptr.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/detail/type_info.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/detail/object_hex_dump.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/detail/error_info_impl.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/detail/attribute_noreturn.hpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "development")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CLucene/ext/boost/exception/detail" TYPE FILE FILES "/home/zr/nblucene/src/ext/boost/exception/detail/is_output_streamable.hpp")
endif()

