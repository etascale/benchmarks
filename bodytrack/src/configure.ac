AC_INIT([Body Tracking Benchmark], [1.0], [cbienia@cs.princeton.edu])
AC_CONFIG_SRCDIR([TrackingBenchmark/main.cpp])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([-Wall -Werror])

#AM_PROG_AR
m4_pattern_allow([AM_PROG_AR])
AM_PROG_AR

# Arguments
AC_ARG_ENABLE([gprof], AC_HELP_STRING([--enable-gprof], [add profiling code for gprof @<:@default=no@:>@]), [], [enable_gprof=no])
AC_ARG_ENABLE([threads], AC_HELP_STRING([--enable-threads], [enable conventional threading @<:@default=no@:>@]), [], [enable_threads=no])
AC_ARG_ENABLE([openmp], AC_HELP_STRING([--enable-openmp], [enable OpenMP threading @<:@default=no@:>@]), [], [enable_openmp=no])
AC_ARG_ENABLE([tbb], AC_HELP_STRING([--enable-tbb], [enable Intel TBB threading @<:@default=no@:>@]), [], [enable_tbb=no])
AC_ARG_ENABLE([ompss], AC_HELP_STRING([--enable-ompss], [enable OmpSs threading @<:@default=no@:>@]), [], [enable_ompss=no])
AC_ARG_ENABLE([ompss_instr], AC_HELP_STRING([--enable-ompss_instr], [enable instrumented OmpSs threading @<:@default=no@:>@]), [], [enable_ompss_instr=no])

# Checks for programs.
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET

# Checks for libraries.
if test "x$enable_threads" == "xyes"; then
  AC_HAVE_LIBRARY([pthread], [], [AC_MSG_ERROR([Unable to find the pthread library.])])
fi

if test "x$enable_tbb" == "xyes"; then
  AC_HAVE_LIBRARY([tbb], [], [AC_MSG_ERROR([Unable to find the TBB library.])])
fi

# Checks for header files.
if test "x$enable_threads" == "xyes"; then
  AC_CHECK_HEADER([pthread.h], [], [AC_MSG_ERROR([Unable to find <pthread.h> header file.])])
fi
AC_CHECK_HEADERS([stdint.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDC
AC_HEADER_STDBOOL
AC_HEADER_TIME
AC_C_CONST
AC_C_INLINE

# Checks for library functions.
#AC_FUNC_MALLOC
AC_CHECK_FUNCS([memset sqrt])

  AC_SUBST(VERSION, serial)

# Process arguments
if test "x$enable_gprof" == "xyes"; then
  CXXFLAGS="$CXXFLAGS -pg"
fi

if test "x$enable_threads" == "xyes"; then
  AC_DEFINE([USE_THREADS], [1], [Define to 1 to enable conventional threading])
    CXXFLAGS="$CXXFLAGS -fexceptions"
    AC_SUBST(VERSION, pthreads)
fi
AM_CONDITIONAL([ENABLE_THREADS], [test "x$enable_threads" == "xyes"])
AC_SUBST([ENABLE_THREADS])

if test "x$enable_openmp" == "xyes"; then
  AC_DEFINE([USE_OPENMP], [1], [Define to 1 to enable OpenMP])
  CXXFLAGS="$CXXFLAGS -fopenmp -DSCHED_POLICY=static"
    AC_SUBST(VERSION, omp2)
fi
AM_CONDITIONAL([ENABLE_OPENMP], [test "x$enable_openmp" == "xyes"])
AC_SUBST([ENABLE_OPENMP])

if test "x$enable_tbb" == "xyes"; then
  AC_DEFINE([USE_TBB], [1], [Define to 1 to enable Intel TBB threading])
  AC_SUBST(VERSION, tbb)
fi
AM_CONDITIONAL([ENABLE_TBB], [test "x$enable_tbb" == "xyes"])
AC_SUBST([ENABLE_TBB])

if test "x$enable_ompss" == "xyes"; then
  AC_DEFINE([USE_OMPSS], [1], [Define to 1 to enable OmpSs])
  CXXFLAGS="$CXXFLAGS --ompss"
  CXX=mcxx
  AC_SUBST(VERSION, ompss)
fi
AM_CONDITIONAL([ENABLE_OMPSS], [test "x$enable_ompss" == "xyes"])
AC_SUBST([ENABLE_OMPSS])

if test "x$enable_ompss_instr" == "xyes"; then
  AC_DEFINE([USE_OMPSS], [1], [Define to 1 to enable instrumented OmpSs])
  CXXFLAGS="$CXXFLAGS --instrument"
  AC_SUBST(VERSION, ompss_instr)
fi

# Extra arguments
CXXFLAGS="$CXXFLAGS -Wall"

# Never warn about unknown pragmas
CXXFLAGS="$CXXFLAGS -Wno-unknown-pragmas"

# Make sure VPATH is always defined
#if test "x$VPATH" == "x"; then
#  VPATH="."
#  AC_SUBST([VPATH])
#fi

# Finish up
AC_CONFIG_FILES([Makefile
                 FlexImageLib/Makefile
                 TrackingBenchmark/Makefile
                 TrackingBenchmark/threads/Makefile])
AC_OUTPUT
