AC_PREREQ([2.63])
AC_INIT([MASA-Serial], [1.0.1.1024], [edans.sandes@gmail.com], [masa-serial], [https://github.com/edanssandes/masa-serial])
AC_CONFIG_AUX_DIR(admin)
AM_INIT_AUTOMAKE([-Wall -Werror])
AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_FILES([Makefile])

AC_CONFIG_SUBDIRS([libs/masa-core])

AC_PROG_CXX

LIBMASA_PATH="libs/masa-core"

AC_CHECK_FILES(
	[$LIBMASA_PATH $LIBMASA_PATH/src],
	,
	AC_MSG_ERROR([
		MASA-Core source directory was not found: $LIBMASA_PATH/src.
	])
)


OPTIMIZATION="-O3"

AC_ARG_ENABLE([debug],
	AC_HELP_STRING([--enable-debug],[enable GDB debug]),
	[DEBUG="-ggdb"; OPTIMIZATION=""]
)

#default: 64bit
AC_ARG_ENABLE([32bit],
	AC_HELP_STRING([--enable-32bit],[force compilation in 32 bit mode.]),
	[ARCH="386"; PLATFORM="-m32";
		AC_MSG_NOTICE([Enabling 32bit compilation])
	],
	[ARCH="x86_64"; PLATFORM="-m64";
		AC_MSG_NOTICE([Enabling 64bit compilation])	
	]
)


CXXFLAGS="-DUNIX $COMMONFLAGS $PLATFORM $OPTIMIZATION $DEBUG -I$LIBMASA_PATH/src"

AC_SUBST(LIBMASA_PATH)
AC_SUBST(CXXFLAGS)

AC_OUTPUT
