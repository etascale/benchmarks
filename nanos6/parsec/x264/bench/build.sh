VERSION=$1

export CFLAGS=" -O2 -g -funroll-loops -fprefetch-loop-arrays ${PORTABILITY_FLAGS}"
export CXXFLAGS="-O2 -g -funroll-loops -fprefetch-loop-arrays -fpermissive -fno-exceptions ${PORTABILITY_FLAGS}"

if [ ${VERSION} = "serial" ]; then
	cd src
	rm .depend
	echo -e "\033[32mConfiguring environment\033[m"
	status=$( ./configure --disable-pthread --disable-asm --prefix=${ROOT}/x264 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mClean Failed!\033[m"
	fi 
	echo -e "\033[32mCompiling ${VERSION} version\033[m"
	status=$( make version=${VERSION} 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mCompilation Failed!\033[m"
		
	fi 
	echo -e "\033[32mInstalling version\033[m"
	status=$( make version=${VERSION} install 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mInstallation Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mCleaning Failed!\033[m"
		
	fi 
	cd ..
	echo -e "\033[32mDone!\033[m"
elif [ ${VERSION} = "pthreads" ]; then 
	cd src
	rm .depend
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --disable-asm --prefix=${ROOT}/x264 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mClean Failed!\033[m"
		
	fi 
	echo -e "\033[32mCompiling ${VERSION} version\033[m"
	status=$( make version=${VERSION} 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mCompilation Failed!\033[m"
		
	fi 
	echo -e "\033[32mInstalling version\033[m"
	status=$( make version=${VERSION} install 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mInstallation Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mCleaning Failed!\033[m"
		
	fi 
	cd ..
	echo -e "\033[32mDone!\033[m"
elif [ ${VERSION} = "ompss" ]; then
	cd src
	rm .depend
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --enable-ompss --disable-asm --prefix=${ROOT}/x264 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mClean Failed!\033[m"
		
	fi 
	echo -e "\033[32mCompiling ${VERSION} version\033[m"
	status=$( make version=${VERSION} 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mCompilation Failed!\033[m"
		
	fi 
	echo -e "\033[32mInstalling version\033[m"
	status=$( make version=${VERSION} install 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mInstallation Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mCleaning Failed!\033[m"
		
	fi 
	cd ..
	echo -e "\033[32mDone!\033[m"
elif [ ${VERSION} = "ompss_instr" ]; then
	cd src
	rm .depend
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --enable-ompss --enable-ompss-instrumentation --disable-asm --prefix=${ROOT}/x264 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mClean Failed!\033[m"
		
	fi 
	echo -e "\033[32mCompiling ${VERSION} version\033[m"
	status=$( make version=${VERSION} 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mCompilation Failed!\033[m"
		
	fi 
	echo -e "\033[32mInstalling version\033[m"
	status=$( make version=${VERSION} install 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mInstallation Failed!\033[m"
		
	fi 
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[33mCleaning Failed!\033[m"
		
	fi 
	cd ..
	echo -e "\033[32mDone!\033[m"
else
	echo -e "\033[31m${VERSION} version not supported!\033[m"
fi

