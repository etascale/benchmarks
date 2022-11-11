VERSION=$1

if [ ${VERSION} = "serial" ]; then
	cd src
	status=$(autoreconf -fiv)
	echo -e "\033[32mConfiguring environment\033[m"
	status=$( ./configure --prefix=${ROOT}/bodytrack )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	#cp _libtool libtool
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
	status=$(autoreconf -fiv)
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --enable-threads --prefix=${ROOT}/bodytrack )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	#cp _libtool libtool
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
	status=$(autoreconf -fiv)
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --enable-ompss --prefix=${ROOT}/bodytrack )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	#cp _libtool libtool
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
	status=$(autoreconf -fiv)
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --enable-ompss --enable-ompss_instr --prefix=${ROOT}/bodytrack )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	sed -i 's/performance/instrumentation/g' libtool
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
elif [ ${VERSION} = "omp2" ]; then 
	cd src
	status=$(autoreconf -fiv)
	echo -e "\033[32mConfiguring environment\033[m"
	status=$(	./configure --enable-openmp --prefix=${ROOT}/bodytrack )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mConfiguring Failed!\033[m"
		
	fi 
	#cp _libtool libtool
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

