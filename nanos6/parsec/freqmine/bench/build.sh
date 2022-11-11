VERSION=$1

if [ ${VERSION} = "serial" ] || [ ${VERSION} = "omp2" ] || [ ${VERSION} = "ompss" ] || [ ${VERSION} = "ompss_instr" ] || [ ${VERSION} = "omp4" ]; then 
	cd src
	echo -e "\033[32mCleaning directory\033[m"
	status=$( make version=${VERSION} clean 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[34mClean Failed!\033[m"
	fi 
	echo -e "\033[32mCompiling ${VERSION} version\033[m"
	status=$( make version=${VERSION} 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mCompilation Failed!\033[m"
	fi 
	echo -e "\033[32mInstalling ${VERSION} version\033[m"
	status=$( make version=${VERSION} install 2>&1 )
	echo "$status"
	if (echo $status | grep -q "Error"); then
		echo -e "\033[31mInstallation Failed!\033[m"
		exit 1
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
