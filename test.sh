
BENCHMARKS='blackscholes bodytrack canneal dedup facesim ferret fluidanimate freqmine streamcluster swaptions x264'
#BENCHMARKS=$1
VERSIONS='serial pthreads omp2 ompss omp4'

ACTIONS='compile run'
#ACTIONS=$2

input=simlarge
ncores=2

FAIL_COMPILE=0
FAIL_EXECUTE=0

source env.sh
#these are specific to Minotauro and MN

if [  "${BSC_MACHINE}" == "nvidia" ]; then
	module unload intel
	module load gcc/4.9.1
	unload ompss
	module load ompss/15.06
fi


for action in $ACTIONS; do

case $action in 

"compile")	echo -e "\nTESTING COMPILATION..."
			echo "============================= COMPLATION LOG =============================" > compile_log.err 
			for bench in ${BENCHMARKS}; do
				echo -e "\n\t${bench}"
				for version in ${VERSIONS}; do
					echo "============================= ${bench}-${version} =============================" >> compile_log.err 
					status=$(./build.sh ${bench} ${version} 2>> compile_log.err)
					echo "=================================== Done ======================================" >> compile_log.err
					
					if (echo $status | grep -q -E "Compilation Failed!|Installation Failed!"); then
						printf '\t* %10s\t\033[31mFAILED!\033[m\n' "${version}"
                  let FAIL_COMPILE=FAIL_COMPILE+1
					elif (echo $status | grep -q "version not supported!"); then
						printf '\t* %10s\t\033[33mNOT SUPPORTED!\033[m\n' "${version}"
					else
						printf '\t* %10s\t\033[32mPASSED!\033[m\n' "${version}"
					fi
				done
			done
			echo -e "\nCheck compile_log.err for errors."
			;;

"run")	echo -e "\nTESTING  EXECUTION..."
		echo "============================= EXECUTION LOG =============================" > exec_log.err 
		echo "" > exec_log.err
		for bench in ${BENCHMARKS}; do
		
			echo -e "\n\t${bench}"

			#check if machine is Minotauro and link input, else try to untar them
			if [  "${BSC_MACHINE}" == "nvidia" ]; then
				ln -s /gpfs/scratch/bsc18/bsc18186/parsec-inputs/${bench}/inputs ${ROOT}/${bench}/inputs
			else
				#untar the correct input archive
				cd ${ROOT}/${bench}/inputs 2>> exec_log.err
				tar xvf input_${input}.tar > /dev/null 2>> exec_log.err 
				cd ${ROOT}
			fi
			
			for version in ${VERSIONS}; do
				echo "============================= ${bench}-${version} =============================" >> exec_log.err 
				
				if ! (${ROOT}/${bench}/bench/run.sh ${version} ${input} ${ncores} 1>> exec_log.err 2>> exec_log.err); then
					printf '\t* %10s\t\033[31mFAILED!\033[m\n' "${version}"
                  let FAIL_EXECUTE=FAIL_EXECUTE+1
				else
					printf '\t* %10s\t\033[32mPASSED!\033[m\n' "${version}"
				fi
				
				echo "==================================== Done =====================================" >> exec_log.err
			done
		done
		echo -e "\nCheck exec_log.err for errors."
		;;


*)	echo "Nothing to be done."

esac

done

echo Total compile bencmark fail\(s\): $FAIL_COMPILE
echo Total execute bencmark fail\(s\): $FAIL_EXECUTE

let FAIL_TOTAL=$FAIL_COMPILE+$FAIL_EXECUTE

exit $FAIL_TOTAL

