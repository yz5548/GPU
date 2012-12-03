SOURCE=GPU.cu
APP=GPU.app
 
module load cuda cuda_SDK
rm -f $APP
nvcc -arch=compute_20 -code=sm_20 -o $APP $SOURCE
chmod 755 $APP

FOLDER=./tests/SIZE_15
NUM_TEST=10
OUTPUT=graph.out

for test in {0..10}
do
	./$APP 1 256 < $FOLDER/$test.in >& $OUTPUT
	if [[ -s $OUTPUT ]]; then
	  echo "FAIL $test "
	else
	  echo "PASS $test "
	fi	
done 

rm -f $OUTPUT