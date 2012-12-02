SOURCE=GPU.cu
APP=GPU.app
 
module load cuda cuda_SDK
rm -f $APP
nvcc -arch=compute_20 -code=sm_20 -o $APP $SOURCE
chmod 755 $APP

FOLDER=./tests/SIZE_4
NUM_TEST=10
OUTPUT=graph.out

for test in {0..10}
do
	./$APP 1 1 < $FOLDER/$test.in > $OUTPUT
	if diff $OUTPUT $FOLDER/$test.out >/dev/null ; then
	  echo "PASS $test "
	else
	  echo "FAIL $test "
	fi	
done 

rm -f $OUTPUT