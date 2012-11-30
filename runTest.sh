SOURCE=GPU.cu
APP=GPU.app

rm -f $APP
nvcc -c $SOURCE -o $APP

FOLDER=./tests/SIZE_4
NUM_TEST=10
OUTPUT=graph.out

for test in {1..1}
do
	./$APP 1 1 < $FOLDER/$test.in
done 

rm -f $OUTPUT
