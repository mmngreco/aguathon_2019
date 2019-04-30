DIR=`dirname "$0"`
xargs -a $DIR/cmd_grid_input.txt -n 1 -P 15 -L 1 -I {} sh -c {}
