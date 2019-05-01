#!/bin/sh
HOST='ftp.ita.es'
USER='xxxx'
PASSWD='xxxxx'
FILE=@0

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
put $FILE
quit
END_SCRIPT
exit 0
