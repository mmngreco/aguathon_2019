#!/bin/sh
HOST='ftp.example.com'
USER='yourid'
PASSWD='zL8sal97a8mo'
FILE=@0

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
put $FILE
quit
END_SCRIPT
exit 0
