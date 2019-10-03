# Algunos comandos útiles

Paralelizar a partir de un fichero en la misma ubicación llamado `file.txt` que
contiene por cada línea un comando a ejecutar. Paraliliza 15 procesos al mismo
tiempo, procesando de uno en uno.
```
DIR=`dirname "$0"`
xargs -a $DIR/file.txt -n 1 -P 15 -L 1 -I {} sh -c {}
```
Mover todos los archivos que estén en el cwd sin a una carpeta bkp.
```
find ./ -type f -maxdepth 1 -exec mv -t bkp/ {} +
```
Eliminar una gran cantidad de ficheros en paralelo.

```
find . -regextype posix-extended -regex ".*0\.02[2-9]+.h5" | parallel rm

```
Instalar tensorflow en un ubuntu de clouding.io.

```
pip install -I -U https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl

```

Ejemplos de grepeo de logs para conseguir distinta información en ellos.

```
grep -lr "JUMP=72" ./ | xargs grep -l "SCORE_TEST=0.06" | xargs grep -r "COMMAND"
grep -r -l -E "SCORE_TEST=0\.0[1][1-2][0-2]" | xargs grep -r "SCORE_TEST"
```




