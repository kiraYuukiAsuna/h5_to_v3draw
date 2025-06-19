#!/bin/bash
#Designed by shengdian, 2024-11-03
#e.g. sh ./teracmd.sh '/home/penglab/PBserver/BRAINTELL/Projects/HumanNeurons/JLH_Images/1116' '/home/penglab/PBserver/BRAINTELL/Projects/HumanNeurons/JLH_Images/shell/teraconverter64'

indir=$1
teraapp=$2

for sdir in $(ls -d ${indir}/*)
do
	echo $sdir
	
	for sdata in $(ls -f ${sdir}/*v3draw)
	do
		sdname=${sdata##*/}
		echo ${sdname}
		outdir=${sdir}/${sdname%.*}
		if [ ! -d ${outdir} ];then
			mkdir ${outdir}
			${teraapp} -s="${sdata}" -d="${outdir}" --resolutions=012 --width=256 --height=256 --depth=256 --sfmt="Vaa3D raw" --dfmt="TIFF (tiled, 3D)" --libtiff_rowsperstrip=-1 --halve=max
			# exit
		fi
	done
done
