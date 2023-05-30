while [ 1 ];
do
    arecord -Dac108 -f S32_LE -r 16000 -d 5 -c 4 sample.wav
    python3 program.py 'sample.wav'  
    sudo rm 'sample.wav'
done